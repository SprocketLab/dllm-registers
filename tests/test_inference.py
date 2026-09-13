"""CPU-only protocol tests; no checkpoint downloads or GPU allocation."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from inference import checkpoint_spec, generate_chunks
from generate import generate


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)
        self.config = SimpleNamespace(mask_token_id=3)
        self.write_inputs = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, output_hidden_states=False):
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        self.write_inputs.append(hidden.detach().clone())
        # A deterministic non-mask prediction, with nontrivial hidden states.
        logits = hidden.new_zeros((*hidden.shape[:2], 64))
        logits[..., 9] = 10
        return SimpleNamespace(logits=logits, hidden_states=(hidden + 1,))


class TinyTokenizer:
    mask_token_id = 3

    def decode(self, ids, **kwargs):
        return " ".join(str(value) for value in ids.tolist())


def options(**overrides):
    return dict(dict(slots=2, chunk_size=8, block_length=4, max_chunks=3,
                     total_tokens=24, steps=4), **overrides)


@pytest.mark.parametrize("channel", ["registers", "memory", "discrete", "none"])
def test_window_does_not_grow(channel):
    model = TinyModel()
    events = list(generate_chunks(model, TinyTokenizer(), torch.tensor([[1, 4, 5]]),
                                  channel=channel, **options()))
    expected = 3 + (0 if channel == "none" else 2) + 8
    assert len(events) == 3
    assert [event["active_tokens"] for event in events] == [expected] * 3
    assert events[-1]["generated_tokens"] == 24


@pytest.mark.parametrize("channel", ["registers", "memory"])
def test_continuous_write_carries_hidden_state_not_text(channel):
    model = TinyModel()
    calls = []

    def fake_generate(model, prompt, tokenizer, **kwargs):
        calls.append((prompt.clone(), kwargs["register_embeds"]))
        continuation = prompt.new_full((1, kwargs["gen_length"]), 9)
        return torch.cat((prompt, continuation), dim=1), None

    with patch("inference.generate", fake_generate):
        list(generate_chunks(model, TinyTokenizer(), torch.tensor([[1, 4, 5]]),
                             channel=channel, **options()))
    assert calls[0][1] is None
    assert calls[1][1].shape == (1, 2, 8)
    assert not calls[1][1].requires_grad
    assert len(model.write_inputs) == 2
    assert all(torch.equal(prompt, torch.tensor([[1, 3, 3, 4, 5]])) for prompt, _ in calls)
    torch.testing.assert_close(calls[2][1], calls[1][1] + 1)


def test_discrete_tail_is_the_only_text_carried():
    calls = []

    def fake_generate(model, prompt, tokenizer, **kwargs):
        calls.append(prompt.clone())
        return torch.cat((prompt, torch.arange(10, 18).unsqueeze(0)), dim=1), None

    with patch("inference.generate", fake_generate):
        list(generate_chunks(TinyModel(), TinyTokenizer(), torch.tensor([[1, 4, 5]]),
                             channel="discrete", **options()))
    assert calls[0].tolist() == [[1, 3, 3, 4, 5]]
    assert calls[1].tolist() == [[1, 16, 17, 4, 5]]
    assert calls[2].tolist() == calls[1].tolist()


def test_reset_disables_continuous_write():
    model = TinyModel()
    calls = []

    def fake_generate(model, prompt, tokenizer, **kwargs):
        calls.append(kwargs["register_embeds"])
        return torch.cat((prompt, prompt.new_full((1, 8), 9)), dim=1), None

    with patch("inference.generate", fake_generate):
        list(generate_chunks(model, TinyTokenizer(), torch.tensor([[1, 4, 5]]),
                             reset_state=True, **options()))
    assert calls == [None, None, None]
    assert not model.write_inputs


def test_wrong_first_answer_stops_without_checking_target():
    tokenizer = TinyTokenizer()
    tokenizer.decode = lambda ids, **kwargs: r"Answer: \boxed{999}"
    events = list(generate_chunks(TinyModel(), tokenizer, torch.tensor([[1, 4, 5]]), **options()))
    assert len(events) == 1
    assert events[0]["stopped"]


def test_final_partial_chunk_obeys_total_budget():
    events = list(generate_chunks(TinyModel(), TinyTokenizer(), torch.tensor([[1, 4, 5]]),
                                  **options(total_tokens=12)))
    assert len(events) == 2
    assert events[-1]["generated_tokens"] == 12
    assert events[-1]["active_tokens"] == 9


def test_generate_preserves_prompt_mask_slots():
    prompt = torch.tensor([[1, 3, 3, 4]])
    output, state = generate(TinyModel(), prompt, TinyTokenizer(), steps=4, gen_length=8,
                             block_length=4, mask_id=3, register_positions=torch.tensor([1, 2]))
    assert torch.equal(output[:, :4], prompt)
    assert (output[:, 4:] == 9).all()
    assert state.shape == (1, 2, 8)


def test_checkpoint_aliases_encode_protocol_and_immutable_revisions():
    for backbone in ("llada", "dream"):
        for task in ("math", "code"):
            for method in ("registers", "discrete", "memory", "sft"):
                spec = checkpoint_spec(f"{backbone}-{task}-{method}")
                assert len(spec["revision"]) == 40
                assert spec["chunk_size"] * spec["max_chunks"] == 1024
                assert spec["slots"] == (0 if method == "sft" else 4)
                assert spec["dtype"] == ("float16" if backbone == "dream" and task == "math" else "bfloat16")


def test_local_llada_forward_and_register_injection():
    from models import LLaDAConfig, LLaDAModelLM
    config = LLaDAConfig(d_model=32, n_heads=4, n_layers=2, mlp_ratio=2,
                         vocab_size=64, embedding_size=64, max_sequence_length=64,
                         block_type="llama", activation_type="silu", mask_token_id=3,
                         init_device="cpu", rope=True)
    model = LLaDAModelLM(config, init_params=True).eval()
    ids = torch.tensor([[1, 3, 3, 4, 5, 3]])
    output = model(ids, output_hidden_states=True)
    embeds = model.get_input_embeddings()(ids)
    embeds[:, 1:3] = output.hidden_states[-1][:, 1:3]
    carried = model(inputs_embeds=embeds, output_hidden_states=True)
    assert carried.logits.shape == (1, 6, 64)
    assert torch.isfinite(carried.logits).all()
    carried.logits.sum().backward()
    assert model.get_input_embeddings().weight.grad is not None


def test_tiny_local_checkpoint_runs_demo_end_to_end(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from models import LLaDAConfig, LLaDAModelLM

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    config = LLaDAConfig(d_model=32, n_heads=4, n_layers=1, mlp_ratio=2,
                         vocab_size=64, embedding_size=64, max_sequence_length=128,
                         block_type="llama", activation_type="silu", rope=True,
                         mask_token_id=3, init_device="cpu")
    LLaDAModelLM(config, init_params=True).save_pretrained(tmp_path)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "[MASK]": 3}
    vocab.update({f"word{i}": i for i in range(4, 64)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                        bos_token="[BOS]", unk_token="[UNK]", mask_token="[MASK]")
    tokenizer.chat_template = "[BOS] {% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(tmp_path)
    run = subprocess.run([sys.executable, "inference.py", "--checkpoint", str(tmp_path),
                          "--device", "cpu", "--chunk-size", "8", "--block-length", "4",
                          "--max-chunks", "2", "--total-tokens", "16", "--prompt", "word4"],
                         cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "chunk 1;" in run.stdout
    assert "chunk 2;" in run.stdout
