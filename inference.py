"""Bounded-window inference: keep the prompt, reset generated text, carry state."""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "SFT"))
from generate import generate
from math_answers import first_math_answer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_spec(name):
    registry = json.loads((ROOT / "configs/checkpoints.json").read_text())
    if name in registry:
        return dict(registry[name])
    return dict(repo_id=name, revision=None, channel="registers", slots=4,
                task="math", chunk_size=128, max_chunks=8, dtype="bfloat16")


def add_model_arguments(parser):
    parser.add_argument("--checkpoint", default="llada-math-registers",
                        help="Alias from configs/checkpoints.json, HF model ID, or local directory")
    parser.add_argument("--revision", help="Override the pinned checkpoint revision")
    parser.add_argument("--code-revision", help="Revision for explicitly trusted remote model code")
    parser.add_argument("--channel", choices=["registers", "memory", "discrete", "none"])
    parser.add_argument("--slots", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--total-tokens", type=int, default=1024)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, help="Denoising steps per chunk; default C/2")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"],
                        help="Denoising autocast precision; weights/bridge use BF16 on GPU")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reset-state", action="store_true", help="Disable state carry between chunks")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Explicitly allow Python code from the selected model repository")


def resolve_spec(args):
    spec = checkpoint_spec(args.checkpoint)
    for name in ("revision", "code_revision", "channel", "slots", "chunk_size", "max_chunks", "dtype"):
        value = getattr(args, name, None)
        if value is not None:
            spec[name] = value
    if spec["channel"] == "none":
        spec["slots"] = 0
    return spec


def load_checkpoint(spec, device="cuda", trust_remote_code=False):
    # Local LLaDA implementation supports register injection and checkpointing.
    import models  # noqa: F401
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA/ROCm device available. Install a GPU PyTorch build; CPU is for tiny tests.")
    kwargs = dict(trust_remote_code=trust_remote_code)
    if spec.get("revision"):
        kwargs["revision"] = spec["revision"]
    if spec.get("code_revision"):
        kwargs["code_revision"] = spec["code_revision"]
    tokenizer = AutoTokenizer.from_pretrained(spec["repo_id"], **kwargs)
    model = AutoModel.from_pretrained(
        spec["repo_id"], torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        **kwargs,
    ).to(device).eval()
    return model, tokenizer


def prompt_ids(tokenizer, prompt, device):
    # Use the same tokenization as the batch-one benchmark loader.
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


@torch.no_grad()
def generate_chunks(model, tokenizer, input_ids, *, channel="registers", slots=4,
                    chunk_size=128, max_chunks=8, total_tokens=1024, block_length=32,
                    steps=None, dtype=torch.bfloat16, temperature=0, task="math",
                    reset_state=False):
    """Yield each decoded chunk. Previous text never enters the next read window.

    Continuous carry reads the last-layer states from a clean previous chunk.
    Discrete carry copies its trailing token IDs. Memory tokens use the same
    inference path as task-trained registers. No-carry SFT has no extra slots.
    """
    if channel not in ("registers", "memory", "discrete", "none"):
        raise ValueError("Unknown carry channel")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("The reference inference loop expects one nonempty prompt")
    if min(chunk_size, max_chunks, total_tokens, block_length) < 1:
        raise ValueError("Generation budgets must be positive")
    if chunk_size % block_length or total_tokens % block_length:
        raise ValueError("Chunk size and token budget must be multiples of block length")
    if channel == "none":
        slots = 0
    elif slots < 1 or slots > chunk_size:
        raise ValueError("Carry slot count must be between one and the chunk size")
    steps = chunk_size // 2 if steps is None else steps
    if steps < 1:
        raise ValueError("Denoising step count must be positive")
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = 126336  # LLaDA mask ID
    placeholders = input_ids.new_full((1, slots), mask_id)
    original = torch.cat((input_ids[:, :1], placeholders, input_ids[:, 1:]), dim=1)
    positions = torch.arange(1, slots + 1, device=input_ids.device)
    continuous = channel in ("registers", "memory")
    state, previous, previous_length = None, None, None
    used, text = 0, ""
    for index in range(max_chunks):
        length = min(chunk_size, total_tokens - used)
        if length <= 0:
            break
        current = original.clone()
        if previous is not None and not reset_state:
            if continuous:
                embeds = model.get_input_embeddings()(previous)
                if state is not None:
                    embeds[:, positions, :] = state.to(embeds.dtype)
                state = model(inputs_embeds=embeds, output_hidden_states=True).hidden_states[-1][:, positions, :].clone()
            elif channel == "discrete":
                tail = previous[:, -previous_length:][:, -slots:]
                if tail.shape[1] < slots:
                    tail = torch.cat((input_ids.new_full((1, slots-tail.shape[1]), mask_id), tail), dim=1)
                current[:, positions] = tail
        output, _ = generate(
            model, current, tokenizer, gen_length=length, block_length=block_length,
            steps=max(1, round(steps * length / chunk_size)), temperature=temperature,
            mask_id=mask_id, register_embeds=state,
            register_positions=positions if continuous else None, autocast_dtype=dtype,
        )
        chunk_text = tokenizer.decode(output[0, -length:], skip_special_tokens=False)
        text += chunk_text
        used += length
        previous, previous_length = output, length
        stopped = (first_math_answer(text) is not None if task == "math"
                   else "<code>" in text and "</code>" in text[text.find("<code>")+6:])
        yield dict(chunk=index, generation=chunk_text, accumulated_generation=text,
                   generated_tokens=used, active_tokens=int(current.shape[1])+length,
                   stopped=stopped)
        if stopped:
            break


def run_options(args, spec):
    return dict(channel=spec["channel"], slots=spec["slots"], chunk_size=spec["chunk_size"],
                max_chunks=spec["max_chunks"], total_tokens=args.total_tokens,
                block_length=args.block_length, steps=args.steps,
                dtype=getattr(torch, spec["dtype"]), temperature=args.temperature,
                task=spec["task"], reset_state=args.reset_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--prompt", default="A shop has 24 pencils, sells 7, then receives 10. How many pencils remain?")
    parser.add_argument("--task", choices=["math", "code"])
    args = parser.parse_args()
    spec = resolve_spec(args)
    if args.task:
        spec["task"] = args.task
    from gsm8k import GSM_SYSTEM_PROMPT
    from human_eval import HUMANEVAL_CODE_TAGS_SYSTEM_PROMPT
    instruction = GSM_SYSTEM_PROMPT if spec["task"] == "math" else HUMANEVAL_CODE_TAGS_SYSTEM_PROMPT
    seed_everything(args.seed)
    model, tokenizer = load_checkpoint(spec, args.device, args.trust_remote_code)
    text = tokenizer.apply_chat_template([{"role": "user", "content": instruction+"\n\n"+args.prompt}],
                                         tokenize=False, add_generation_prompt=True)
    if spec["task"] == "math":
        text += "<reasoning>"
    for event in generate_chunks(model, tokenizer, prompt_ids(tokenizer, text, model.device),
                                 **run_options(args, spec)):
        print(f"\n--- chunk {event['chunk']+1}; active window {event['active_tokens']} tokens ---", flush=True)
        print(event["generation"], flush=True)


if __name__ == "__main__":
    main()
