import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import Trainer
    from transformers import DefaultDataCollator
    _TRANSFORMERS_TRAINER_IMPORT_ERROR = None
except Exception as exc:
    # Chunked SFT imports this module for preprocessing/training helpers, but
    # does not instantiate the Hugging Face Trainer path. Some environments include
    # an apex package without apex.amp, which makes transformers.Trainer fail
    # at import time; defer that failure to the vanilla Trainer path.
    _TRANSFORMERS_TRAINER_IMPORT_ERROR = exc

    class Trainer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "dLLMTrainer requires transformers.Trainer, but importing it failed."
            ) from _TRANSFORMERS_TRAINER_IMPORT_ERROR

    class DefaultDataCollator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "dLLMDataCollator requires transformers.DefaultDataCollator, but importing it failed."
            ) from _TRANSFORMERS_TRAINER_IMPORT_ERROR
import random
import math
from contextlib import nullcontext
from tqdm import tqdm
import pickle
import torch.distributed as dist
import inspect


class LatentSlotHead(nn.Module):
    """v5 latent-slot parameter container: slot_offset + W_slot.

    slot_offset gives per-slot identity added to e_mask at init. W_slot projects
    final-layer hidden states at slot positions back into embedding space, so
    the recurrent update lands on the same manifold layer-0 Q/K/V expects to
    read.
    """

    def __init__(self, d_hidden: int, d_embed: int, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        self.d_hidden = d_hidden
        self.d_embed = d_embed
        self.slot_offset = nn.Parameter(torch.randn(num_slots, d_embed) * 0.02)
        self.W_slot = nn.Linear(d_hidden, d_embed, bias=False)
        nn.init.xavier_uniform_(self.W_slot.weight)

    def project(self, h_slot: torch.Tensor) -> torch.Tensor:
        """Project hidden-state vectors (..., d_hidden) to embedding space."""
        return self.W_slot(h_slot)

    def forward(self, h_slot: torch.Tensor) -> torch.Tensor:
        return self.project(h_slot)


class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        return loss if not return_outputs else (loss, outputs)


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        batch["input_ids"] = noisy_batch.long()
        return batch


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""

CODE_TAGS_SYSTEM_PROMPT = """
You are a coding expert. Solve the programming problem.

Respond in the following format:
<code>
Your Python code here
</code>

Continue the Python source exactly when more space is needed. Do not use
Markdown fences.
"""


def _strip_markdown_code_fence(text):
    """Return raw code from a possibly fenced Markdown code block."""
    text = (text or "").strip()
    if not text.startswith("```"):
        return text

    first_newline = text.find("\n")
    if first_newline == -1:
        return ""
    text = text[first_newline + 1:]

    stripped = text.rstrip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].rstrip()
    return stripped


def _model_accepts_kwarg(model, kwarg):
    """Return whether the underlying model.forward explicitly accepts a keyword."""
    target = getattr(model, "module", model)
    try:
        return kwarg in inspect.signature(target.forward).parameters
    except (TypeError, ValueError):
        return False


def _forward_with_attention_bias(model, attention_bias=None, **kwargs):
    """Route our 4D additive attention bias through the model's supported API."""
    if attention_bias is None:
        return model(**kwargs)
    if _model_accepts_kwarg(model, "attention_bias"):
        return model(**kwargs, attention_bias=attention_bias)
    if _model_accepts_kwarg(model, "attention_mask"):
        return model(**kwargs, attention_mask=attention_bias)
    return model(**kwargs)


def preprocess_dataset(data, tokenizer, max_length, test_split=0.01):
    preprocessed_data = []
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
        trajectory = f"<reasoning>{data[i]['thinking_trajectories'][0]}</reasoning>\n<answer>{data[i]['attempt']}</answer>"
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)
        num_tokens = tokenized_input.shape[0]
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data


def _insert_registers_after_bos(token_ids, reg_token_ids):
    """Insert register token IDs right after BOS (position 0) in a token sequence.

    Returns new token IDs with registers at positions 1..num_registers.
    """
    # token_ids[0] is BOS, insert registers after it
    return torch.cat([
        token_ids[:1],                                      # BOS
        torch.tensor(reg_token_ids, dtype=torch.long),      # registers
        token_ids[1:],                                      # rest of sequence
    ])


def build_chunk_to_prompt_mask(seq_len, prompt_length, num_registers, device, dtype=torch.float):
    """Build attention bias that blocks chunk-completion query positions from attending to prompt-text key positions.

    Sequence layout (front_registers mode):
        [BOS][reg_0..reg_{num_registers-1}][prompt_text][chunk_completion]
         0   1..num_registers              1+nr..P-1    P..L-1

    Allows chunk completion to attend to: BOS (pos 0), registers (pos 1..num_registers),
    other chunk-completion positions (pos prompt_length..seq_len-1).
    Blocks attention to: prompt text (pos 1+num_registers..prompt_length-1).

    Args:
        seq_len: total sequence length L
        prompt_length: number of prompt tokens (including BOS and registers) P
        num_registers: number of register tokens (positions 1..num_registers after BOS)
        device: torch device for the mask
        dtype: float dtype for the mask

    Returns:
        (1, 1, seq_len, seq_len) float tensor — 0 where allowed, -1e4 where blocked.
        The mask broadcasts across batch and head dimensions in LLaDA's forward.
    """
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
    prompt_text_start = 1 + num_registers
    prompt_text_end = prompt_length
    completion_start = prompt_length
    if prompt_text_end > prompt_text_start and completion_start < seq_len:
        mask[0, 0, completion_start:seq_len, prompt_text_start:prompt_text_end] = -1e4
    return mask


def build_chunk_state_subgraph_mask(seq_len, prompt_length, num_registers, device, dtype=torch.float,
                                    num_tail_positions=0):
    """Build attention bias that makes registers+tail+completion a closed subgraph on continuation chunks.

    Sequence layout (front_registers mode, with optional tail region for hybrid/tail variants):
        [BOS][reg_0..reg_{R-1}][tail_0..tail_{T-1}][prompt_text][chunk_completion]
         0   1..1+R            1+R..1+R+T           1+R+T..P-1   P..L-1

    where R = num_registers, T = num_tail_positions. Set T=0 for the registers-only layout.

    On masked continuation chunks, the "state subgraph" is {registers, tail, completion}. Query
    positions inside this subgraph may attend only to the subgraph; BOS and prompt-text keys are
    blocked. This forces cross-chunk information to flow through the continuous register channel
    (R) and/or the discrete tail channel (T).
    """
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
    reg_start = 1
    reg_end = 1 + num_registers
    tail_end = reg_end + num_tail_positions
    prompt_text_start = tail_end
    prompt_text_end = prompt_length
    completion_start = prompt_length

    state_query_slices = []
    if tail_end > reg_start:
        state_query_slices.append(slice(reg_start, tail_end))
    if completion_start < seq_len:
        state_query_slices.append(slice(completion_start, seq_len))

    for query_slice in state_query_slices:
        mask[0, 0, query_slice, 0] = -1e4
        if prompt_text_end > prompt_text_start:
            mask[0, 0, query_slice, prompt_text_start:prompt_text_end] = -1e4

    return mask


def apply_task_mode(register_embeds, mode_token_id, wte_module):
    """Overwrite register slot 0 with the task-mode embedding for the given mode token.

    Used when aux_recon_loss is active: register slot 0 carries a "what task am I doing"
    signal so the model can distinguish predict-next-chunk (WRITE) from reconstruct-prev-chunk
    (READ) on otherwise-identical all-masked-completion inputs.

    Returns a new tensor; does not mutate the input.
    """
    if register_embeds is None:
        return None
    out = register_embeds.clone()
    mode_vec = wte_module(torch.tensor([mode_token_id], device=register_embeds.device))[0]
    out[:, 0, :] = mode_vec.to(out.dtype)
    return out


def route_primary_register_embeds(register_embeds, detach_primary_register_bridge=False):
    """Select the bridge state used by the current-chunk task loss.

    Detaching here preserves the exact forward values and inference capacity
    while removing the task-loss gradient path through the preceding writer
    pass. Auxiliary reconstruction must consume the original live bridge tensor
    separately so it can still train the writer.
    """
    if register_embeds is None or not detach_primary_register_bridge:
        return register_embeds
    return register_embeds.detach()


def build_completion_mask(seq_len, prompt_length, completion_length, device):
    """Mark real completion tokens without inferring padding from token IDs."""
    completion_length = seq_len - prompt_length if completion_length is None else int(completion_length)
    if prompt_length < 0 or completion_length < 0 or prompt_length + completion_length > seq_len:
        raise ValueError(
            "Invalid prompt/completion lengths for sequence: "
            f"prompt={prompt_length}, completion={completion_length}, seq_len={seq_len}"
        )
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    return (positions >= prompt_length) & (positions < prompt_length + completion_length)


def mask_invalid_attention_keys(attention_bias, valid_key_mask, dtype=torch.float):
    """Block padded key positions while preserving any existing attention bias."""
    if valid_key_mask.ndim != 2:
        raise ValueError(f"valid_key_mask must have shape (batch, seq), got {valid_key_mask.shape}")
    if attention_bias is None and bool(valid_key_mask.all()):
        return None
    batch_size, seq_len = valid_key_mask.shape
    if attention_bias is None:
        masked_bias = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            dtype=dtype,
            device=valid_key_mask.device,
        )
    else:
        if attention_bias.shape[-2:] != (seq_len, seq_len):
            raise ValueError(
                f"attention bias shape {attention_bias.shape} is incompatible with seq_len={seq_len}"
            )
        masked_bias = attention_bias.expand(batch_size, -1, -1, -1).clone()
    return masked_bias.masked_fill(~valid_key_mask[:, None, None, :], -1e4)


def scale_loss_for_active_ddp_ranks(loss, is_padding):
    """Average gradients over ranks with real chunks, not synthetic sync padding."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return loss
    active = torch.tensor(
        0.0 if is_padding else 1.0,
        dtype=torch.float32,
        device=loss.device,
    )
    dist.all_reduce(active, op=dist.ReduceOp.SUM)
    if active.item() <= 0:
        raise RuntimeError("DDP chunk step has no active ranks.")
    return loss * (dist.get_world_size() / active)


def apply_reconstruction_read_condition(embeds, prompt_mask, condition, completion_mask=None):
    """Add a training-only read-mode offset to completion query embeddings."""
    completion_positions = ~prompt_mask if completion_mask is None else completion_mask
    completion_positions = completion_positions.unsqueeze(-1)
    return embeds + completion_positions.to(embeds.dtype) * condition.to(embeds.dtype)


def apply_residual_detached_ln_register_carry(bridge_register_embeds, register_embeds_from_prev):
    """Layer-normalized residual carry: LN(bridge_update + stopgrad(previous_state))."""
    if bridge_register_embeds is None or register_embeds_from_prev is None:
        return bridge_register_embeds
    residual = register_embeds_from_prev.detach().to(bridge_register_embeds.dtype)
    return F.layer_norm(
        bridge_register_embeds + residual,
        normalized_shape=(bridge_register_embeds.shape[-1],),
    )


def preprocess_s1k_chunked(data, tokenizer, completion_per_chunk=256, num_registers=1,
                           max_chunks_per_trace=16, test_split=0.01,
                           trace_field=None, prompt_removal=False, front_registers=False,
                           channel_mode="registers", tail_length=0,
                           use_mask_token_for_registers=False,
                           variable_chunk_size=False, min_chunk_size=None,
                           code_target_format="default"):
    """Split each trace into chunks for chunked curriculum training.

    Two independent flags control register placement and continuation template:

    - front_registers: if True, register tokens are inserted at positions 1..num_registers
      (after BOS) in every chunk. If False, registers are embedded in the user message text.
      prompt_removal=True implies front_registers=True (Option A always placed at front).

    - prompt_removal: if True, chunk 1+ uses a minimal "Continue your reasoning." template
      instead of the full question, forcing register dependence (Option A).
      If False, all chunks use the full prompt.

    Combinations:
      front=F, pr=F: legacy v3 (registers in text, full prompt)
      front=T, pr=F: new (registers at pos 1-16, full prompt every chunk) — this experiment
      front=T, pr=T: Option A (registers at pos 1-16, continuation template at chunk 1+)

    channel_mode / tail_length / use_mask_token_for_registers control the carry-channel layout:
      - channel_mode="registers" (default): continuous-register channel only. Layout
        [BOS][R_1..R_nr][prompt][completion], same as the legacy recipe.
      - channel_mode="tail": discrete-tail channel only. Layout
        [BOS][T_1..T_X][prompt][completion], where X=tail_length. On chunk 0 the tail
        slots hold mask tokens; on chunk k>0 they hold the last X token ids of the
        ground-truth completion for chunk k-1. No bridge pass is needed.
      - channel_mode="hybrid": both channels. Layout
        [BOS][R_1..R_nr][T_1..T_X][prompt][completion]. Registers (positions 1..nr)
        are bridge-injected at train time; tail positions (nr+1..nr+X) hold the
        prior-chunk tail token ids.
      - use_mask_token_for_registers=True: replace dedicated <register_i> ids with
        tokenizer.mask_token_id at register positions, so the positions inherit the
        model's pretrained "mask-token prior" instead of a freshly-added token that
        the base model has no prior over.

    Supports two data formats:
    - s1K: fields 'question', 'thinking_trajectories', 'attempt'
    - GPT-5.4 traces: fields 'question', 'gpt54_reasoning_trace', 'solution'
    """
    if code_target_format not in ("default", "code_tags"):
        raise ValueError(f"code_target_format must be one of default|code_tags, got {code_target_format}")
    use_code_tags = code_target_format == "code_tags"

    if channel_mode not in ("registers", "tail", "hybrid"):
        raise ValueError(f"channel_mode must be one of registers|tail|hybrid, got {channel_mode}")
    use_tail = channel_mode in ("tail", "hybrid")
    if use_tail and tail_length <= 0:
        raise ValueError(f"channel_mode={channel_mode} requires tail_length > 0, got {tail_length}")
    if channel_mode == "tail" and num_registers != 0:
        raise ValueError(f"channel_mode=tail requires num_registers=0, got {num_registers}")
    # Resolve mask token id. LLaDA's tokenizer doesn't expose .mask_token_id; fall back to
    # the known LLaDA constant so tail/mask-prior features work with the base model.
    resolved_mask_token_id = tokenizer.mask_token_id
    if resolved_mask_token_id is None:
        resolved_mask_token_id = 126336
    mask_token_id = resolved_mask_token_id

    # prompt_removal implies front placement (Option A always put registers at positions 1-16)
    use_front = front_registers or prompt_removal

    if use_mask_token_for_registers:
        reg_token_ids = [mask_token_id] * num_registers
    else:
        reg_token_ids = [tokenizer.convert_tokens_to_ids(f"<register_{i}>") for i in range(num_registers)]

    if use_tail and prompt_removal:
        raise ValueError("channel_mode=tail|hybrid is not compatible with prompt_removal.")
    if use_tail and not use_front:
        raise ValueError(
            "channel_mode=tail|hybrid requires front_registers (or num_registers=0 for tail). "
            "Tail tokens must live at fixed front positions immediately after the register slots."
        )

    # Tail placeholder for chunk 0 (no prior chunk): all mask tokens.
    tail_placeholder_ids = [mask_token_id] * tail_length if use_tail else []
    num_front = num_registers + (tail_length if use_tail else 0)

    # Auto-detect format if trace_field not specified
    if trace_field is None:
        if "gpt54_reasoning_trace" in data[0]:
            trace_field = "gpt54_reasoning_trace"
        else:
            trace_field = "thinking_trajectories"

    # Build continuation prompt (used for chunk 1+ in prompt_removal mode)
    if prompt_removal:
        cont_content = (
            "Continue the Python source exactly from the previous chunk. Do not restart the answer."
            if use_code_tags else "Continue your reasoning."
        )
        cont_msgs = [{"role": "user", "content": cont_content}]
        cont_text = tokenizer.apply_chat_template(cont_msgs, tokenize=False, add_generation_prompt=True)
        cont_base_ids = tokenizer(cont_text, return_tensors="pt", truncation=False).input_ids.squeeze(0)
        # Insert registers after BOS
        cont_ids = _insert_registers_after_bos(cont_base_ids, reg_token_ids)
        # Append the target prefix to match training format.
        cont_prefix = "<code>\n" if use_code_tags else "<reasoning>"
        cont_prefix_ids = tokenizer(cont_prefix, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
        cont_ids = torch.cat([cont_ids, cont_prefix_ids])
        cont_len = cont_ids.shape[0]
        cont_reg_positions = list(range(1, 1 + num_registers))

    all_traces = []
    skipped = 0

    for i in tqdm(range(len(data)), desc="Preprocessing chunked"):
        question = (CODE_TAGS_SYSTEM_PROMPT if use_code_tags else SYSTEM_PROMPT) + "\n\n" + data[i]["question"]

        # Build trajectory based on data format.
        if use_code_tags:
            if trace_field == "gpt54_reasoning_trace":
                raw_code = data[i].get("solution") or data[i].get("gpt54_reasoning_trace", "")
            else:
                raw_code = data[i].get("attempt", "")
            code = _strip_markdown_code_fence(raw_code)
            if not code.strip():
                skipped += 1
                continue
            trajectory = f"<code>\n{code}\n</code>"
        elif trace_field == "gpt54_reasoning_trace":
            raw_trace = data[i]["gpt54_reasoning_trace"]
            trajectory = f"<reasoning>{raw_trace}</reasoning>"
            if data[i].get("solution") and "<answer>" not in raw_trace:
                trajectory += f"\n<answer>{data[i]['solution']}</answer>"
        else:
            trajectory = f"<reasoning>{data[i]['thinking_trajectories'][0]}</reasoning>\n<answer>{data[i]['attempt']}</answer>"

        # Build full prompt (with or without registers in user message)
        if use_front:
            # Registers go at start of sequence (positions 1..num_registers after BOS)
            user_content = question
        else:
            # Legacy: registers embedded in user message
            register_str = " ".join([f"<register_{i}>" for i in range(num_registers)])
            user_content = f"{question} {register_str}"

        prompt_msgs = [{"role": "user", "content": user_content}]
        response_msgs = [{"role": "assistant", "content": trajectory}]

        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(prompt_msgs + response_msgs, tokenize=False)

        prompt_base_ids = tokenizer(prompt_text, return_tensors="pt", truncation=False).input_ids.squeeze(0)
        full_base_ids = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids.squeeze(0)

        if use_front:
            # Insert registers after BOS in the full prompt
            prompt_ids = _insert_registers_after_bos(prompt_base_ids, reg_token_ids)
            full_ids = _insert_registers_after_bos(full_base_ids, reg_token_ids)
            reg_positions = list(range(1, 1 + num_registers))
        else:
            prompt_ids = prompt_base_ids
            full_ids = full_base_ids
            # Find register positions within the prompt (legacy)
            reg_positions = []
            for rid in reg_token_ids:
                positions = (prompt_ids == rid).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    reg_positions.append(positions[0].item())
            if len(reg_positions) != num_registers:
                skipped += 1
                continue

        prompt_len = prompt_ids.shape[0]
        completion_ids = full_ids[prompt_len:]
        completion_len = completion_ids.shape[0]

        if completion_len == 0:
            skipped += 1
            continue

        # Tail/hybrid layouts splice `tail_length` additional positions in between the
        # register slots and the prompt text. On chunk 0 these are mask tokens; on chunk
        # k>0 they carry the last `tail_length` token ids of chunk k-1's completion.
        effective_tail_length = tail_length if use_tail else 0
        prompt_len_with_tail = prompt_len + effective_tail_length

        # Variable chunking: pick a per-trace chunk_size so that long traces fit within
        # max_chunks_per_trace without truncation, while never going below min_chunk_size.
        # Falls back to fixed completion_per_chunk when variable_chunk_size is False.
        if variable_chunk_size:
            floor = min_chunk_size if min_chunk_size is not None else completion_per_chunk
            n_chunks = min(max_chunks_per_trace, max(1, math.ceil(completion_len / floor)))
            this_completion_per_chunk = max(floor, math.ceil(completion_len / n_chunks))
        else:
            this_completion_per_chunk = completion_per_chunk
            n_chunks = min(math.ceil(completion_len / this_completion_per_chunk), max_chunks_per_trace)

        # Determine max sequence length for padding (consistent within this trace)
        total_chunk_len = prompt_len_with_tail + this_completion_per_chunk

        trace_chunks = []

        for c in range(n_chunks):
            start = c * this_completion_per_chunk
            end = min((c + 1) * this_completion_per_chunk, completion_len)
            chunk_completion = completion_ids[start:end]

            if prompt_removal and c > 0:
                # Chunk 1+: use continuation template (no question)
                chunk_prefix = cont_ids
                chunk_prompt_len = cont_len
                chunk_reg_positions = cont_reg_positions
            else:
                # Chunk 0 (or legacy mode): use full prompt
                chunk_prefix = prompt_ids
                chunk_prompt_len = prompt_len
                chunk_reg_positions = reg_positions

            # Splice tail region between registers and prompt text (tail/hybrid modes).
            # Tail ids come from chunk c-1's completion slice for c>0; for chunk 0 the slots
            # hold mask tokens because no prior chunk exists.
            if use_tail:
                reg_end_idx = 1 + num_registers
                if c == 0:
                    tail_ids = torch.tensor(tail_placeholder_ids, dtype=torch.long)
                else:
                    prev_start = (c - 1) * this_completion_per_chunk
                    prev_end = min(c * this_completion_per_chunk, completion_len)
                    prev_completion_slice = completion_ids[prev_start:prev_end]
                    if prev_completion_slice.shape[0] >= tail_length:
                        tail_ids = prev_completion_slice[-tail_length:]
                    else:
                        # Prior chunk shorter than tail_length: left-pad with mask tokens so the
                        # tail region stays fixed-size.
                        pad_count = tail_length - prev_completion_slice.shape[0]
                        pad_tail = torch.full((pad_count,), mask_token_id, dtype=torch.long)
                        tail_ids = torch.cat([pad_tail, prev_completion_slice])
                chunk_prefix = torch.cat([
                    chunk_prefix[:reg_end_idx],
                    tail_ids,
                    chunk_prefix[reg_end_idx:],
                ])
                chunk_prompt_len = chunk_prompt_len + tail_length

            # Build chunk: prefix + completion slice + padding
            chunk_ids = torch.cat([chunk_prefix, chunk_completion])
            if chunk_ids.shape[0] < total_chunk_len:
                pad_len = total_chunk_len - chunk_ids.shape[0]
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                pad = torch.full((pad_len,), pad_id, dtype=torch.long)
                chunk_ids = torch.cat([chunk_ids, pad])

            trace_chunks.append({
                "input_ids": chunk_ids[:total_chunk_len],
                "prompt_length": chunk_prompt_len,
                "completion_length": chunk_completion.shape[0],
                "register_positions": chunk_reg_positions,
                "chunk_index": c,
            })

        all_traces.append(trace_chunks)

    if skipped > 0:
        print(f"Skipped {skipped} examples (empty completion or prompt too long)")

    random.shuffle(all_traces)
    test_traces = all_traces[:int(len(all_traces) * test_split)]
    train_traces = all_traces[int(len(all_traces) * test_split):]
    return train_traces, test_traces


class ChunkedSFTDataset(torch.utils.data.Dataset):
    """Dataset that returns lists of chunks per trace."""

    def __init__(self, traces):
        self.traces = traces

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        return self.traces[idx]


def preprocess_gpt54_vanilla_with_registers(data, tokenizer, num_registers=16,
                                            max_length=4096, test_split=0.01,
                                            trace_field=None):
    """Preprocess GPT-5.4 traces for vanilla (non-chunked) SFT with always-on registers.

    Each sample = [BOS + <register_0> ... <register_{N-1}> + full_prompt + full_trajectory],
    padded to max_length. This is the "always-on memory" variant: registers are present
    in the sequence, but there's no chunking, no bridge, no prompt removal.

    The model sees the full trace end-to-end with register tokens available for attention.
    The loss is standard diffusion loss on the non-prompt, non-padding tokens.

    Returns list of {"input_ids", "prompt_length"} dicts, where prompt_length includes
    the register tokens (so they are not masked during training).
    """
    reg_token_ids = [tokenizer.convert_tokens_to_ids(f"<register_{i}>") for i in range(num_registers)]

    # Auto-detect format
    if trace_field is None:
        if "gpt54_reasoning_trace" in data[0]:
            trace_field = "gpt54_reasoning_trace"
        else:
            trace_field = "thinking_trajectories"

    preprocessed = []
    skipped = 0

    for item in tqdm(data, desc="Preprocessing vanilla+registers"):
        question = SYSTEM_PROMPT + "\n\n" + item["question"]

        # Build trajectory
        if trace_field == "gpt54_reasoning_trace":
            raw_trace = item["gpt54_reasoning_trace"]
            trajectory = f"<reasoning>{raw_trace}</reasoning>"
            if item.get("solution") and "<answer>" not in raw_trace:
                trajectory += f"\n<answer>{item['solution']}</answer>"
        else:
            trajectory = f"<reasoning>{item['thinking_trajectories'][0]}</reasoning>\n<answer>{item['attempt']}</answer>"

        prompt_msgs = [{"role": "user", "content": question}]
        response_msgs = [{"role": "assistant", "content": trajectory}]

        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(prompt_msgs + response_msgs, tokenize=False)

        prompt_base_ids = tokenizer(prompt_text, return_tensors="pt", truncation=False).input_ids.squeeze(0)
        full_base_ids = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids.squeeze(0)

        # Insert registers after BOS (same as Option A chunked preprocessing)
        prompt_ids = _insert_registers_after_bos(prompt_base_ids, reg_token_ids)
        full_ids = _insert_registers_after_bos(full_base_ids, reg_token_ids)

        prompt_len = prompt_ids.shape[0]  # includes the 16 register tokens

        if full_ids.shape[0] <= prompt_len:
            skipped += 1
            continue

        # Truncate to max_length
        if full_ids.shape[0] > max_length:
            full_ids = full_ids[:max_length]

        # Pad to max_length
        if full_ids.shape[0] < max_length:
            pad_len = max_length - full_ids.shape[0]
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            full_ids = torch.cat([full_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])

        preprocessed.append({
            "input_ids": full_ids[:max_length],
            "prompt_length": prompt_len,
        })

    if skipped > 0:
        print(f"Skipped {skipped} vanilla examples (empty completion)")

    random.shuffle(preprocessed)
    test_data = preprocessed[:int(len(preprocessed) * test_split)]
    train_data = preprocessed[int(len(preprocessed) * test_split):]
    return train_data, test_data


def train_vanilla_step(model, input_ids, prompt_length, mask_token_id,
                       optimizer, accelerator, max_grad_norm):
    """Standard diffusion SFT step on a full sequence (no chunking, no bridge).

    Matches sft_train_vanilla.py's train_step but lives here so the hybrid trainer
    can call it. Note: uses the SAME no-no_grad pattern as train_chunk (see gradient
    flow fix) — keeps the embedding lookup in the autograd graph.

    Args:
        input_ids: (batch_size, seq_len) clean token IDs for the full sequence
        prompt_length: int, number of prompt tokens (INCLUDING any register tokens). Not masked.
        mask_token_id: mask token ID (126336 for LLaDA)

    Returns:
        loss.item()
    """
    unwrapped_model = accelerator.unwrap_model(model)
    b, l = input_ids.shape
    device = input_ids.device

    prompt_mask = torch.arange(l, device=device).unsqueeze(0).expand(b, l) < prompt_length
    # Also don't mask padding (pad tokens are token id 126081 for LLaDA; some code uses 0)
    pad_id = mask_token_id  # avoid matching pad to a real-ish id; use a value that never matches
    # We actually know LLaDA pad is 126081 but 0 also works for our padding
    pad_mask = (input_ids == 0) | (input_ids == 126081)
    dont_mask = prompt_mask | pad_mask

    num_completion_tokens = (~dont_mask).sum()
    if num_completion_tokens == 0:
        return 0.0

    # Random timestep
    t = torch.rand((b,), device=device)
    t = (1 - 1e-3) * t + 1e-3
    mask_indices = torch.rand((b, l), device=device) < t.unsqueeze(1)
    mask_indices = mask_indices & ~dont_mask

    noisy_ids = torch.where(mask_indices, mask_token_id, input_ids)

    optimizer.zero_grad()
    # Use input_ids path (no inputs_embeds trick); vanilla step doesn't need graph-through-register
    outputs = model(input_ids=noisy_ids)
    logits = outputs.logits

    labels = input_ids.clone()
    labels[~mask_indices] = -100

    unscaled_loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
    ).view(b, l)

    t_expanded = t.unsqueeze(1).expand(b, l)
    scaled_loss = unscaled_loss / t_expanded
    loss = scaled_loss.sum() / num_completion_tokens

    accelerator.backward(loss)
    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss.item()


def train_chunk(model, chunk_input_ids, prompt_length, register_positions,
                num_registers, register_embeds_from_prev,
                mask_token_id, optimizer, accelerator, max_grad_norm,
                num_passes=4, prev_chunk_input_ids=None, is_padding=False,
                bridge_loops=1,
                bridge_loss_each_loop=False,
                bridge_loss_reduction="mean",
                force_full_mask_first_pass=False,
                mask_chunk_to_prompt=False,
                mask_chunk_state_subgraph=False,
                aux_recon_loss=False,
                aux_recon_weight=0.0,
                write_mode_id=None,
                read_mode_id=None,
                aux_recon_task_mode=False,
                detach_primary_register_bridge=False,
                num_tail_positions=0,
                prompt_dropout_rate=1.0,
                residual_detached_ln_register_carry=False,
                completion_length=None,
                prev_prompt_length=None,
                prev_completion_length=None):
    """Train one chunk with K (bridge + SFT) passes.

    Each pass:
      1. Bridge forward (live) on PREVIOUS chunk's clean text → extract register
      2. SFT forward on CURRENT chunk with register injected, random timestep t → loss
      3. backward + step

    The bridge always runs on the previous chunk's tokens (not the current chunk),
    matching eval behavior where the register is extracted from the previous
    generation's output.

    Args:
        chunk_input_ids: (batch_size, seq_len) clean token IDs for THIS chunk
        prompt_length: int, number of prompt tokens (not masked)
        register_positions: list of int, positions of register tokens in the chunk
        num_registers: number of register tokens
        register_embeds_from_prev: (batch_size, num_registers, hidden_dim) or None
            For chunk 0, this is None (no previous chunk).
        mask_token_id: mask token ID (126336 for LLaDA)
        optimizer: optimizer
        accelerator: accelerator
        max_grad_norm: gradient clipping
        num_passes: number of (bridge + SFT) passes per chunk
        prev_chunk_input_ids: (batch_size, seq_len) clean token IDs for PREVIOUS chunk.
            Bridge runs on this. None for chunk 0 (no bridge needed).

    Returns:
        (avg_loss, final_register_embeds)
        final_register_embeds is NOT detached — carries gradient for next chunk's loss
    """
    unwrapped_model = accelerator.unwrap_model(model)
    b, l = chunk_input_ids.shape
    device = chunk_input_ids.device

    reg_pos = torch.tensor(register_positions, device=device)
    prompt_mask = torch.arange(l, device=device).unsqueeze(0).expand(b, l) < prompt_length
    completion_mask = build_completion_mask(
        l, prompt_length, completion_length, device
    ).expand(b, -1)
    valid_key_mask = prompt_mask | completion_mask
    prev_completion_mask = None
    prev_valid_key_mask = None
    if prev_chunk_input_ids is not None:
        effective_prev_prompt_length = (
            prompt_length if prev_prompt_length is None else prev_prompt_length
        )
        prev_prompt_mask = (
            torch.arange(l, device=device).unsqueeze(0).expand(b, l)
            < effective_prev_prompt_length
        )
        prev_completion_mask = build_completion_mask(
            l,
            effective_prev_prompt_length,
            prev_completion_length,
            device,
        ).expand(b, -1)
        prev_valid_key_mask = prev_prompt_mask | prev_completion_mask

    total_loss = 0.0
    total_recon_loss = 0.0
    no_sync = model.no_sync if hasattr(model, 'no_sync') else nullcontext

    if bridge_loops < 1:
        raise ValueError("bridge_loops must be >= 1.")

    if bridge_loops > 1 and num_passes != 1:
        raise ValueError("bridge_loops > 1 currently requires num_passes == 1.")

    if bridge_loops > 1 and not force_full_mask_first_pass:
        raise ValueError("bridge_loops > 1 currently requires force_full_mask_first_pass.")

    if mask_chunk_to_prompt and mask_chunk_state_subgraph:
        raise ValueError("mask_chunk_to_prompt and mask_chunk_state_subgraph are mutually exclusive.")

    if aux_recon_loss and not mask_chunk_state_subgraph:
        raise ValueError(
            "aux_recon_loss requires mask_chunk_state_subgraph=True; without CSG the recon pass "
            "leaks prompt text and can trivially reconstruct chunk_{i-1} via the shared prompt prefix."
        )

    if aux_recon_task_mode and not aux_recon_loss:
        raise ValueError("aux_recon_task_mode requires aux_recon_loss=True.")

    if aux_recon_task_mode and (write_mode_id is None or read_mode_id is None):
        raise ValueError(
            "aux_recon_task_mode requires write_mode_id and read_mode_id. The training script "
            "must add and resolve <MODE_WRITE> / <MODE_READ> tokens."
        )

    if mask_chunk_to_prompt or mask_chunk_state_subgraph:
        if num_registers + num_tail_positions <= 0:
            raise ValueError(
                "attention masking requires at least one front-channel position: set "
                "num_registers > 0 (registers/hybrid) or num_tail_positions > 0 (tail)."
            )
        expected_reg_positions = list(range(1, 1 + num_registers))
        if list(register_positions) != expected_reg_positions:
            raise ValueError(
                "attention masking assumes front-register layout with register_positions "
                f"{expected_reg_positions}, got {list(register_positions)}."
            )

    # FSDP-safe embedding lookup: under FSDP, the embedding weight is sharded into 1-D per rank,
    # so calling unwrapped_model.get_input_embeddings() directly raises "'weight' must be 2-D".
    # Instead we route through the FSDP-wrapped model's forward, which triggers the gather hook.
    # Fast path for DDP/single-device stays direct.
    from accelerate import DistributedType as _DistType
    _is_fsdp = accelerator.state.distributed_type == _DistType.FSDP

    def _token_embeds(ids):
        """Return 2-D token embeddings for `ids`, FSDP-safe.

        Under FSDP, runs a no_grad forward to extract hidden_states[0] (the output of the
        embedding layer). The returned tensor has no grad_fn; tied lm_head still flows the
        wte gradient via the downstream loss forward.
        """
        if _is_fsdp:
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True)
                embeds = out.hidden_states[0].clone()
                del out
            return embeds
        return unwrapped_model.get_input_embeddings()(ids)

    # Pre-compute clean embeddings for bridge passes on PREVIOUS chunk. Skipped when
    # num_registers == 0 (e.g. channel_mode=tail) — no continuous channel to extract.
    bridge_clean_embeds = None
    if prev_chunk_input_ids is not None and num_registers > 0:
        with torch.no_grad():
            bridge_clean_embeds = _token_embeds(prev_chunk_input_ids).clone()
    bridge_attention_bias = (
        mask_invalid_attention_keys(None, prev_valid_key_mask)
        if prev_valid_key_mask is not None
        else None
    )

    # Build attention mask for the SFT pass only.
    # Only applied for chunks i>0 (i.e., when there's a previous chunk and registers carry state).
    # Chunk 0 keeps full bidirectional attention since it has no carried registers.
    # The bridge pass always uses full attention regardless of this flag.
    # prompt_dropout_rate gates whether the CSG (or chunk-to-prompt) mask fires on this chunk:
    # rate=1.0 always applies (legacy behavior); rate=0.3 applies on 30% of chunks
    # (the remaining 70% the model sees the prompt normally — a stochastic replacement for
    # the fixed CSG mask that creates posterior-collapse-resistant training signal).
    sft_attention_bias = None
    apply_csg_this_chunk = (
        prompt_dropout_rate >= 1.0 or random.random() < prompt_dropout_rate
    )
    if prev_chunk_input_ids is not None and apply_csg_this_chunk:
        if mask_chunk_state_subgraph:
            sft_attention_bias = build_chunk_state_subgraph_mask(
                seq_len=l,
                prompt_length=prompt_length,
                num_registers=num_registers,
                num_tail_positions=num_tail_positions,
                device=device,
                dtype=torch.float,
            )
        elif mask_chunk_to_prompt:
            sft_attention_bias = build_chunk_to_prompt_mask(
                seq_len=l,
                prompt_length=prompt_length,
                num_registers=num_registers,
                device=device,
                dtype=torch.float,
            )

    # Reconstruction must always be register-only, even when the primary
    # p_block draw leaves the prompt visible for this trace.
    recon_attention_bias = None
    recon_read_condition = None
    if aux_recon_loss and prev_chunk_input_ids is not None:
        recon_attention_bias = build_chunk_state_subgraph_mask(
            seq_len=l,
            prompt_length=prompt_length,
            num_registers=num_registers,
            num_tail_positions=num_tail_positions,
            device=device,
            dtype=torch.float,
        )
        if not aux_recon_task_mode:
            # Capacity-matched read condition: reconstruction completion
            # queries receive one extra detached mask embedding. This gives the
            # shared decoder an explicit training-only mode signal without
            # consuming a register slot, adding vocabulary, or changing
            # inference.
            with torch.no_grad():
                condition_id = torch.tensor([[mask_token_id]], device=device)
                recon_read_condition = _token_embeds(condition_id).detach()

    def compute_sft_loss(register_embeds, fully_mask_completion, target_ids=None,
                         attention_bias=None, reconstruction_mode=False,
                         target_completion_mask=None, target_valid_key_mask=None):
        """Run the masked SFT pass for the current chunk with the provided register state.

        target_ids: if provided, used as both the noisy_ids source AND the label source
            (used by the reconstruction aux loss to predict prev-chunk tokens instead of
            current-chunk tokens). Must share layout with chunk_input_ids — same prompt_length,
            same total length. Defaults to chunk_input_ids (primary loss behavior).
        """
        tgt = chunk_input_ids if target_ids is None else target_ids
        active_completion_mask = (
            completion_mask if target_completion_mask is None else target_completion_mask
        )
        active_valid_key_mask = (
            valid_key_mask if target_valid_key_mask is None else target_valid_key_mask
        )
        if fully_mask_completion:
            t = torch.ones((b,), device=device)
        else:
            t = torch.rand((b,), device=device)
            t = (1 - 1e-3) * t + 1e-3

        mask_indices = torch.rand((b, l), device=device) < t.unsqueeze(1)
        mask_indices = mask_indices & active_completion_mask

        noisy_ids = torch.where(mask_indices, mask_token_id, tgt)

        embeds = _token_embeds(noisy_ids).clone()
        if register_embeds is not None:
            embeds[:, reg_pos, :] = register_embeds.to(embeds.dtype)
        if reconstruction_mode:
            if recon_read_condition is None:
                raise RuntimeError("Reconstruction read condition was not initialized.")
            embeds = apply_reconstruction_read_condition(
                embeds,
                prompt_mask,
                recon_read_condition,
                completion_mask=active_completion_mask,
            )

        effective_attention_bias = mask_invalid_attention_keys(
            attention_bias,
            active_valid_key_mask,
        )
        outputs = _forward_with_attention_bias(
            model,
            inputs_embeds=embeds,
            attention_bias=effective_attention_bias,
        )
        logits = outputs.logits

        labels = tgt.clone()
        labels[~mask_indices] = -100

        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(b, l)

        t_expanded = t.unsqueeze(1).expand(b, l)
        scaled_loss = unscaled_loss / t_expanded
        if is_padding:
            return logits.sum() * 0.0
        num_completion_tokens = active_completion_mask.sum()
        if num_completion_tokens > 0:
            return scaled_loss.sum() / num_completion_tokens
        return logits.sum() * 0.0

    if bridge_loops > 1:
        optimizer.zero_grad()

        # Chunk 0 has no previous chunk to refine from, so keep the original single-pass behavior.
        effective_loops = bridge_loops if bridge_clean_embeds is not None else 1
        loop_reg_input = register_embeds_from_prev
        loop_losses = []
        final_register_embeds = None

        for loop_idx in range(effective_loops):
            if bridge_clean_embeds is not None:
                bridge_embeds = bridge_clean_embeds.clone()
                if loop_reg_input is not None:
                    bridge_embeds[:, reg_pos, :] = loop_reg_input.to(bridge_embeds.dtype)
                bridge_context = (
                    torch.no_grad()
                    if detach_primary_register_bridge and not aux_recon_loss
                    else nullcontext()
                )
                with bridge_context:
                    bridge_out = _forward_with_attention_bias(
                        model,
                        inputs_embeds=bridge_embeds,
                        output_hidden_states=True,
                        attention_bias=bridge_attention_bias,
                    )
                bridge_register_embeds = bridge_out.hidden_states[-1][:, reg_pos, :]
                del bridge_out
                if residual_detached_ln_register_carry:
                    register_embeds = apply_residual_detached_ln_register_carry(
                        bridge_register_embeds,
                        loop_reg_input,
                    )
                else:
                    register_embeds = bridge_register_embeds
                final_register_embeds = register_embeds
                loop_reg_input = register_embeds
            else:
                register_embeds = None
                final_register_embeds = None

            if bridge_loss_each_loop or loop_idx == effective_loops - 1:
                primary_reg_embeds = route_primary_register_embeds(
                    register_embeds,
                    detach_primary_register_bridge=detach_primary_register_bridge,
                )
                # Optional legacy mode bit. memory-token's capacity-matched control leaves
                # this disabled so all register slots remain available.
                if aux_recon_task_mode and primary_reg_embeds is not None:
                    primary_reg_embeds = apply_task_mode(
                        primary_reg_embeds, write_mode_id, unwrapped_model.get_input_embeddings()
                    )
                loop_losses.append(
                    compute_sft_loss(
                        register_embeds=primary_reg_embeds,
                        fully_mask_completion=force_full_mask_first_pass,
                        attention_bias=sft_attention_bias,
                    )
                )

        if not loop_losses:
            loss = torch.zeros((), device=device)
        elif bridge_loss_reduction == "sum":
            loss = torch.stack(loop_losses).sum()
        elif bridge_loss_reduction == "mean":
            loss = torch.stack(loop_losses).mean()
        else:
            raise ValueError(f"Unsupported bridge_loss_reduction: {bridge_loss_reduction}")

        # Aux reconstruction loss — uses the final loop's bridged registers with READ mode at slot 0
        recon_loss_value = 0.0
        if aux_recon_loss and final_register_embeds is not None:
            assert prev_chunk_input_ids.shape == chunk_input_ids.shape
            read_reg_embeds = final_register_embeds
            if aux_recon_task_mode:
                read_reg_embeds = apply_task_mode(
                    read_reg_embeds, read_mode_id, unwrapped_model.get_input_embeddings()
                )
            recon = compute_sft_loss(
                register_embeds=read_reg_embeds,
                fully_mask_completion=True,
                target_ids=prev_chunk_input_ids,
                attention_bias=recon_attention_bias,
                reconstruction_mode=not aux_recon_task_mode,
                target_completion_mask=prev_completion_mask,
                target_valid_key_mask=prev_valid_key_mask,
            )
            loss = loss + aux_recon_weight * recon
            recon_loss_value = recon.item()

        loss = scale_loss_for_active_ddp_ranks(loss, is_padding)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        final_reg = final_register_embeds.detach() if final_register_embeds is not None else None
        return loss.item(), final_reg, recon_loss_value

    for p in range(num_passes):
        # 1. Bridge forward on PREVIOUS chunk (live) → register with gradient
        if bridge_clean_embeds is not None:
            with no_sync():
                bridge_embeds = bridge_clean_embeds.clone()
                if register_embeds_from_prev is not None:
                    bridge_embeds[:, reg_pos, :] = register_embeds_from_prev.to(bridge_embeds.dtype)
                bridge_requires_grad = (
                    not detach_primary_register_bridge
                    or (aux_recon_loss and p == num_passes - 1)
                )
                bridge_context = nullcontext() if bridge_requires_grad else torch.no_grad()
                with bridge_context:
                    bridge_out = _forward_with_attention_bias(
                        model,
                        inputs_embeds=bridge_embeds,
                        output_hidden_states=True,
                        attention_bias=bridge_attention_bias,
                    )
                bridge_register_embeds = bridge_out.hidden_states[-1][:, reg_pos, :]
                del bridge_out
                if residual_detached_ln_register_carry:
                    register_embeds = apply_residual_detached_ln_register_carry(
                        bridge_register_embeds,
                        register_embeds_from_prev,
                    )
                else:
                    register_embeds = bridge_register_embeds
        else:
            # Chunk 0: no previous chunk, no bridge, no register injection
            register_embeds = None

        optimizer.zero_grad()

        primary_reg_embeds = route_primary_register_embeds(
            register_embeds,
            detach_primary_register_bridge=detach_primary_register_bridge,
        )
        # Optional legacy mode bit. The capacity-matched memory-token path does not use it.
        if aux_recon_task_mode and primary_reg_embeds is not None:
            primary_reg_embeds = apply_task_mode(
                primary_reg_embeds, write_mode_id, unwrapped_model.get_input_embeddings()
            )

        loss = compute_sft_loss(
            register_embeds=primary_reg_embeds,
            fully_mask_completion=(force_full_mask_first_pass and p == 0),
            attention_bias=sft_attention_bias,
        )

        # Aux reconstruction loss (fires once per chunk on the last pass).
        # The capacity-matched path uses a training-only completion-query
        # condition; the optional legacy path uses READ mode at register slot 0.
        # CSG leaves the bridged registers as the only source-information path.
        recon_loss_value = 0.0
        run_reconstruction = (
            aux_recon_loss
            and p == num_passes - 1
            and register_embeds is not None
        )
        if run_reconstruction and detach_primary_register_bridge:
            # The primary path cannot reach the live bridge tensor, so its
            # backward is independent of reconstruction. Release the primary
            # graph before constructing the reconstruction decoder graph; keeping
            # both resident can exceed accelerator memory despite identical
            # gradients and one shared optimizer step.
            scaled_primary_loss = scale_loss_for_active_ddp_ranks(loss, is_padding)
            accelerator.backward(scaled_primary_loss)
            primary_loss_value = loss.item()

            assert prev_chunk_input_ids.shape == chunk_input_ids.shape, (
                "aux_recon_loss requires prev and current chunks share shape "
                f"(got {prev_chunk_input_ids.shape} vs {chunk_input_ids.shape})"
            )
            read_reg_embeds = register_embeds
            if aux_recon_task_mode:
                read_reg_embeds = apply_task_mode(
                    read_reg_embeds, read_mode_id, unwrapped_model.get_input_embeddings()
                )
            recon = compute_sft_loss(
                register_embeds=read_reg_embeds,
                fully_mask_completion=True,
                target_ids=prev_chunk_input_ids,
                attention_bias=recon_attention_bias,
                reconstruction_mode=not aux_recon_task_mode,
                target_completion_mask=prev_completion_mask,
                target_valid_key_mask=prev_valid_key_mask,
            )
            recon_loss_value = recon.item()
            weighted_recon = aux_recon_weight * recon
            scaled_recon = scale_loss_for_active_ddp_ranks(weighted_recon, is_padding)
            accelerator.backward(scaled_recon)
            loss_value = primary_loss_value + aux_recon_weight * recon_loss_value
        elif run_reconstruction:
            assert prev_chunk_input_ids.shape == chunk_input_ids.shape, (
                "aux_recon_loss requires prev and current chunks share shape "
                f"(got {prev_chunk_input_ids.shape} vs {chunk_input_ids.shape})"
            )
            read_reg_embeds = register_embeds
            if aux_recon_task_mode:
                read_reg_embeds = apply_task_mode(
                    read_reg_embeds, read_mode_id, unwrapped_model.get_input_embeddings()
                )
            recon = compute_sft_loss(
                register_embeds=read_reg_embeds,
                fully_mask_completion=True,
                target_ids=prev_chunk_input_ids,
                attention_bias=recon_attention_bias,
                reconstruction_mode=not aux_recon_task_mode,
                target_completion_mask=prev_completion_mask,
                target_valid_key_mask=prev_valid_key_mask,
            )
            loss = loss + aux_recon_weight * recon
            recon_loss_value = recon.item()
            loss = scale_loss_for_active_ddp_ranks(loss, is_padding)
            accelerator.backward(loss)
            loss_value = loss.item()
        else:
            # 3. backward — each pass is independent.
            # Gradient flows: SFT loss → register → bridge → model weights
            loss = scale_loss_for_active_ddp_ranks(loss, is_padding)
            accelerator.backward(loss)
            loss_value = loss.item()

        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss_value
        total_recon_loss += recon_loss_value

        # NOTE: do NOT update register_embeds_from_prev here.
        # All passes in the same chunk should use the same reg_0 input to the bridge.
        # The bridge output (register_embeds) changes between passes because model
        # weights change, but the INPUT to the bridge stays constant — this reinforces
        # that the register should be globally useful across all timesteps t.

    # Return the last bridge output (register_embeds) as the register for the next chunk.
    # This is final_reg_0**** in the notation — the bridge output from the last pass,
    # which is model(prev_chunk_text, reg_input) with the latest weights.
    # For chunk 0 (no bridge), register_embeds is None, so next chunk gets None
    # and its bridge will use reg_0 (default embedding) as input.
    #
    # Detach since the graph was freed by the last pass's backward.
    final_reg = register_embeds.detach() if register_embeds is not None else None
    # Recon loss is fired at most once per chunk (last pass); normalize by 1 not num_passes
    avg_recon = total_recon_loss
    return total_loss / num_passes, final_reg, avg_recon


def train_chunk_accumulate(model, chunk_input_ids, prompt_length, register_positions,
                           num_registers, register_embeds_from_prev,
                           mask_token_id, accelerator,
                           num_passes=4, prev_chunk_input_ids=None, is_padding=False,
                           force_full_mask_first_pass=False,
                           mask_chunk_state_subgraph=False,
                           num_tail_positions=0,
                           prompt_dropout_rate=1.0,
                           residual_detached_ln_register_carry=False,
                           completion_length=None,
                           prev_prompt_length=None,
                           prev_completion_length=None):
    """BPTT variant of train_chunk: runs the bridge + M SFT passes on one chunk but
    does NOT call backward/step. Returns the summed loss (graph-alive) and the
    final register embeddings (graph-alive) so the caller can keep gradient flowing
    across multiple chunks before a single backward.

    Key differences vs train_chunk:
      * Bridge pass runs ONCE per chunk, not inside the M loop. With no optimizer
        step between passes, the bridge output is identical for all M passes
        anyway; running it once is cheaper and preserves semantics.
      * No optimizer.zero_grad / backward / step. Caller is responsible.
      * Strips aux_recon_loss / bridge_loops features; this path only supports
        the baseline CSG+FFM setup that our experiments use.

    Returns:
        (chunk_loss, final_register_embeds, num_effective_passes)
        chunk_loss: scalar tensor summed over M passes, graph alive
        final_register_embeds: (batch, num_registers, hidden_dim) or None, graph alive
        num_effective_passes: M (useful for normalization at caller)
    """
    unwrapped_model = accelerator.unwrap_model(model)
    b, l = chunk_input_ids.shape
    device = chunk_input_ids.device

    reg_pos = torch.tensor(register_positions, device=device)
    prompt_mask = torch.arange(l, device=device).unsqueeze(0).expand(b, l) < prompt_length
    completion_mask = build_completion_mask(
        l, prompt_length, completion_length, device
    ).expand(b, -1)
    valid_key_mask = prompt_mask | completion_mask
    num_completion_tokens = completion_mask.sum()
    prev_valid_key_mask = None
    if prev_chunk_input_ids is not None:
        effective_prev_prompt_length = (
            prompt_length if prev_prompt_length is None else prev_prompt_length
        )
        prev_prompt_mask = (
            torch.arange(l, device=device).unsqueeze(0).expand(b, l)
            < effective_prev_prompt_length
        )
        prev_completion_mask = build_completion_mask(
            l,
            effective_prev_prompt_length,
            prev_completion_length,
            device,
        ).expand(b, -1)
        prev_valid_key_mask = prev_prompt_mask | prev_completion_mask

    if mask_chunk_state_subgraph:
        if num_registers + num_tail_positions <= 0:
            raise ValueError(
                "BPTT with mask_chunk_state_subgraph requires at least one front-channel position "
                "(num_registers > 0 or num_tail_positions > 0)."
            )
        expected_reg_positions = list(range(1, 1 + num_registers))
        if list(register_positions) != expected_reg_positions:
            raise ValueError(
                f"CSG mask assumes front-register layout {expected_reg_positions}, "
                f"got {list(register_positions)}."
            )

    sft_attention_bias = None
    apply_csg_this_chunk = (
        prompt_dropout_rate >= 1.0 or random.random() < prompt_dropout_rate
    )
    if prev_chunk_input_ids is not None and mask_chunk_state_subgraph and apply_csg_this_chunk:
        sft_attention_bias = build_chunk_state_subgraph_mask(
            seq_len=l, prompt_length=prompt_length,
            num_registers=num_registers,
            num_tail_positions=num_tail_positions,
            device=device, dtype=torch.float,
        )

    # 1. Bridge pass — run ONCE per chunk (graph alive). Bridge uses CLEAN prev-chunk
    #    text (matches inference). register_embeds_from_prev is the previous window's
    #    detached reg (or the current window's still-graph-alive reg mid-window).
    #    Skipped for tail-only channel (num_registers == 0): no continuous state to extract.
    if prev_chunk_input_ids is not None and num_registers > 0:
        bridge_clean_embeds = unwrapped_model.get_input_embeddings()(prev_chunk_input_ids).clone()
        if register_embeds_from_prev is not None:
            bridge_clean_embeds = bridge_clean_embeds.clone()
            bridge_clean_embeds[:, reg_pos, :] = register_embeds_from_prev.to(bridge_clean_embeds.dtype)
        bridge_attention_bias = mask_invalid_attention_keys(None, prev_valid_key_mask)
        bridge_out = _forward_with_attention_bias(
            model,
            inputs_embeds=bridge_clean_embeds,
            output_hidden_states=True,
            attention_bias=bridge_attention_bias,
        )
        bridge_register_embeds = bridge_out.hidden_states[-1][:, reg_pos, :]
        if residual_detached_ln_register_carry:
            register_embeds = apply_residual_detached_ln_register_carry(
                bridge_register_embeds,
                register_embeds_from_prev,
            )
        else:
            register_embeds = bridge_register_embeds
    else:
        register_embeds = None

    # 2. M SFT passes with different noise realizations, all using the shared register_embeds
    chunk_loss = torch.zeros((), device=device)
    for p in range(num_passes):
        fully_masked = force_full_mask_first_pass and p == 0
        if fully_masked:
            t = torch.ones((b,), device=device)
        else:
            t = torch.rand((b,), device=device)
            t = (1 - 1e-3) * t + 1e-3

        mask_indices = torch.rand((b, l), device=device) < t.unsqueeze(1)
        mask_indices = mask_indices & completion_mask
        noisy_ids = torch.where(mask_indices, mask_token_id, chunk_input_ids)

        embeds = unwrapped_model.get_input_embeddings()(noisy_ids).clone()
        if register_embeds is not None:
            embeds[:, reg_pos, :] = register_embeds.to(embeds.dtype)

        effective_attention_bias = mask_invalid_attention_keys(
            sft_attention_bias,
            valid_key_mask,
        )
        outputs = _forward_with_attention_bias(
            model,
            inputs_embeds=embeds,
            attention_bias=effective_attention_bias,
        )
        logits = outputs.logits

        labels = chunk_input_ids.clone()
        labels[~mask_indices] = -100

        unscaled = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(b, l)
        scaled = unscaled / t.unsqueeze(1).expand(b, l)

        if is_padding:
            pass_loss = logits.sum() * 0.0
        elif num_completion_tokens > 0:
            pass_loss = scaled.sum() / num_completion_tokens
        else:
            pass_loss = logits.sum() * 0.0

        chunk_loss = chunk_loss + pass_loss

    return chunk_loss, register_embeds, num_passes
