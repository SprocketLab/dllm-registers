"""Full-sequence SFT with full fine-tuning on reasoning traces.

Clean comparison baseline: same data, same model, same epochs,
but full-sequence training (no chunking, no registers).
Uses FSDP for multi-GPU full FT.
"""

import argparse
import json
import os
import random
import math
import sys
import glob
import inspect
import re
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedType
from transformers import AutoTokenizer, AutoModel, get_constant_schedule_with_warmup
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

# Register dllm's LLaDA model (gradient checkpointing support) before any AutoModel calls.
sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402,F401


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


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess_gpt54(data, tokenizer, max_length=4096, test_split=0.01,
                     pad_to_max_length=True, max_completion_length=0,
                     code_target_format="default"):
    """Preprocess GPT-5.4 traces for standard (non-chunked) SFT."""
    if code_target_format not in ("default", "code_tags"):
        raise ValueError(
            f"code_target_format must be one of default|code_tags, got {code_target_format}"
        )
    use_code_tags = code_target_format == "code_tags"
    preprocessed = []
    skipped = 0

    for item in tqdm(data, desc="Preprocessing"):
        system_prompt = CODE_TAGS_SYSTEM_PROMPT if use_code_tags else SYSTEM_PROMPT
        question = system_prompt + "\n\n" + item["question"]
        if use_code_tags:
            raw_code = item.get("solution") or item.get("gpt54_reasoning_trace", "")
            code = _strip_markdown_code_fence(raw_code)
            if not code.strip():
                skipped += 1
                continue
            trajectory = f"<code>\n{code}\n</code>"
        else:
            raw_trace = item["gpt54_reasoning_trace"]
            trajectory = f"<reasoning>{raw_trace}</reasoning>"
            if item.get("solution") and "<answer>" not in raw_trace:
                trajectory += f"\n<answer>{item['solution']}</answer>"

        prompt_msgs = [{"role": "user", "content": question}]
        response_msgs = [{"role": "assistant", "content": trajectory}]

        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(prompt_msgs + response_msgs, tokenize=False)

        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.squeeze(0)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.squeeze(0)

        prompt_len = prompt_ids.shape[0]
        if max_completion_length > 0:
            full_ids = full_ids[:prompt_len + max_completion_length]
        if full_ids.shape[0] <= prompt_len:
            skipped += 1
            continue

        # The legacy trainer padded every batch-1 example to max_length. This is
        # unnecessary for batch size 1 and is especially wasteful for mix60k
        # (roughly 500 tokens including the prompt on average). Keep it available
        # for exact legacy reproduction, but allow dynamic-length training.
        if pad_to_max_length and full_ids.shape[0] < max_length:
            pad_len = max_length - full_ids.shape[0]
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            full_ids = torch.cat([full_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])

        preprocessed.append({
            "input_ids": full_ids[:max_length],
            "prompt_length": prompt_len,
        })

    if skipped > 0:
        print(f"Skipped {skipped} examples (empty completion)")

    random.shuffle(preprocessed)
    test_data = preprocessed[:int(len(preprocessed) * test_split)]
    train_data = preprocessed[int(len(preprocessed) * test_split):]
    return train_data, test_data


def train_step(model, input_ids, prompt_length, mask_token_id, pad_token_id, optimizer, accelerator, max_grad_norm):
    """Standard diffusion SFT: one forward pass with random masking."""
    unwrapped_model = accelerator.unwrap_model(model)
    b, l = input_ids.shape
    device = input_ids.device

    prompt_mask = torch.arange(l, device=device).unsqueeze(0).expand(b, l) < prompt_length
    # Also mask padding
    pad_mask = input_ids == pad_token_id
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
    try:
        outputs = model(input_ids=noisy_ids, use_cache=False)
    except TypeError:
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


def copy_dynamic_model_code(objects, output_dir):
    """Package trusted remote-model Python modules with a saved checkpoint.

    Transformers copies some AutoClass files during ``save_pretrained`` but
    Dream checkpoints also need the model module and its local generation
    dependency to reload. Copying every Python file beside a dynamically
    loaded model/config/tokenizer class keeps intermediate and final
    checkpoints self-contained without affecting native Transformers models.
    """
    module_dirs = set()
    for obj in objects:
        try:
            source = inspect.getfile(type(obj))
        except (OSError, TypeError):
            continue
        if "transformers_modules" in source and source.endswith(".py"):
            module_dirs.add(os.path.dirname(source))

    copied = []
    for module_dir in sorted(module_dirs):
        for source in sorted(glob.glob(os.path.join(module_dir, "*.py"))):
            if os.path.basename(source) == "__init__.py":
                continue
            destination = os.path.join(output_dir, os.path.basename(source))
            shutil.copy2(source, destination)
            copied.append(os.path.basename(source))
    if copied:
        print(f"Packaged dynamic model code in {output_dir}: {sorted(set(copied))}")


def checkpoint_has_complete_weights(checkpoint_dir, require_success=True):
    """Return whether a checkpoint has a config and every expected weight file."""
    if require_success and not os.path.isfile(os.path.join(checkpoint_dir, "_SUCCESS")):
        return False
    if not os.path.isfile(os.path.join(checkpoint_dir, "config.json")):
        return False

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path) as handle:
                weight_map = json.load(handle).get("weight_map", {})
        except (OSError, ValueError, TypeError):
            return False
        shards = sorted(set(weight_map.values()))
        return bool(shards) and all(
            os.path.isfile(os.path.join(checkpoint_dir, shard))
            and os.path.getsize(os.path.join(checkpoint_dir, shard)) > 0
            for shard in shards
        )

    return any(
        os.path.isfile(os.path.join(checkpoint_dir, filename))
        and os.path.getsize(os.path.join(checkpoint_dir, filename)) > 0
        for filename in ("model.safetensors", "pytorch_model.bin")
    )


def mark_checkpoint_complete(checkpoint_dir):
    """Atomically mark a checkpoint after its complete weight set is durable."""
    if not checkpoint_has_complete_weights(checkpoint_dir, require_success=False):
        raise RuntimeError(f"Refusing to mark incomplete checkpoint: {checkpoint_dir}")
    marker = os.path.join(checkpoint_dir, "_SUCCESS")
    temporary = marker + ".tmp"
    with open(temporary, "w") as handle:
        handle.write("complete\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, marker)


def save_pretrained_distributed(accelerator, model, tokenizer, output_dir):
    """Save a HF checkpoint from FSDP without materializing the full state on GPU.

    FSDP FULL_STATE_DICT can OOM on 40GB GPUs during Transformers' save path because
    it all-gathers and clones parameters. This context offloads the gathered full
    state dict to CPU and only returns it on rank 0.
    """
    accelerator.wait_for_everyone()
    os.makedirs(output_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)

    if accelerator.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()

        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(output_dir)
        del state_dict
    else:
        unwrapped.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

    if accelerator.is_main_process:
        copy_dynamic_model_code((unwrapped, getattr(unwrapped, "config", None), tokenizer), output_dir)

    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_to", choices=["none", "wandb"], default="none",
                        help="External experiment tracking is disabled unless explicitly enabled.")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--trust_remote_model_code", action="store_true",
                        help="Load the model with trust_remote_code=True. Needed for non-LLaDA dLLMs such as Dream.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training traces JSONL")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=1024,
                        help="Optional completion-only token cap applied after the full prompt. "
                             "Use 1024 to match an 8x128 generated-token budget without "
                             "counting prompt tokens against that budget.")
    parser.add_argument("--dynamic_length", action="store_true",
                        help="Do not pad batch-1 examples to max_length. The sequence is still "
                             "truncated at max_length; this changes no non-padding tokens or losses.")
    parser.add_argument("--code_target_format", type=str, default="default",
                        choices=["default", "code_tags"],
                        help="Use <code>...</code> targets for a code-only continuation run.")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_passes", type=int, default=4,
                        help="Number of (noise-mask, backward, optimizer.step) passes per training "
                             "example. The default is 4 to match the chunked recipe's "
                             "per-token number of noise-loss passes.")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=4,
        help="Maximum number of complete resumable weight checkpoints to retain.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default="",
                        help="Training output directory containing checkpoint-N folders. Loads the "
                             "latest valid checkpoint and skips the completed optimizer passes.")
    args = parser.parse_args()
    if args.report_to == "wandb" and wandb is None:
        parser.error("Install wandb to enable --report_to wandb")

    if args.save_total_limit < 1:
        parser.error("--save_total_limit must be >= 1")

    init_seed(args.seed)

    # Set NCCL timeout for large model saves
    os.environ["NCCL_TIMEOUT"] = "5400000"

    accelerator = Accelerator(mixed_precision="bf16")
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Resume from the latest complete weight checkpoint. Optimizer state is not
    # restored, matching the chunked trainer's constant-LR resume behavior.
    resume_step = 0
    if args.resume_from:
        candidates = []
        for checkpoint_dir in glob.glob(os.path.join(args.resume_from, "checkpoint-*")):
            match = re.search(r"checkpoint-(\d+)$", checkpoint_dir)
            if match and checkpoint_has_complete_weights(checkpoint_dir):
                candidates.append((int(match.group(1)), checkpoint_dir))
        if candidates:
            resume_step, resume_dir = max(candidates)
            args.model_name = resume_dir

    # Load model
    if accelerator.is_main_process:
        print(f"Loading model: {args.model_name}")
        if resume_step:
            print(f"Resuming from optimizer-pass step {resume_step}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        trust_remote_code=args.trust_remote_model_code,
        use_fast=True,
    )

    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_model_code,
        torch_dtype=torch.bfloat16,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            inner = model.model if hasattr(model, 'model') else model
            if hasattr(inner, "set_activation_checkpointing"):
                from models.configuration_llada import ActivationCheckpointingStrategy
                inner.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
            else:
                raise RuntimeError(
                    f"{type(model).__name__} does not expose a known gradient checkpointing API."
                )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if accelerator.is_main_process:
            print("Gradient checkpointing enabled")

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = 126336
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if accelerator.is_main_process:
        print(f"Using mask_token_id={mask_token_id}, pad_token_id={pad_token_id}")

    # Load and preprocess data
    with open(args.train_data) as f:
        raw_data = [json.loads(line) for line in f]

    if accelerator.is_main_process:
        print(f"Loaded {len(raw_data)} traces")

    train_data, test_data = preprocess_gpt54(
        raw_data,
        tokenizer,
        args.max_length,
        pad_to_max_length=not args.dynamic_length,
        max_completion_length=args.max_completion_length,
        code_target_format=args.code_target_format,
    )

    if accelerator.is_main_process:
        print(f"Train: {len(train_data)}, Test: {len(test_data)}")
        lengths = [int(item["input_ids"].shape[0]) for item in train_data]
        print(
            f"Sequence lengths: mean={sum(lengths) / max(len(lengths), 1):.1f}, "
            f"min={min(lengths)}, max={max(lengths)}, "
            f"dynamic={args.dynamic_length}"
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    total_steps = math.ceil(len(train_data) / world_size) * args.num_epochs * args.num_passes
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=min(50, total_steps // 10))

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if resume_step:
        # Warmup is only 50 steps; resumed production checkpoints are beyond it.
        # Advance the scheduler so its reported state remains aligned.
        for _ in range(resume_step):
            scheduler.step()

    # Optional external tracking
    if accelerator.is_main_process and args.report_to == "wandb":
        wandb.init(
            project="dllm-registers-reasoning",
            name=args.run_name or f"vanilla-sft-{args.model_name.split('/')[-1]}-maxlen{args.max_length}",
            config=vars(args),
        )

    # Data sharding
    indices = list(range(len(train_data)))
    max_per_rank = math.ceil(len(indices) / world_size)
    padded_indices = indices + indices[:max_per_rank * world_size - len(indices)]

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = resume_step
    loop_started = time.perf_counter()
    examples_processed = 0

    for epoch in range(args.num_epochs):
        random.shuffle(indices)
        padded_indices = indices + indices[:max_per_rank * world_size - len(indices)]
        rank_indices = padded_indices[rank::world_size]

        if accelerator.is_main_process:
            print(f"Epoch {epoch}: {len(rank_indices)} examples/rank, {len(train_data)} total, {world_size} ranks")

        epoch_loss = 0.0
        epoch_steps = 0

        for i, idx in enumerate(rank_indices):
            item = train_data[idx]
            input_ids = item["input_ids"].unsqueeze(0).to(accelerator.device)
            prompt_len = item["prompt_length"]

            # Each pass samples a fresh random noise timestep t inside train_step.
            # For --num_passes 4, this matches the per-(token, noise) gradient exposure
            # of chunked SFT with --num_passes_per_chunk 4.
            pass_losses = []
            for pass_idx in range(args.num_passes):
                absolute_pass = (
                    epoch * len(rank_indices) * args.num_passes
                    + i * args.num_passes
                    + pass_idx
                )
                if absolute_pass < resume_step:
                    continue
                pass_loss = train_step(
                    model=model,
                    input_ids=input_ids,
                    prompt_length=prompt_len,
                    mask_token_id=mask_token_id,
                    pad_token_id=pad_token_id,
                    optimizer=optimizer,
                    accelerator=accelerator,
                    max_grad_norm=args.max_grad_norm,
                )
                pass_losses.append(pass_loss)
                scheduler.step()
                global_step += 1

            if not pass_losses:
                continue

            loss = sum(pass_losses) / len(pass_losses)
            epoch_loss += loss
            epoch_steps += 1
            examples_processed += 1

            if accelerator.is_main_process and global_step % args.logging_steps == 0:
                avg = epoch_loss / epoch_steps
                print(f"  Step {global_step} loss={loss:.4f} avg={avg:.4f}")
                if args.report_to == "wandb":
                    wandb.log({"loss": loss, "avg_loss": avg, "epoch": epoch, "lr": scheduler.get_last_lr()[0]}, step=global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}"
                )
                if accelerator.is_main_process:
                    success_marker = os.path.join(checkpoint_dir, "_SUCCESS")
                    if os.path.exists(success_marker):
                        os.remove(success_marker)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print(f"Saving checkpoint at step {global_step}")
                save_pretrained_distributed(
                    accelerator,
                    model,
                    tokenizer,
                    checkpoint_dir,
                )
                if accelerator.is_main_process:
                    mark_checkpoint_complete(checkpoint_dir)
                    checkpoints = []
                    for checkpoint_dir in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
                        match = re.search(r"checkpoint-(\d+)$", checkpoint_dir)
                        if match and checkpoint_has_complete_weights(checkpoint_dir):
                            checkpoints.append((int(match.group(1)), checkpoint_dir))
                    checkpoints.sort()
                    for _, old_checkpoint in checkpoints[:-args.save_total_limit]:
                        shutil.rmtree(old_checkpoint)
                accelerator.wait_for_everyone()

        if accelerator.is_main_process and epoch_steps:
            print(f"Epoch {epoch} done. Avg loss: {epoch_loss / epoch_steps:.4f}")

    loop_seconds = time.perf_counter() - loop_started
    if accelerator.is_main_process:
        print(
            f"TRAINING_LOOP_SECONDS={loop_seconds:.1f} "
            f"OPTIMIZER_PASSES={global_step - resume_step} "
            f"EXAMPLES_PER_RANK={examples_processed}"
        )
        with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as metrics_file:
            json.dump(
                {
                    "training_loop_seconds": loop_seconds,
                    "optimizer_passes": global_step - resume_step,
                    "examples_per_rank": examples_processed,
                    "world_size": world_size,
                    "resume_step": resume_step,
                    "final_step": global_step,
                    "dynamic_length": args.dynamic_length,
                },
                metrics_file,
                indent=2,
            )

    # Save final model
    save_pretrained_distributed(
        accelerator,
        model,
        tokenizer,
        os.path.join(args.output_dir, "final_model"),
    )
    if accelerator.is_main_process:
        if args.report_to == "wandb":
            wandb.finish()
        print("Training complete!")


if __name__ == "__main__":
    main()
