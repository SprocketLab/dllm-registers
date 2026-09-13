"""Chunked SFT for continuous registers, discrete carry, and memory tokens.

Each continuation loss differentiates through its preceding register write.
State is detached between chunks; the default recipe uses four loss passes.
"""
import sys
import os
import glob
import json
import hashlib
import inspect
import re
import shutil
# Register dllm's LLaDA model (gradient checkpointing support) before any AutoModel calls
sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402 — registers LLaDAModelLM with AutoModel

import torch
import argparse
from transformers import AutoTokenizer, AutoModel, get_constant_schedule_with_warmup
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import random
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None

from sft_trainer import (
    preprocess_s1k_chunked,
    preprocess_gpt54_vanilla_with_registers,
    ChunkedSFTDataset,
    train_chunk,
    train_chunk_accumulate,
    train_vanilla_step,
)


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def is_complete_model_checkpoint(path):
    """Return whether a checkpoint has config plus a complete weight-shard set."""
    if not (
        os.path.isfile(os.path.join(path, "config.json"))
        or os.path.isfile(os.path.join(path, "adapter_config.json"))
    ):
        return False

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path) as handle:
                weight_map = json.load(handle).get("weight_map", {})
        except (OSError, json.JSONDecodeError):
            return False
        shards = set(weight_map.values())
        return bool(shards) and all(os.path.isfile(os.path.join(path, shard)) for shard in shards)

    weight_files = [
        name for name in os.listdir(path)
        if name.endswith(".safetensors") or re.fullmatch(r"pytorch_model.*\.bin", name)
    ]
    if len(weight_files) != 1:
        return False
    sharded_name = re.search(r"-(\d+)-of-(\d+)\.", weight_files[0])
    return sharded_name is None or int(sharded_name.group(2)) == 1


def copy_dynamic_model_code(objects, output_dir):
    """Copy trusted remote-model Python dependencies into a checkpoint."""
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


def save_pretrained_atomic(model, tokenizer, destination):
    """Publish a complete model directory in one rename."""
    tmp_path = f"{destination}.tmp-{os.getpid()}"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    if os.path.exists(destination):
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {destination}")
    os.makedirs(tmp_path)
    try:
        model.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
        copy_dynamic_model_code((model, getattr(model, "config", None), tokenizer), tmp_path)
        if not is_complete_model_checkpoint(tmp_path):
            raise RuntimeError(f"Checkpoint save did not produce a complete model: {tmp_path}")
        with open(os.path.join(tmp_path, "_SUCCESS"), "w") as handle:
            handle.write("complete\n")
        os.replace(tmp_path, destination)
    except BaseException:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise


def is_complete_training_checkpoint(path, world_size):
    """Return whether model, optimizer/scheduler, and every rank RNG state are published."""
    required = [
        os.path.join(path, "_SUCCESS"),
        os.path.join(path, "training_state.pt"),
        *[
            os.path.join(path, f"rng_state_rank{rank}.pt")
            for rank in range(world_size)
        ],
    ]
    return is_complete_model_checkpoint(path) and all(
        os.path.isfile(filename) for filename in required
    )


def save_training_checkpoint_atomic(
    accelerator,
    model,
    tokenizer,
    optimizer,
    scheduler,
    destination,
    training_progress,
    hybrid_rng,
):
    """Publish a trajectory-resumable DDP checkpoint in one shared-filesystem rename."""
    tmp_path = f"{destination}.tmp"
    if accelerator.is_main_process:
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        if os.path.exists(destination):
            raise FileExistsError(
                f"Refusing to overwrite existing checkpoint: {destination}"
            )
        os.makedirs(tmp_path)
    accelerator.wait_for_everyone()

    try:
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(tmp_path)
            tokenizer.save_pretrained(tmp_path)
            copy_dynamic_model_code(
                (unwrapped_model, getattr(unwrapped_model, "config", None), tokenizer),
                tmp_path,
            )
            torch.save(
                {
                    **training_progress,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                os.path.join(tmp_path, "training_state.pt"),
            )

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "hybrid": hybrid_rng.getstate(),
        }
        torch.save(
            rng_state,
            os.path.join(
                tmp_path,
                f"rng_state_rank{accelerator.process_index}.pt",
            ),
        )
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if not is_complete_model_checkpoint(tmp_path):
                raise RuntimeError(
                    f"Checkpoint save did not produce a complete model: {tmp_path}"
                )
            expected_rng = [
                os.path.join(tmp_path, f"rng_state_rank{rank}.pt")
                for rank in range(accelerator.num_processes)
            ]
            if not all(os.path.isfile(filename) for filename in expected_rng):
                raise RuntimeError(
                    f"Checkpoint save is missing per-rank RNG state: {tmp_path}"
                )
            with open(os.path.join(tmp_path, "_SUCCESS"), "w") as handle:
                handle.write("complete\n")
            os.replace(tmp_path, destination)
        accelerator.wait_for_everyone()
    except BaseException:
        if accelerator.is_main_process:
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise


def build_training_semantics(args, world_size):
    """Arguments that define the optimization trajectory and channel capacity."""
    return {
        "num_registers": args.num_registers,
        "channel_mode": args.channel_mode,
        "tail_length": args.tail_length,
        "use_mask_token_for_registers": args.use_mask_token_for_registers,
        "completion_per_chunk": args.completion_per_chunk,
        "max_chunks_per_trace": args.max_chunks_per_trace,
        "num_passes_per_chunk": args.num_passes_per_chunk,
        "bptt_window": args.bptt_window,
        "bridge_loops": args.bridge_loops,
        "bridge_loss_each_loop": args.bridge_loss_each_loop,
        "bridge_loss_reduction": args.bridge_loss_reduction,
        "front_registers": args.front_registers,
        "prompt_removal": args.prompt_removal,
        "mask_chunk_to_prompt": args.mask_chunk_to_prompt,
        "mask_chunk_state_subgraph": args.mask_chunk_state_subgraph,
        "prompt_dropout_rate": args.prompt_dropout_rate,
        "force_full_mask_first_pass": args.force_full_mask_first_pass,
        "variable_chunk_size": args.variable_chunk_size,
        "min_chunk_size": args.min_chunk_size,
        "residual_detached_ln_register_carry": args.residual_detached_ln_register_carry,
        "detach_primary_register_bridge": args.detach_primary_register_bridge,
        "aux_recon_loss": args.aux_recon_loss,
        "aux_recon_weight": args.aux_recon_weight,
        "aux_recon_task_mode": args.aux_recon_task_mode,
        "vanilla_fraction": args.vanilla_fraction,
        "vanilla_max_length": args.vanilla_max_length,
        "code_target_format": args.code_target_format,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "world_size": world_size,
        "train_data_sha256": args.train_data_sha256,
    }


def validate_resume_semantics(config, tokenizer, args, requested_training_semantics):
    expected_condition = (
        "register_slot_mode_v0" if args.aux_recon_task_mode
        else "completion_mask_offset_v1" if args.aux_recon_loss
        else "none"
    )
    resume_semantics = {
        "d1_detach_primary_register_bridge": args.detach_primary_register_bridge,
        "d1_aux_recon_loss": args.aux_recon_loss,
        "d1_aux_recon_task_mode": args.aux_recon_task_mode,
        "d1_aux_recon_condition": expected_condition,
    }
    prior_task_mode = getattr(config, "d1_aux_recon_task_mode", None)
    for field, expected in resume_semantics.items():
        prior = getattr(config, field, None)
        if prior is not None and prior != expected:
            raise ValueError(
                f"Resume checkpoint {field}={prior!r} does not match requested {expected!r}."
            )
        if prior is None and (
            args.detach_primary_register_bridge
            or args.aux_recon_loss
            or args.aux_recon_task_mode
        ):
            raise ValueError(
                f"Resume checkpoint lacks {field}; refusing an ambiguous memory-token resume. "
                "Start the memory-token arm from its declared base model or a checkpoint with saved semantics."
            )

    write_id = tokenizer.convert_tokens_to_ids("<MODE_WRITE>")
    read_id = tokenizer.convert_tokens_to_ids("<MODE_READ>")
    has_legacy_mode_tokens = (
        write_id != tokenizer.unk_token_id and read_id != tokenizer.unk_token_id
    )
    if prior_task_mode is None and has_legacy_mode_tokens and not args.aux_recon_task_mode:
        raise ValueError(
            "Resume checkpoint contains legacy <MODE_WRITE>/<MODE_READ> tokens but has no "
            "saved mode metadata. Pass --aux_recon_task_mode to preserve its semantics."
        )

    prior_training_semantics = getattr(config, "d1_training_semantics", None)
    if prior_training_semantics is None:
        if args.detach_primary_register_bridge or args.aux_recon_loss:
            raise ValueError(
                "Resume checkpoint lacks d1_training_semantics; refusing an ambiguous "
                "memory-token weights-only warm start."
            )
    elif prior_training_semantics != requested_training_semantics:
        differing = sorted(
            key for key in set(prior_training_semantics) | set(requested_training_semantics)
            if prior_training_semantics.get(key) != requested_training_semantics.get(key)
        )
        raise ValueError(
            "Resume checkpoint training semantics differ for: "
            + ", ".join(differing)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_to", choices=["none", "wandb"], default="none",
                        help="External experiment tracking is disabled unless explicitly enabled.")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument(
        "--model_revision",
        type=str,
        default="",
        help="Immutable Hugging Face revision for model_name.",
    )
    parser.add_argument("--trust_remote_model_code", action="store_true",
                        help="Load the model with trust_remote_code=True. Needed for non-LLaDA dLLMs such as Dream.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--completion_per_chunk", type=int, default=128,
                        help="Number of completion tokens per chunk (total chunk = prompt_len + this)")
    parser.add_argument("--max_chunks_per_trace", type=int, default=8,
                        help="Cap on number of chunks per trace (truncates long traces)")
    parser.add_argument("--num_passes_per_chunk", type=int, default=4,
                        help="Number of bridge+loss passes per chunk; the default recipe steps once per pass.")
    parser.add_argument("--bptt_window", type=int, default=1,
                        help="Truncated BPTT window: keep the graph alive across this many consecutive "
                             "chunks before backward+detach. 1 = current per-chunk backward behavior. "
                             "2 = pair chunks (i, i+1) share a backward; gradient from chunk i+1 flows "
                             "back through the bridge that wrote chunk i's registers, training the "
                             "registers for multi-chunk lookahead. Memory cost scales with window size.")
    parser.add_argument("--bridge_loops", type=int, default=1,
                        help="Number of iterative bridge-refinement loops on continuation chunks before "
                             "using the refined registers for the SFT pass.")
    parser.add_argument("--bridge_loss_each_loop", action="store_true",
                        help="When bridge_loops > 1, compute the masked SFT loss after each refinement loop "
                             "instead of only on the final loop.")
    parser.add_argument("--bridge_loss_reduction", type=str, default="mean", choices=["mean", "sum"],
                        help="How to aggregate per-loop losses when bridge_loops > 1.")
    parser.add_argument("--num_registers", type=int, default=4)
    parser.add_argument("--channel_mode", type=str, default="registers",
                        choices=["registers", "tail", "hybrid"],
                        help="Carry-channel layout between chunks. 'registers' = continuous bridge channel "
                             "at front positions (legacy). 'tail' = discrete last-X-tokens channel at front "
                             "positions, no bridge pass. 'hybrid' = registers then tail (positions 1..R, "
                             "R+1..R+T). Requires --front_registers and is incompatible with --prompt_removal.")
    parser.add_argument("--tail_length", type=int, default=0,
                        help="Length of the discrete tail channel (last-X tokens of prior chunk). "
                             "Must be > 0 when channel_mode is 'tail' or 'hybrid', else 0.")
    parser.add_argument("--use_mask_token_for_registers", action="store_true",
                        help="Place [MASK] token id at register front positions instead of adding "
                             "dedicated <register_i> tokens to the vocabulary. Gives the model's "
                             "pretrained mask-position prior a non-trivial initialization for the "
                             "continuous channel, mitigating posterior collapse.")
    parser.add_argument("--residual_detached_ln_register_carry", action="store_true",
                        help="Ablation: pass LN(bridge_register_embeds + stopgrad(previous_register_embeds)) "
                             "to the next chunk. Preserves earlier compressed state without extending "
                             "the backward graph beyond the local bridge.")
    parser.add_argument("--prompt_dropout_rate", type=float, default=0.3,
                        help="Probability of applying the CSG (or chunk-to-prompt) mask on each chunk's "
                             "SFT pass. 1.0 = always apply (legacy CSG). 0.3 = apply on 30%% of chunks, "
                             "prompt visible on the rest. Stochastic replacement for the fixed mask; "
                             "resists posterior collapse by creating training steps where only the "
                             "carry channel is available.")
    parser.add_argument("--variable_chunk_size", action="store_true",
                        help="Per-trace chunk_size: pick chunk_size = max(min_chunk_size, ceil(L / "
                             "min(max_chunks_per_trace, ceil(L/min_chunk_size)))) so long traces fit "
                             "within max_chunks_per_trace without truncation while never going below "
                             "min_chunk_size. Trains the model across a curriculum of chunk sizes "
                             "rather than a single fixed point — addresses the chunk-size-coupling "
                             "limitation where a fixed-256 recipe goes OOD at 64-128 token chunks at eval.")
    parser.add_argument("--min_chunk_size", type=int, default=None,
                        help="Floor for variable_chunk_size mode. Defaults to completion_per_chunk if unset.")
    parser.add_argument("--prompt_removal", action="store_true",
                        help="Option A: remove question from chunk 1+ prompts, force register dependence")
    parser.add_argument("--front_registers", action="store_true",
                        help="Insert registers at positions 1..num_registers after BOS (instead of embedded "
                             "in user message text). Full prompt still shown at every chunk (unlike prompt_removal).")
    parser.add_argument("--force_full_mask_first_pass", action="store_true",
                        help="Force t=1 (fully masked) on first pass of each chunk, ensuring register dependence")
    parser.add_argument("--mask_chunk_to_prompt", action="store_true",
                        help="Block chunk-completion query positions from attending to prompt-text key positions "
                             "during the SFT pass on chunks i>0. Chunk completion can still attend to BOS, registers, "
                             "and itself. Forces the model to route prompt info through registers via the bridge pass. "
                             "Bridge pass is unmasked (full attention). Requires registers at positions "
                             "1..num_registers, so use with --front_registers (or --prompt_removal).")
    parser.add_argument("--mask_chunk_state_subgraph", action="store_true",
                        help="Stronger continuation mask for chunks i>0: registers and chunk completion may attend "
                             "only to registers and chunk completion during the SFT pass. BOS and prompt text stay "
                             "present in the sequence but are unreadable from the continuation-state subgraph. "
                             "Bridge pass is unmasked (full attention). Requires registers at positions "
                             "1..num_registers, so use with --front_registers (or --prompt_removal).")
    parser.add_argument("--aux_recon_loss", action="store_true",
                        help="Add source-reconstruction auxiliary loss. On each chunk i>0's last pass, runs a "
                             "second forward that asks the model to reconstruct chunk_{i-1}'s tokens from the "
                             "bridged registers alone. Requires --mask_chunk_state_subgraph to prevent the recon "
                             "pass from reading the prompt. By default this preserves all register slots and adds "
                             "no inference-only machinery; use --aux_recon_task_mode only for the legacy mode-bit "
                             "variant.")
    parser.add_argument("--aux_recon_weight", type=float, default=0.05,
                        help="Weight on auxiliary reconstruction loss in total_loss = primary + weight * recon.")
    parser.add_argument("--aux_recon_task_mode", action="store_true",
                        help="Legacy auxiliary-reconstruction variant: add <MODE_WRITE>/<MODE_READ> tokens and "
                             "replace register slot 0 with a task-mode embedding. This reduces effective memory "
                             "capacity by one slot and is not the capacity-matched memory-token control.")
    parser.add_argument("--detach_primary_register_bridge", action="store_true",
                        help="Memory-token control: detach the bridged register tensor before injecting it into "
                             "the current-chunk task-loss pass. Forward values and inference stay identical, but "
                             "the task loss cannot train the preceding writer. With --aux_recon_loss, the "
                             "reconstruction loss still reaches the live writer tensor.")
    parser.add_argument("--vanilla_fraction", type=float, default=0.0,
                        help="Hybrid training: fraction of outer iterations that use vanilla full-sequence batches. "
                             "0.0 = pure chunked (backward compat). 0.99 = hybrid primary.")
    parser.add_argument("--vanilla_max_length", type=int, default=4096,
                        help="Max sequence length for vanilla batches (chunked uses completion_per_chunk).")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs/registers")
    parser.add_argument("--train_data", type=str, default="simplescaling/s1K")
    parser.add_argument(
        "--train_data_sha256",
        type=str,
        default="",
        help="Expected SHA-256 of local train_data, persisted for resume compatibility.",
    )
    parser.add_argument("--code_target_format", type=str, default="default", choices=["default", "code_tags"],
                        help="Target format for code-only SFT. default preserves existing traces; "
                             "code_tags strips Markdown fences and trains <code>...</code> outputs.")
    parser.add_argument("--use_peft", action="store_true", help="Use LoRA instead of full FT")
    parser.add_argument("--export_merged_model", action="store_true",
                        help="When using LoRA, merge adapters into the base model and save a full checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=4,
        help="Maximum number of complete resumable checkpoints to retain.",
    )
    parser.add_argument("--resume_from", type=str, default="",
                        help="Resume from the latest complete checkpoint-{N}. New-format checkpoints "
                             "restore optimizer, scheduler, per-rank RNG, and data cursors exactly. "
                             "Legacy non-memory-token checkpoints fall back to a labeled weights-only warm "
                             "start. Single-epoch (num_epochs=1) only.")
    parser.add_argument("--cooldown", action="store_true",
                        help="WSD-style cooldown branch: with --resume_from, decay the LR "
                             "linearly from --learning_rate to 0 over the remaining steps of "
                             "the epoch (instead of the constant schedule). The LR is set "
                             "directly on the optimizer each iteration, bypassing the "
                             "scheduler, so accelerate's scheduler-stepping semantics cannot "
                             "distort the decay. Requires a valid resume checkpoint.")
    parser.add_argument("--debugging", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.save_steps < 1:
        raise ValueError("--save_steps must be >= 1.")
    if args.save_total_limit < 1:
        raise ValueError("--save_total_limit must be >= 1.")

    if args.bridge_loops < 1:
        raise ValueError("--bridge_loops must be >= 1.")

    if args.bridge_loops > 1:
        if args.num_passes_per_chunk != 1:
            raise ValueError(
                "bridge_loops > 1 currently requires --num_passes_per_chunk 1 to avoid ambiguous K x passes "
                "training behavior."
            )
        if not args.force_full_mask_first_pass:
            raise ValueError(
                "bridge_loops > 1 currently requires --force_full_mask_first_pass so the single outer pass "
                "cannot shortcut through visible completion tokens."
            )

    if args.mask_chunk_to_prompt and args.mask_chunk_state_subgraph:
        raise ValueError("--mask_chunk_to_prompt and --mask_chunk_state_subgraph are mutually exclusive.")

    if args.bptt_window < 1:
        raise ValueError("--bptt_window must be >= 1.")

    if args.bptt_window > 1:
        if args.aux_recon_loss:
            raise ValueError(
                "--bptt_window > 1 does not currently support --aux_recon_loss. The BPTT code path "
                "skips the aux-recon features; remove one of the flags."
            )
        if args.bridge_loops > 1:
            raise ValueError(
                "--bptt_window > 1 does not currently support --bridge_loops > 1. Use one or the other."
            )
        if args.mask_chunk_to_prompt:
            raise ValueError(
                "--bptt_window > 1 currently implements only CSG masking; use "
                "--mask_chunk_state_subgraph, not --mask_chunk_to_prompt."
            )

    if args.aux_recon_loss and not args.mask_chunk_state_subgraph:
        raise ValueError(
            "--aux_recon_loss requires --mask_chunk_state_subgraph; without CSG the recon pass can "
            "read prompt text and trivially reconstruct chunk_{i-1} via the shared prompt prefix."
        )

    if args.aux_recon_task_mode and not args.aux_recon_loss:
        raise ValueError("--aux_recon_task_mode requires --aux_recon_loss.")

    if args.aux_recon_loss and args.prompt_removal:
        raise ValueError(
            "--aux_recon_loss does not support --prompt_removal because previous and current "
            "chunks have different prompt layouts. Use --front_registers for the memory-token control."
        )

    if args.detach_primary_register_bridge and args.num_registers <= 0:
        raise ValueError("--detach_primary_register_bridge requires --num_registers > 0.")

    if args.detach_primary_register_bridge and args.bptt_window > 1:
        raise ValueError(
            "--detach_primary_register_bridge is implemented for the standard bptt_window=1 path only."
        )

    if args.residual_detached_ln_register_carry and args.num_registers <= 0:
        raise ValueError("--residual_detached_ln_register_carry requires --num_registers > 0.")

    if not (args.mask_chunk_to_prompt or args.mask_chunk_state_subgraph):
        return

    # Attention masking needs at least one front-channel position — register slots, tail slots,
    # or both. Tail-only variants (num_registers=0, tail_length>0) are valid.
    if args.num_registers <= 0 and args.tail_length <= 0:
        raise ValueError("attention masking requires --num_registers > 0 or --tail_length > 0.")

    if not (args.front_registers or args.prompt_removal):
        raise ValueError(
            "attention masking requires --front_registers or --prompt_removal so the carry channel "
            "lives at fixed front positions."
        )

    if args.channel_mode in ("tail", "hybrid"):
        if args.tail_length <= 0:
            raise ValueError(f"--channel_mode={args.channel_mode} requires --tail_length > 0.")
        if not args.front_registers:
            raise ValueError(f"--channel_mode={args.channel_mode} requires --front_registers.")
        if args.prompt_removal:
            raise ValueError(f"--channel_mode={args.channel_mode} is incompatible with --prompt_removal.")
    if args.channel_mode == "tail" and args.num_registers != 0:
        raise ValueError("--channel_mode=tail requires --num_registers 0.")
    if args.channel_mode == "registers" and args.tail_length != 0:
        raise ValueError("--tail_length must be 0 when --channel_mode=registers.")

    if not (0.0 <= args.prompt_dropout_rate <= 1.0):
        raise ValueError(f"--prompt_dropout_rate must be in [0, 1], got {args.prompt_dropout_rate}.")


def load_model_and_tokenizer(args):
    tokenizer_kwargs = {
        "padding_side": "right",
        "trust_remote_code": args.trust_remote_model_code,
        "use_fast": True,
    }
    model_kwargs = {
        "trust_remote_code": args.trust_remote_model_code,
        "torch_dtype": torch.bfloat16,
    }
    if args.model_revision:
        tokenizer_kwargs["revision"] = args.model_revision
        model_kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, **tokenizer_kwargs
    )

    model = AutoModel.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    # Add register tokens (and optional legacy mode tokens). Skipped when
    # --use_mask_token_for_registers is set: register positions reuse the existing mask
    # token id, so no vocabulary addition / embedding resize is needed.
    if args.num_registers > 0 and not args.use_mask_token_for_registers:
        register_tokens = [f"<register_{i}>" for i in range(args.num_registers)]
        special_tokens = list(register_tokens)
        if args.aux_recon_task_mode:
            # R_0 carries a "task mode" bit (write=predict next chunk, read=reconstruct prev chunk)
            # to disambiguate the two training objectives at the same all-masked-completion input.
            # Both modes are rows in the wte table and get gradient signal via primary+recon losses.
            special_tokens += ["<MODE_WRITE>", "<MODE_READ>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        mode_msg = " + 2 mode tokens" if args.aux_recon_task_mode else ""
        print(f"Added {args.num_registers} register tokens{mode_msg}, vocab size: {len(tokenizer)}")
    elif args.use_mask_token_for_registers:
        if args.aux_recon_task_mode:
            raise ValueError("--use_mask_token_for_registers is not compatible with --aux_recon_task_mode "
                             "(the legacy mode variant repurposes register slot 0 with dedicated mode tokens).")
        # LLaDA's tokenizer doesn't expose .mask_token_id; Dream and similar models usually do.
        resolved_mask_id = tokenizer.mask_token_id
        if resolved_mask_id is None:
            resolved_mask_id = getattr(model.config, "mask_token_id", None)
        if resolved_mask_id is None:
            resolved_mask_id = 126336
        print(f"Using mask_token_id ({resolved_mask_id}) at {args.num_registers} register positions "
              f"(no vocab addition, mask-prior init).")

    if args.use_peft:
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model = model.to(torch.bfloat16)
        print(f"LoRA params: {model.print_trainable_parameters()}")
    else:
        print(f"Full fine-tuning, total params: {sum(p.numel() for p in model.parameters()):,}")

    return tokenizer, model


def train(args):
    validate_args(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ],
    )
    if args.train_data_sha256:
        if not os.path.isfile(args.train_data):
            raise ValueError("--train_data_sha256 requires train_data to be a local file.")
        actual_hash = [None]
        if accelerator.is_main_process:
            digest = hashlib.sha256()
            with open(args.train_data, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_hash[0] = digest.hexdigest()
        if accelerator.num_processes > 1:
            import torch.distributed as dist
            dist.broadcast_object_list(actual_hash, src=0)
        if actual_hash[0] != args.train_data_sha256:
            raise ValueError(
                f"train_data SHA-256 mismatch: expected {args.train_data_sha256}, "
                f"got {actual_hash[0]}"
            )

    # --- Resume detection.
    if args.cooldown and not args.resume_from:
        raise ValueError("--cooldown is a branch off an existing run: it requires --resume_from.")
    if args.cooldown and args.vanilla_fraction > 0.0:
        raise ValueError("--cooldown does not support hybrid (vanilla_fraction > 0) runs.")

    resume_step = 0
    resume_dir = None
    exact_resume = False
    if args.resume_from:
        import glob
        if args.num_epochs != 1:
            raise ValueError("--resume_from currently supports single-epoch runs (num_epochs=1) only.")
        candidates = []
        for d in glob.glob(os.path.join(args.resume_from, "checkpoint-*")):
            m = re.search(r"checkpoint-(\d+)$", d)
            if m and is_complete_model_checkpoint(d):
                candidates.append((int(m.group(1)), d))
        if candidates:
            resume_step, resume_dir = max(candidates)
            exact_resume = (
                not args.cooldown
                and is_complete_training_checkpoint(
                    resume_dir,
                    accelerator.num_processes,
                )
            )
            args.model_name = resume_dir
            args.model_revision = ""
            if (
                (args.detach_primary_register_bridge or args.aux_recon_loss)
                and not exact_resume
            ):
                raise ValueError(
                    f"memory-token resume checkpoint {resume_dir} lacks exact optimizer/scheduler/"
                    "per-rank RNG state. Restart this arm from its declared base checkpoint."
                )
            if accelerator.is_main_process:
                if exact_resume:
                    print(
                        f"EXACT RESUME from {resume_dir} at global_step={resume_step}"
                    )
                else:
                    print(
                        f"WEIGHTS-ONLY WARM START from {resume_dir} at "
                        f"global_step={resume_step}; optimizer/scheduler/RNG state will be fresh"
                    )
        elif accelerator.is_main_process:
            print(f"--resume_from={args.resume_from}: no valid checkpoint found; starting fresh.")

    if args.cooldown:
        if resume_step <= 0:
            raise ValueError(f"--cooldown: no valid checkpoint found under {args.resume_from!r}.")
        if accelerator.is_main_process:
            print(f"COOLDOWN branch: linear LR decay {args.learning_rate} -> 0 "
                  f"from step {resume_step} to end of epoch")

    tokenizer, model = load_model_and_tokenizer(args)
    requested_training_semantics = build_training_semantics(
        args,
        accelerator.num_processes,
    )

    if resume_step > 0:
        validate_resume_semantics(
            model.config,
            tokenizer,
            args,
            requested_training_semantics,
        )

    # Persist control semantics in every saved checkpoint.
    model.config.d1_detach_primary_register_bridge = args.detach_primary_register_bridge
    model.config.d1_aux_recon_loss = args.aux_recon_loss
    model.config.d1_aux_recon_task_mode = args.aux_recon_task_mode
    model.config.d1_aux_recon_condition = (
        "register_slot_mode_v0" if args.aux_recon_task_mode
        else "completion_mask_offset_v1" if args.aux_recon_loss
        else "none"
    )
    model.config.d1_training_semantics = requested_training_semantics

    # Resolve mode token IDs only for the optional legacy task-mode variant.
    # to identify register slot 0 as "write next chunk" (primary) vs "reconstruct prev chunk" (recon).
    write_mode_id = None
    read_mode_id = None
    if args.aux_recon_task_mode:
        write_mode_id = tokenizer.convert_tokens_to_ids("<MODE_WRITE>")
        read_mode_id = tokenizer.convert_tokens_to_ids("<MODE_READ>")
        if write_mode_id == tokenizer.unk_token_id or read_mode_id == tokenizer.unk_token_id:
            raise RuntimeError(
                "Mode tokens <MODE_WRITE>/<MODE_READ> are not in the tokenizer. "
                "Expected them to be added during load_model_and_tokenizer when aux_recon_loss=True."
            )
        print(f"Mode tokens: WRITE={write_mode_id}, READ={read_mode_id}")

    # Load and preprocess data
    if args.train_data.endswith(".jsonl"):
        # Local jsonl file (e.g., GPT-5.4 traces)
        import json
        with open(args.train_data) as f:
            data = [json.loads(line) for line in f]
        if accelerator.is_main_process:
            print(f"Loaded {len(data)} examples from {args.train_data}")
    else:
        # HuggingFace dataset
        data = load_dataset(args.train_data, split="train")
    train_traces, eval_traces = preprocess_s1k_chunked(
        data, tokenizer,
        completion_per_chunk=args.completion_per_chunk,
        num_registers=args.num_registers,
        max_chunks_per_trace=args.max_chunks_per_trace,
        prompt_removal=args.prompt_removal,
        front_registers=args.front_registers,
        channel_mode=args.channel_mode,
        tail_length=args.tail_length,
        use_mask_token_for_registers=args.use_mask_token_for_registers,
        variable_chunk_size=args.variable_chunk_size,
        min_chunk_size=args.min_chunk_size,
        code_target_format=args.code_target_format,
    )

    print(f"Train traces: {len(train_traces)}, Eval traces: {len(eval_traces)}")
    total_chunks = sum(len(t) for t in train_traces)
    print(f"Total training chunks: {total_chunks}")

    train_dataset = ChunkedSFTDataset(train_traces)

    # Hybrid: also load vanilla full-sequence dataset (with registers at positions 1-16)
    vanilla_train = None
    if args.vanilla_fraction > 0.0:
        vanilla_train, _vanilla_eval = preprocess_gpt54_vanilla_with_registers(
            data, tokenizer,
            num_registers=args.num_registers,
            max_length=args.vanilla_max_length,
        )
        print(f"Vanilla train samples: {len(vanilla_train)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    if args.cooldown:
        # WSD-style cooldown branch: LR decays linearly from base to 0 over the
        # remaining steps [resume_step, epoch_iterations]. Both endpoints are
        # derived (resume detection above; single-epoch length below) so the
        # branch needs no step arithmetic from the caller. The resume
        # fast-forward after accelerator.prepare keeps the schedule aligned.
        from torch.optim.lr_scheduler import LambdaLR
        if resume_step <= 0:
            raise ValueError("--cooldown found no valid checkpoint-{N} under --resume_from; "
                             "a cooldown must branch off a trained trunk checkpoint.")
        ws = args.warmup_steps
        cs = resume_step
        ce = (len(train_dataset) + accelerator.num_processes - 1) // accelerator.num_processes
        if ce <= cs:
            raise ValueError(f"cooldown branch point (step {cs}) is at/past the epoch end ({ce}).")
        print(f"Cooldown schedule: constant until step {cs}, linear decay to 0 at step {ce}")

        def _wsd_lambda(step):
            if step < ws:
                return step / max(1, ws)
            if step < cs:
                return 1.0
            return max(0.0, (ce - step) / (ce - cs))

        scheduler = LambdaLR(optimizer, _wsd_lambda)
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
        )

    if args.gradient_checkpointing:
        # Re-enable input grad requirement so gradient flows into checkpointed blocks.
        # Needed for --bptt_window > 1 to avoid excessive activation memory.
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if accelerator.is_main_process:
            print("[gradient_checkpointing] enabled on backbone")

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    resume_training_state = None
    if exact_resume:
        resume_training_state = torch.load(
            os.path.join(resume_dir, "training_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if int(resume_training_state.get("global_step", -1)) != resume_step:
            raise ValueError(
                f"Checkpoint directory step {resume_step} disagrees with training_state.pt "
                f"global_step={resume_training_state.get('global_step')}"
            )
        optimizer.load_state_dict(resume_training_state["optimizer"])
        scheduler.load_state_dict(resume_training_state["scheduler"])

    # On resume, fast-forward the (constant-after-warmup) LR schedule to the resume step so
    # get_last_lr() is correct. Warnings about stepping the scheduler before the optimizer are
    # expected here and suppressed.
    if resume_step > 0 and not exact_resume:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for _ in range(resume_step):
                scheduler.step()

    # Wandb
    if accelerator.is_main_process and not args.debugging and args.report_to == "wandb":
        if args.vanilla_fraction > 0.0:
            run_name = f"hybrid-van{args.vanilla_fraction}-R{args.num_registers}-c{args.completion_per_chunk}"
        else:
            run_name = f"chunked-R{args.num_registers}-c{args.completion_per_chunk}-cap{args.max_chunks_per_trace}"
        if args.mask_chunk_state_subgraph:
            run_name += "-state-subgraph"
        elif args.mask_chunk_to_prompt:
            run_name += "-mask-to-prompt"
        if args.bridge_loops > 1:
            run_name += f"-bridge{args.bridge_loops}"
        if args.residual_detached_ln_register_carry:
            run_name += "-resid-detach-ln"
        if args.detach_primary_register_bridge:
            run_name += "-detached-memory"
        if args.aux_recon_loss:
            run_name += f"-recon{args.aux_recon_weight:g}"
        if args.aux_recon_task_mode:
            run_name += "-modebit"
        wandb.init(
            project="dllm-registers-reasoning",
            name=run_name,
            config=vars(args),
        )

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = getattr(accelerator.unwrap_model(model).config, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = 126336
    if accelerator.is_main_process:
        print(f"Using training mask_token_id={mask_token_id}")
    global_step = resume_step

    # Data parallelism: each rank processes different traces
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    import torch.distributed as dist

    def shard_indices(n, epoch_seed):
        """Return shuffled + rank-sharded + padded indices for this epoch."""
        idxs = list(range(n))
        rng = random.Random(epoch_seed)
        rng.shuffle(idxs)
        max_per_rank = (len(idxs) + world_size - 1) // world_size
        padded = idxs + idxs[:max_per_rank * world_size - len(idxs)]
        return padded[rank::world_size]

    def run_chunked_iteration(trace_chunks):
        """Process one chunked trace. Returns (total_loss, total_recon, num_chunks)."""
        register_embeds = None
        prev_chunk_ids = None
        prev_prompt_len = None
        prev_completion_len = None
        num_trace_chunks = len(trace_chunks)

        # Sync chunk count across ranks for DDP collective sync
        num_chunks_tensor = torch.tensor([num_trace_chunks], device=accelerator.device)
        if dist.is_initialized():
            dist.all_reduce(num_chunks_tensor, op=dist.ReduceOp.MAX)
        max_chunks_this_step = int(num_chunks_tensor.item())

        # Per-trace CSG decision: sample Bernoulli(prompt_dropout_rate) once for the whole trace
        # and pass the deterministic 0.0/1.0 to each chunk call. Keeps the attention regime
        # consistent across the chain of chunks within a trace — otherwise the bridge on chunk k
        # is trained under no CSG pressure while chunk k+1 unexpectedly requires registers to
        # carry everything, which is an incoherent training signal.
        if args.prompt_dropout_rate >= 1.0:
            trace_csg_rate = 1.0
        elif args.prompt_dropout_rate <= 0.0:
            trace_csg_rate = 0.0
        else:
            trace_csg_rate = 1.0 if random.random() < args.prompt_dropout_rate else 0.0

        total_loss = 0.0
        total_recon = 0.0
        n = 0

        if args.bptt_window > 1:
            # Truncated BPTT path. Accumulate loss across `bptt_window` chunks then
            # single backward/step/detach. Normalize by (bptt_window * num_passes) so
            # the effective per-parameter update magnitude matches the per-chunk path.
            accumulated_loss = torch.zeros((), device=accelerator.device)
            accumulated_n_terms = 0
            optimizer.zero_grad()
            for ci in range(max_chunks_this_step):
                is_real = ci < num_trace_chunks
                chunk_data = trace_chunks[ci] if is_real else trace_chunks[-1]
                chunk_ids = chunk_data["input_ids"].unsqueeze(0).to(accelerator.device)
                prompt_len = chunk_data["prompt_length"]
                completion_len = chunk_data.get("completion_length")
                reg_positions = chunk_data["register_positions"]

                chunk_loss, register_embeds, m_eff = train_chunk_accumulate(
                    model=model,
                    chunk_input_ids=chunk_ids,
                    prompt_length=prompt_len,
                    register_positions=reg_positions,
                    num_registers=args.num_registers,
                    register_embeds_from_prev=register_embeds,
                    mask_token_id=mask_token_id,
                    accelerator=accelerator,
                    num_passes=args.num_passes_per_chunk,
                    prev_chunk_input_ids=prev_chunk_ids,
                    is_padding=not is_real,
                    force_full_mask_first_pass=args.force_full_mask_first_pass,
                    mask_chunk_state_subgraph=args.mask_chunk_state_subgraph,
                    num_tail_positions=args.tail_length,
                    prompt_dropout_rate=trace_csg_rate,
                    residual_detached_ln_register_carry=args.residual_detached_ln_register_carry,
                    completion_length=completion_len,
                    prev_prompt_length=prev_prompt_len,
                    prev_completion_length=prev_completion_len,
                )
                accumulated_loss = accumulated_loss + chunk_loss
                accumulated_n_terms += m_eff
                prev_chunk_ids = chunk_ids
                prev_prompt_len = prompt_len
                prev_completion_len = completion_len
                n += int(is_real)

                at_window_end = ((ci + 1) % args.bptt_window == 0) or (ci == max_chunks_this_step - 1)
                if at_window_end:
                    window_loss = accumulated_loss / max(accumulated_n_terms, 1)
                    accelerator.backward(window_loss)
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    # Detach so next window starts a fresh graph
                    if register_embeds is not None:
                        register_embeds = register_embeds.detach()
                    total_loss += window_loss.item() * accumulated_n_terms
                    accumulated_loss = torch.zeros((), device=accelerator.device)
                    accumulated_n_terms = 0
            return total_loss, total_recon, n

        for ci in range(max_chunks_this_step):
            is_real = ci < num_trace_chunks
            chunk_data = trace_chunks[ci] if is_real else trace_chunks[-1]
            chunk_ids = chunk_data["input_ids"].unsqueeze(0).to(accelerator.device)
            prompt_len = chunk_data["prompt_length"]
            completion_len = chunk_data.get("completion_length")
            reg_positions = chunk_data["register_positions"]

            chunk_loss, register_embeds, chunk_recon = train_chunk(
                model=model,
                chunk_input_ids=chunk_ids,
                prompt_length=prompt_len,
                register_positions=reg_positions,
                num_registers=args.num_registers,
                register_embeds_from_prev=register_embeds,
                mask_token_id=mask_token_id,
                optimizer=optimizer,
                accelerator=accelerator,
                max_grad_norm=args.max_grad_norm,
                num_passes=args.num_passes_per_chunk,
                bridge_loops=args.bridge_loops,
                bridge_loss_each_loop=args.bridge_loss_each_loop,
                bridge_loss_reduction=args.bridge_loss_reduction,
                prev_chunk_input_ids=prev_chunk_ids,
                is_padding=not is_real,
                force_full_mask_first_pass=args.force_full_mask_first_pass,
                mask_chunk_to_prompt=args.mask_chunk_to_prompt,
                mask_chunk_state_subgraph=args.mask_chunk_state_subgraph,
                aux_recon_loss=args.aux_recon_loss,
                aux_recon_weight=args.aux_recon_weight,
                write_mode_id=write_mode_id,
                read_mode_id=read_mode_id,
                aux_recon_task_mode=args.aux_recon_task_mode,
                detach_primary_register_bridge=args.detach_primary_register_bridge,
                num_tail_positions=args.tail_length,
                prompt_dropout_rate=trace_csg_rate,
                residual_detached_ln_register_carry=args.residual_detached_ln_register_carry,
                completion_length=completion_len,
                prev_prompt_length=prev_prompt_len,
                prev_completion_length=prev_completion_len,
            )
            prev_chunk_ids = chunk_ids
            prev_prompt_len = prompt_len
            prev_completion_len = completion_len
            total_loss += chunk_loss
            total_recon += chunk_recon
            n += int(is_real)
        return total_loss, total_recon, n

    def run_vanilla_iteration(vanilla_item):
        """Process one vanilla full-sequence sample."""
        input_ids = vanilla_item["input_ids"].unsqueeze(0).to(accelerator.device)
        prompt_len = vanilla_item["prompt_length"]
        loss = train_vanilla_step(
            model=model,
            input_ids=input_ids,
            prompt_length=prompt_len,
            mask_token_id=mask_token_id,
            optimizer=optimizer,
            accelerator=accelerator,
            max_grad_norm=args.max_grad_norm,
        )
        return loss

    # Shared RNG across ranks for hybrid coin flips (all ranks MUST pick the same mode)
    hybrid_rng = random.Random(args.seed + 999)
    if exact_resume:
        rng_state = torch.load(
            os.path.join(
                resume_dir,
                f"rng_state_rank{accelerator.process_index}.pt",
            ),
            map_location="cpu",
            weights_only=False,
        )
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        if torch.cuda.is_available() and rng_state["torch_cuda"]:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        hybrid_rng.setstate(rng_state["hybrid"])

    for epoch in range(args.num_epochs):
        model.train()

        # Build rank-sharded index lists for both modes
        chunked_indices = shard_indices(len(train_dataset), args.seed + epoch)
        vanilla_indices = shard_indices(len(vanilla_train), args.seed + epoch + 10000) if vanilla_train is not None else []

        if accelerator.is_main_process:
            print(f"Epoch {epoch}: {len(chunked_indices)} chunked/rank, {len(vanilla_indices)} vanilla/rank, "
                  f"world_size={world_size}, vanilla_fraction={args.vanilla_fraction}")

        # Epoch length: number of outer iterations this epoch.
        # For hybrid, this is set by the vanilla dataset size (one epoch = one pass through vanilla).
        # For pure chunked (vanilla_fraction=0), use chunked dataset length.
        if args.vanilla_fraction > 0.0:
            epoch_iterations = len(vanilla_indices)
        else:
            epoch_iterations = len(chunked_indices)

        # Resume at the exact saved data cursors when available. Legacy warm starts
        # retain the historical deterministic-prefix approximation.
        if exact_resume and epoch == 0:
            if int(resume_training_state.get("epoch", -1)) != epoch:
                raise ValueError("Exact resume checkpoint epoch does not match requested epoch.")
            start_it = int(resume_training_state["next_iteration"])
            chunked_pos = int(resume_training_state["chunked_pos"])
            vanilla_pos = int(resume_training_state["vanilla_pos"])
        else:
            start_it = resume_step if epoch == 0 else 0
            chunked_pos = start_it
            vanilla_pos = 0
        if start_it > epoch_iterations:
            raise ValueError(
                f"Resume iteration {start_it} exceeds epoch length {epoch_iterations}."
            )
        if start_it > 0 and accelerator.is_main_process:
            print(f"Epoch {epoch}: resuming at iteration {start_it}/{epoch_iterations}")

        for it in range(start_it, epoch_iterations):
            # WSD cooldown branch: linear LR decay from base to 0 across the remaining
            # iterations, written directly into the optimizer each iteration so the
            # (prepared) constant scheduler's step() cannot overwrite what training uses.
            if args.cooldown:
                cd_factor = (epoch_iterations - it) / max(1, epoch_iterations - start_it)
                for pg in optimizer.param_groups:
                    pg["lr"] = args.learning_rate * cd_factor

            # Decide mode via shared RNG (same on all ranks)
            if args.vanilla_fraction > 0.0:
                use_vanilla = hybrid_rng.random() < args.vanilla_fraction
            else:
                use_vanilla = False

            if use_vanilla:
                # Vanilla full-sequence step
                idx = vanilla_indices[vanilla_pos % len(vanilla_indices)]
                vanilla_pos += 1
                loss = run_vanilla_iteration(vanilla_train[idx])

                if accelerator.is_main_process and (global_step < 5 or global_step % args.logging_steps == 0):
                    print(f"  Step {global_step} VANILLA loss={loss:.4f}")

                if global_step % args.logging_steps == 0 and accelerator.is_main_process and not args.debugging and args.report_to == "wandb":
                    lr = optimizer.param_groups[0]["lr"]  # actual LR (cooldown writes here; scheduler may lag)
                    wandb.log({
                        "train/loss_vanilla": loss,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/mode": 0,  # 0 = vanilla
                    }, step=global_step)
            else:
                # Chunked trace step
                idx = chunked_indices[chunked_pos % len(chunked_indices)]
                chunked_pos += 1
                trace_chunks = train_dataset[idx]

                if global_step == 0 and accelerator.is_main_process:
                    num_trace_chunks = len(trace_chunks)
                    print(f"First chunked trace: {num_trace_chunks} chunks, completion_per_chunk={args.completion_per_chunk}")

                chunked_total_loss, chunked_total_recon, num_chunks = run_chunked_iteration(trace_chunks)
                avg_chunk_loss = chunked_total_loss / max(num_chunks, 1)
                avg_recon_loss = chunked_total_recon / max(num_chunks - 1, 1)  # recon only fires on chunks 1+

                # Chunked iterations are rare in hybrid mode — always log them
                if accelerator.is_main_process:
                    recon_msg = f" recon={avg_recon_loss:.4f}" if args.aux_recon_loss else ""
                    print(f"  Step {global_step} CHUNKED avg_loss={avg_chunk_loss:.4f}{recon_msg} chunks={num_chunks}")
                    if not args.debugging:
                        lr = optimizer.param_groups[0]["lr"]  # actual LR (cooldown writes here; scheduler may lag)
                        log_dict = {
                            "train/loss_chunked": avg_chunk_loss,
                            "train/chunks_per_trace": num_chunks,
                            "train/lr": lr,
                            "train/epoch": epoch,
                            "train/step": global_step,
                            "train/mode": 1,  # 1 = chunked
                        }
                        if args.aux_recon_loss:
                            log_dict["train/recon_loss_chunked"] = avg_recon_loss
                        wandb.log(log_dict, step=global_step)

            scheduler.step()
            global_step += 1

            if global_step % args.save_steps == 0:
                # Keep every rank at the same training boundary while rank 0 writes.
                # Otherwise non-main DDP ranks can enter the next forward and deadlock.
                accelerator.wait_for_everyone()
                save_path = os.path.join(
                    args.output_dir,
                    f"checkpoint-{global_step}",
                )
                save_training_checkpoint_atomic(
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    destination=save_path,
                    training_progress={
                        "global_step": global_step,
                        "epoch": epoch,
                        "next_iteration": it + 1,
                        "chunked_pos": chunked_pos,
                        "vanilla_pos": vanilla_pos,
                    },
                    hybrid_rng=hybrid_rng,
                )
                if accelerator.is_main_process:
                    print(f"Saved checkpoint to {save_path}")

                    # Keep only the configured number of complete resumable checkpoints.
                    # Atomic saves ensure the previous complete checkpoint remains available
                    # until the newest one has its _SUCCESS marker.
                    import glob as _glob
                    import re as _re3
                    import shutil as _shutil
                    existing = sorted(
                        (int(_re3.search(r"checkpoint-(\d+)$", d).group(1)), d)
                        for d in _glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
                        if _re3.search(r"checkpoint-(\d+)$", d)
                    )
                    for _, old in existing[:-args.save_total_limit]:
                        _shutil.rmtree(old, ignore_errors=True)
                accelerator.wait_for_everyone()

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final_model")
        unwrapped_model = accelerator.unwrap_model(model)
        save_pretrained_atomic(unwrapped_model, tokenizer, final_path)
        print(f"Saved final model to {final_path}")

        if args.use_peft and args.export_merged_model:
            if hasattr(unwrapped_model, "merge_and_unload"):
                merged_model = unwrapped_model.merge_and_unload()
                merged_path = os.path.join(args.output_dir, "final_model_merged")
                os.makedirs(merged_path, exist_ok=True)
                merged_model.save_pretrained(merged_path)
                tokenizer.save_pretrained(merged_path)
                print(f"Saved merged full model to {merged_path}")
            else:
                print("WARNING: export_merged_model requested but model lacks merge_and_unload(); skipping.")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and not args.debugging and args.report_to == "wandb":
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    if args.report_to == "wandb" and wandb is None:
        raise ImportError("Install wandb to enable --report_to wandb")
    init_seed(args.seed)
    train(args)
