"""Launch the documented SFT recipes. --dry-run prints commands without training."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_command(args):
    if args.gpus < 1:
        raise ValueError("--gpus must be positive")
    if args.fsdp and args.gpus < 2:
        raise ValueError("FSDP requires at least two processes")
    if args.fsdp and args.method != "full-sft":
        raise ValueError("The chunked trainer supports single-device/DDP checkpointing, not FSDP")
    model = args.model or {"llada": "GSAI-ML/LLaDA-8B-Base", "dream": "Dream-org/Dream-v0-Base-7B"}[args.backbone]
    if args.backbone == "dream" and not args.trust_remote_code:
        raise ValueError("Dream requires --trust-remote-code; inspect the selected model code first")
    command = [sys.executable, "-m", "accelerate.commands.launch"]
    if args.gpus > 1:
        config = "SFT/accelerate_fsdp.yaml" if args.fsdp else "SFT/accelerate_multigpu.yaml"
        command += ["--config_file", str(ROOT/config)]
        if args.fsdp and args.backbone == "dream":
            command += ["--fsdp_transformer_layer_cls_to_wrap", "DreamDecoderLayer"]
    command += ["--num_processes", str(args.gpus), "--mixed_precision", "bf16"]
    full = args.method == "full-sft"
    command += [str(ROOT/"SFT"/("sft_train_vanilla.py" if full else "sft_train_chunked.py")),
                "--model_name", model, "--train_data", str(args.data.resolve()),
                "--output_dir", str(args.output.resolve()), "--num_epochs", "1",
                "--learning_rate", "1e-5" if args.code else "2e-5",
                "--batch_size", "1", "--weight_decay", "0.1", "--seed", str(args.seed),
                "--gradient_checkpointing", "--report_to", args.report_to]
    if args.trust_remote_code:
        command += ["--trust_remote_model_code"]
    if args.code:
        command += ["--code_target_format", "code_tags"]
    if full:
        command += ["--max_length", "4096", "--max_completion_length", "1024",
                    "--num_passes", "4", "--dynamic_length"]
    else:
        command += ["--completion_per_chunk", "64" if args.code else "128",
                    "--max_chunks_per_trace", "8", "--num_passes_per_chunk", "4",
                    "--force_full_mask_first_pass", "--front_registers", "--use_mask_token_for_registers"]
        if args.method == "discrete":
            command += ["--channel_mode", "tail", "--num_registers", "0",
                        "--tail_length", "4", "--prompt_dropout_rate", "0"]
        else:
            command += ["--channel_mode", "registers", "--num_registers", "4",
                        "--mask_chunk_state_subgraph", "--prompt_dropout_rate", "0.7" if args.code else "0.3"]
            if args.method == "memory":
                command += ["--aux_recon_loss", "--aux_recon_weight", "0.05", "--detach_primary_register_bridge"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["registers", "discrete", "memory", "full-sft"], required=True)
    parser.add_argument("--backbone", choices=["llada", "dream"], default="llada")
    parser.add_argument("--model", help="Override the initialization with a trusted model ID or local checkpoint")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--code", action="store_true", help="Code-only continuation: requires code-only data and --model")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.code and not args.model:
        parser.error("--code is a continuation; supply --model for the corresponding mixture-trained checkpoint")
    try:
        command = build_command(args)
    except ValueError as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not args.data.is_file():
        parser.error("Training JSONL does not exist")
    if args.output.exists():
        parser.error("Output exists. Use a fresh directory; for explicit resume use the SFT entry point directly.")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
