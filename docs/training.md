# Training

## Data

Review the upstream data licenses, then download the public [mix60k mixture](https://huggingface.co/datasets/albertge/mix60k-math-code-sft):

```bash
python scripts/download_data.py
```

The downloader pins the dataset revision, checks the JSONL schema, and refuses to overwrite an existing file. The mixture contains approximately 30K OpenMathInstruct-2 and 30K OpenCodeInstruct traces. It is not bundled in Git.

To use your own data, supply a JSONL with `question` and `gpt54_reasoning_trace` strings. The latter is the existing schema's name for the target trace; it does not require a particular trace-generation API. Preserve `source_type` and `solution` when using the released data. Code rows use `source_type: "oci_python"`.

## Mixture training

The launcher exposes four methods:

| Method | Cross-window state | Training signal |
| --- | --- | --- |
| `registers` | Four continuous hidden-state vectors | Next-chunk loss through the preceding write |
| `discrete` | Last four generated token IDs | Chunked denoising loss |
| `memory` | Four continuous hidden-state vectors | Reconstruction loss; next-chunk loss is detached at the write/read bridge |
| `full-sft` | None | Denoising loss over the full target, capped at 1,024 completion tokens |

Preview the register recipe first:

```bash
python train.py --method registers --data data/mix60k.jsonl \
  --output outputs/llada-registers --gpus 8 --dry-run
```

Remove `--dry-run` to train. Replace `registers` with `discrete`, `memory`, or `full-sft`, and give each run a separate output directory. Add `--backbone dream --trust-remote-code` to train Dream. Use `--gpus 1` for a single-device run if it fits.

The default mixture recipes use one epoch, learning rate 2e-5, weight decay 0.1, BF16, gradient checkpointing, and one example per process. Chunked training uses C=128, at most eight chunks, and four denoising-loss passes per chunk. The first pass fully masks the completion. Registers and memory use a prompt-blocking probability of 0.3; discrete carry does not block the prompt. Memory adds reconstruction loss with weight 0.05. Each loss pass takes an optimizer step; the persistent carried state is detached between chunks.

Full-sequence SFT instead uses four noise-loss passes per full target, a 1,024-token completion cap, and a 4,096-token prompt-plus-completion cap. It does not train a carry state.

Changing the number of processes changes the effective batch size and optimization trajectory. These commands expose the recipes, not a promise of bitwise-identical training across hardware. Full-parameter 7B/8B training needs high-memory accelerators; ordinary DDP replicates weights and optimizer state on each process. More DDP GPUs alone do not make an oversized per-device model fit.

For **full-sequence SFT only**, `--gpus 8 --fsdp` enables the supplied sharded configuration. The chunked trainer's checkpoint/resume path supports single-device and DDP training, not FSDP; the launcher rejects that combination. Do not assume a new sharding setup is validated by the CPU tests.

## Code continuation

First select the code rows:

```bash
python SFT/data_prep/filter_jsonl_by_source.py \
  --input data/mix60k.jsonl --output data/code30k.jsonl --source-type oci_python
```

Continue a corresponding mixture-trained model:

```bash
python train.py --method registers --code \
  --model outputs/llada-registers/final_model \
  --data data/code30k.jsonl --output outputs/llada-code-registers \
  --gpus 8 --dry-run
```

Use the matching mixture-trained initialization for each baseline. Code continuation uses `<code>...</code>` targets, learning rate 1e-5, and C=64 with up to eight chunks during chunked training. Register and memory prompt blocking is 0.7. Evaluation allows up to 16 chunks, keeping the total generation budget at 1,024 tokens. The full-SFT continuation keeps its full-target completion cap instead of chunking the training trace.

## Checkpoints and tracking

Completed model weights and tokenizer files are saved under `OUTPUT/final_model`. For inference from that directory, specify the carry layout explicitly; an arbitrary directory name is not a checkpoint alias.

The launcher refuses an existing output directory to prevent accidental overwrites. The lower-level entry points expose checkpoint/resume options:

```bash
python SFT/sft_train_chunked.py --help
python SFT/sft_train_vanilla.py --help
```

Chunked DDP checkpoints include optimizer, scheduler, and per-rank RNG state. Full-SFT restart restores saved weights and progress but is not an exact optimizer-state resume. Load only checkpoints you trust. Periodic checkpoint retention removes older checkpoint directories inside the selected output directory; review `--save_total_limit` before a long run.

No external tracking service is enabled by default. If desired, install `wandb` separately and pass `--report-to wandb`. Authenticate outside the repository and never commit credentials or training data.
