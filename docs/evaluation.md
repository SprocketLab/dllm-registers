# Inference and evaluation

## Window and carry layout

The original prompt remains available at inference. Four carry positions are inserted after its first token. At each boundary:

1. Finish denoising the current C-token completion window.
2. For registers or memory, run a clean forward pass over that completed window and read the last-layer carry-position states. For discrete text, copy the last four completion token IDs.
3. Clear the generated text from the active input. Retain the original prompt and initialize a new masked completion window with the carried state.

The full output is a concatenation of chunks stored outside the model's active input. Full-SFT evaluation uses the same bounded-window loop with no carry slots. It reuses the original prompt after a reset; it cannot read earlier generated chunks. To evaluate full-SFT decoding on a single long canvas instead, explicitly set `--chunk-size 1024 --max-chunks 1`.

`--reset-state` keeps the selected slot layout but disables cross-chunk updates. It is not the same as switching to the separately trained full-SFT checkpoint.

## Default protocol

| Setting | Math | Code |
| --- | --- | --- |
| Completion window C | 128 | 64 |
| Maximum chunks | 8 | 16 |
| Total generated-token budget | 1,024 | 1,024 |
| Denoising block length | 32 | 32 |
| Denoising steps per window | C/2 | C/2 |
| Temperature | 0 | 0 |
| Prompting | Zero-shot, benchmark-specific instruction | Zero-shot, `<code>` instruction |
| Score | First-answer accuracy | One-program pass@1 |

The prompt and carry positions are additional to C. This implementation evaluates one problem at a time. A smaller `--limit` selects a seeded random subset, not the first rows. Set `--seed` to control selection and generation randomness.

Math prompts request a boxed answer and append `<reasoning>` after the chat template. Generation stops at the first answer recognized by the parser, regardless of correctness. A later correct answer cannot rescue an earlier wrong answer. Code chunks are concatenated into one program and scored once against the problem's tests; generation stops when a complete `<code>...</code>` region appears or the budget runs out. There is no test feedback during generation.

## Examples

Full GSM8K evaluation:

```bash
python eval/eval.py --checkpoint llada-math-registers --dataset gsm8k \
  --output outputs/llada-registers-gsm8k.json
```

Dream MATH500 smoke evaluation:

```bash
python eval/eval.py --checkpoint dream-math-registers --trust-remote-code \
  --dataset math --limit 10 --output outputs/dream-registers-math-smoke.json
```

Use the corresponding `-discrete`, `-memory`, or `-sft` alias for another trained method. Use `gsm_hard` and `omni_math_easy` for the remaining math benchmarks.

### Code execution safety

**Code scoring executes model-generated Python. Run it in a disposable, isolated environment with no credentials, sensitive mounts, or access to important services.** The scorer has time/resource limits and a temporary working directory, but these are not a security sandbox. The `--allow-code-execution` flag acknowledges the risk; it does not create isolation.

Only inside such an environment:

```bash
python eval/eval.py --checkpoint llada-code-registers --dataset humaneval \
  --limit 5 --allow-code-execution --output outputs/humaneval-smoke.json
```

Replace `humaneval` with `mbpp` for MBPP. Choose a **code** checkpoint alias; the evaluator does not silently replace a math-trained checkpoint with its code continuation. Use `inference.py` if you only want to inspect generated code without running it.

## Precision and reproducibility

All GPU checkpoints are loaded in BF16. The aliases select BF16 denoising for LLaDA and code, and FP16 denoising autocast for Dream math. Continuous-state extraction between chunks uses the loaded model precision, outside denoising autocast. `--dtype` changes denoising autocast, not the weight dtype. Use the same settings when comparing methods.

Checkpoint aliases pin full revision hashes, including a separate remote-code revision for Dream. Benchmark loaders also pin dataset revisions. Output JSON records the checkpoint, decoding settings, dataset revision, seed, PyTorch version, GPU model, per-example predictions, scores, and decoded chunks. Existing output files are never overwritten. Results are saved when the evaluation finishes; this small reference runner does not resume interrupted benchmark jobs.

Hardware kernels and precision can change deterministic token choices, so a seed alone is not a guarantee of identical scores across devices. This release's CPU tests validate mechanics and scoring, not full-model benchmark accuracy. Keep the output JSON and environment information when reporting results.
