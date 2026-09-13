"""Batch-one benchmark evaluation with fixed context windows and first-answer scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import (add_model_arguments, generate_chunks, load_checkpoint,
                       prompt_ids, resolve_spec, run_options, seed_everything)
from math_answers import MATH_DATASETS, METRIC_VERSION, first_math_answer, score_math_generation


def first_answer_end_char(text, dataset, code_target_format="code_tags"):
    if dataset in MATH_DATASETS:
        answer = first_math_answer(text)
        return answer.end if answer is not None else None
    start = text.find("<code>")
    end = text.find("</code>", start+6) if start >= 0 else -1
    return end+7 if end >= 0 else None


def has_answer(text, dataset, code_target_format="code_tags"):
    return first_answer_end_char(text, dataset, code_target_format) is not None


def score_generation(generated_text, gt_answer, dataset, code_target_format="code_tags"):
    if dataset in MATH_DATASETS:
        return score_math_generation(generated_text, gt_answer, dataset)
    from parsers import Parser, test_solution
    program = Parser.extract_answer_code(generated_text, code_target_format=code_target_format)
    if program is None:
        return 0.0, None
    if dataset == "humaneval":
        tests = gt_answer["test"] + "\ncheck(" + gt_answer["entry_point"] + ")"
    elif dataset == "mbpp":
        tests = gt_answer
    else:
        raise ValueError("Unsupported benchmark")
    return float(test_solution(program+"\n\n"+tests)), program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--dataset", choices=["gsm8k", "gsm_hard", "math", "omni_math_easy", "humaneval", "mbpp"], required=True)
    parser.add_argument("--limit", type=int, default=-1, help="Random seeded subset size; -1 evaluates every row")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing files are never overwritten")
    parser.add_argument("--allow-code-execution", action="store_true",
                        help="Acknowledge that code scoring executes generated programs; use an isolated environment")
    args = parser.parse_args()
    code = args.dataset in ("humaneval", "mbpp")
    if code and not args.allow_code_execution:
        parser.error("Code scoring executes generated programs. Use an isolated, credential-free environment and --allow-code-execution.")
    if args.output.exists():
        parser.error("Output exists; choose a new filename")
    if args.limit == 0 or args.limit < -1:
        parser.error("--limit must be -1 or a positive count")
    from gsm8k import GSM8KDataset, GSMHardDataset
    from math500 import MATH500Dataset
    from omni_math import OmniMathEasyDataset
    from human_eval import HumanEvalDataset
    from mbpp import MBPPDataset
    datasets = dict(gsm8k=GSM8KDataset, gsm_hard=GSMHardDataset, math=MATH500Dataset,
                    omni_math_easy=OmniMathEasyDataset, humaneval=HumanEvalDataset, mbpp=MBPPDataset)
    spec = resolve_spec(args)
    spec["task"] = "code" if code else "math"
    seed_everything(args.seed)
    model, tokenizer = load_checkpoint(spec, args.device, args.trust_remote_code)
    kwargs = dict(num_examples=0, subsample=args.limit, num_registers=0, add_reasoning=not code)
    if code:
        kwargs["code_target_format"] = "code_tags"
    dataset = datasets[args.dataset](tokenizer, **kwargs)
    if not len(dataset):
        parser.error("The selected dataset is empty")
    records = []
    for index in range(len(dataset)):
        prompt, question, answer = dataset[index]
        chunks = list(generate_chunks(model, tokenizer, prompt_ids(tokenizer, prompt, model.device),
                                      **run_options(args, spec)))
        text = chunks[-1]["accumulated_generation"]
        score, prediction = score_generation(text, answer, args.dataset)
        records.append(dict(question=question, ground_truth=answer, prediction=prediction,
                            score=score, sample_id=hashlib.sha256(question.encode()).hexdigest(), chunks=chunks))
        print(f"{index+1}/{len(dataset)}: accuracy={sum(r['score'] for r in records)/(index+1):.4f}", flush=True)
    import torch
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(metric="pass@1" if code else METRIC_VERSION, n=len(records),
                  accuracy=sum(r["score"] for r in records)/len(records),
                  checkpoint=spec, dataset=args.dataset, dataset_revision=getattr(dataset, "dataset_revision", None),
                  seed=args.seed, total_tokens=args.total_tokens, block_length=args.block_length,
                  steps=args.steps or spec["chunk_size"]//2, reset_state=args.reset_state,
                  temperature=args.temperature, torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                  generations=records)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
