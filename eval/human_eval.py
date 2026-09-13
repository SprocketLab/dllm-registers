import re
from gsm8k import GSM8KDataset
from datasets import load_dataset
from parsers import Parser, test_solution
import warnings

HUMANEVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"

HUMANEVAL_SYSTEM_PROMPT = """You are a coding expert. You will be given a coding problem to solve. Solve it step by step.

Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
```python
Your code here
```
</answer>"\x20
"""

HUMANEVAL_CODE_TAGS_SYSTEM_PROMPT = """
You are a coding expert. Solve the programming problem.

Respond in the following format:
<code>
Your Python code here
</code>

Continue the Python source exactly when more space is needed. Do not use
Markdown fences.
"""


class HumanEvalDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=HUMANEVAL_SYSTEM_PROMPT,
        subsample=-1,
        output_dir=None,
        num_registers=0,
        prompt_removal=False,
        use_mask_token_for_registers=False,
        channel_mode="registers",
        tail_length=0,
        code_target_format="default",
    ):
        if code_target_format not in ("default", "code_tags"):
            raise ValueError(f"Unsupported code_target_format: {code_target_format}")
        self.code_target_format = code_target_format
        if code_target_format == "code_tags":
            system_prompt = HUMANEVAL_CODE_TAGS_SYSTEM_PROMPT
            add_reasoning = False
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for HumanEval. Overriding num_examples to 0.")
        super().__init__(
            tokenizer,
            0,
            add_reasoning,
            system_prompt,
            subsample,
            num_registers=num_registers,
            prompt_removal=prompt_removal,
            use_mask_token_for_registers=use_mask_token_for_registers,
            channel_mode=channel_mode,
            tail_length=tail_length,
        )
        self.output_dir = output_dir

    def load_test_dataset(self):
        self.dataset_revision = HUMANEVAL_REVISION
        self.dataset = load_dataset(
            "openai/openai_humaneval",
            split="test",
            revision=HUMANEVAL_REVISION,
        )

    def parse_answer_and_score(self, generated_texts, ground_truths, **kwargs):
        preds = []
        num_correct = 0
        for generated_text, ground_truth in zip(generated_texts, ground_truths):
            program = Parser.extract_answer_code(generated_text, code_target_format=self.code_target_format)
            if isinstance(ground_truth, dict):
                unit_test = ground_truth["test"]
                entry_point = ground_truth["entry_point"]
            else:
                unit_test = ground_truth[ground_truth.index("def") :]
                solution_match = re.search(r"def (\w+)\(", program or "")
                entry_point = solution_match.group(1) if solution_match else None
            if program is None:
                preds.append(program)
                continue
            if not entry_point:
                preds.append(program)
                continue
            test_code = program + "\n\n" + unit_test + "\n\n" + f"check({entry_point})"
            preds.append(test_code)
            num_correct += test_solution(test_code, self.output_dir)
        return preds, num_correct, len(preds)

    def __getitem__(self, idx):
        row = self.dataset[self.subsample[idx].item()]
        question = (
            "Complete the following HumanEval problem. Preserve the provided imports and "
            "function signature, and return the complete executable implementation.\n\n"
            + row["prompt"]
        )
        answer = {
            "test": row["test"],
            "entry_point": row["entry_point"],
        }
        prompt = self.create_prompt(question)
        return prompt, question, answer
