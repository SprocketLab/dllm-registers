import re
from gsm8k import GSM8KDataset
from datasets import load_dataset
from parsers import Parser, test_solution
import warnings

MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"

MBPP_SYSTEM_PROMPT = """You are a coding expert. You will be given a coding problem to solve. Solve it step by step. Ensure you wrap the answer in ```python````

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

MBPP_CODE_TAGS_SYSTEM_PROMPT = """
You are a coding expert. Solve the programming problem.

Respond in the following format:
<code>
Your Python code here
</code>

Continue the Python source exactly when more space is needed. Do not use
Markdown fences.
"""


class MBPPDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MBPP_SYSTEM_PROMPT,
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
            system_prompt = MBPP_CODE_TAGS_SYSTEM_PROMPT
            add_reasoning = False
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for MBPP. Overriding num_examples to 0.")
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
        self.dataset_revision = MBPP_REVISION
        self.dataset = load_dataset(
            "google-research-datasets/mbpp",
            "sanitized",
            split="test",
            revision=MBPP_REVISION,
        )

    def parse_answer_and_score(self, generated_texts, ground_truths, **kwargs):
        preds = []
        num_correct = 0
        for generated_text, unit_test in zip(generated_texts, ground_truths):
            program = Parser.extract_answer_code(generated_text, code_target_format=self.code_target_format)
            if program is None:
                preds.append(program)
                continue
            # NOTE: Don't need to extract func name like human_eval since prompt specifies it
            # If model doesn't follow the instruction, it shouldn't pass
            test_code = program + "\n\n" + unit_test
            preds.append(test_code)
            num_correct += test_solution(test_code, self.output_dir)
        return preds, num_correct, len(preds)

    def __getitem__(self, idx):
        unit_tests = "\n".join(self.dataset[self.subsample[idx].item()]["test_list"])
        question = (
            self.dataset[self.subsample[idx].item()]["prompt"]
            + "\n\n Your code should pass the following tests:\n\n"
            + unit_tests
        )
        answer = "\n".join(self.dataset[self.subsample[idx].item()]["test_imports"]) + f"\n\n{unit_tests}"
        prompt = self.create_prompt(question)
        return prompt, question, answer
