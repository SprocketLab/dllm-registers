import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
from generate import generate
import random
import re
from datasets import load_dataset
from parsers import Parser, is_equiv
import torch.distributed as dist

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM_HARD_REVISION = "960448f73503112d4226baeb8eb41d3fb5ae2506"

GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.\x20
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=GSM_SYSTEM_PROMPT,
        subsample=-1,
        num_registers=0,
        prompt_removal=False,
        use_mask_token_for_registers=False,
        channel_mode="registers",
        tail_length=0,
        plain_prompt=False,
    ):
        if channel_mode not in ("registers", "tail", "hybrid"):
            raise ValueError(f"channel_mode must be registers|tail|hybrid, got {channel_mode}")
        if channel_mode == "tail" and num_registers != 0:
            raise ValueError("channel_mode=tail requires num_registers=0")
        if channel_mode in ("tail", "hybrid") and tail_length <= 0:
            raise ValueError(f"channel_mode={channel_mode} requires tail_length > 0")
        if channel_mode == "registers":
            tail_length = 0

        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.plain_prompt = plain_prompt
        self.system_prompt = system_prompt
        self.num_registers = num_registers
        self.prompt_removal = prompt_removal
        self.use_mask_token_for_registers = use_mask_token_for_registers
        self.channel_mode = channel_mode
        self.tail_length = tail_length
        self.mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336
        self.register_str = " ".join([f"<register_{i}>" for i in range(num_registers)]) if num_registers > 0 else ""
        if num_registers > 0:
            if use_mask_token_for_registers:
                self.reg_token_ids = [self.mask_token_id] * num_registers
            else:
                self.reg_token_ids = [
                    tokenizer.convert_tokens_to_ids(f"<register_{i}>")
                    for i in range(num_registers)
                ]
        else:
            self.reg_token_ids = []
        self.tail_token_ids = [self.mask_token_id] * (tail_length if channel_mode in ("tail", "hybrid") else 0)
        self.front_token_ids = self.reg_token_ids + self.tail_token_ids
        self.load_test_dataset()
        self.create_few_shot_prompt()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        self.dataset_revision = GSM8K_REVISION
        self.dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split="test",
            revision=GSM8K_REVISION,
        )

    def _insert_front_channels_after_bos(self, token_ids):
        """Insert register/tail channel token IDs right after BOS (position 0)."""
        import torch
        return torch.cat([
            token_ids[:1],
            torch.tensor(self.front_token_ids, dtype=torch.long),
            token_ids[1:],
        ])

    def create_prompt(self, input_text):
        # Plain continuation format (official-style base-model evaluation):
        # no chat template, no system prompt, no <reasoning> prefill. The model
        # imitates the few-shot answers' native format (GSM8K '#### N', MATH
        # '\boxed{}'), both of which score_generation parses.
        if self.plain_prompt:
            if self.num_examples > 0:
                return f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
            return f"Question: {input_text}\nAnswer:\n"
        # Format similar to your chat function
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text
        user_content = self.system_prompt + "\n\n" + prompt
        if self.register_str and not self.prompt_removal:
            # Legacy: registers embedded in user message text
            user_content = f"{user_content} {self.register_str}"
        messages = [{"role": "user", "content": user_content}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            return user_input + "<reasoning>"
        else:
            return user_input

    def create_continuation_prompt(self):
        """Create minimal continuation prompt for chunk 1+ (no question, prompt_removal mode)."""
        messages = [{"role": "user", "content": "Continue your reasoning."}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            text += "<reasoning>"
        return text

    def load_few_shot_examples(self):
        # Few-shot dataset selection: this guard previously tested
        # isinstance(self.dataset, GSM8KDataset) — but self.dataset is the HF
        # dataset object, so it was ALWAYS False and GSM8K few-shot silently
        # loaded zero examples (MATH500 overrides this method and was
        # unaffected). Correct intent: subclasses without their own override
        # (GSM-Hard, SVAMP) may reuse the GSM8K train split for demos.
        if isinstance(self, GSM8KDataset):
            train_data = load_dataset(
                "openai/gsm8k",
                "main",
                split="train",
                revision=GSM8K_REVISION,
            )
            examples = random.sample(range(len(train_data)), self.num_examples)
            return [train_data[example] for example in examples]
        else:
            return []

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples"""
        if self.num_examples == 0:
            self.few_shot_prompt = ""
            return
        few_shot_examples = self.load_few_shot_examples()

        formatted_examples = []
        for example in few_shot_examples:
            input_text = example["question"]
            answer = example["answer"]
            formatted_examples.append(f"Question: {input_text}\nAnswer:\n{answer}")
        self.few_shot_prompt = "\n\n".join(formatted_examples)

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["question"]
        answer = Parser.extract_answer_gsm8k(self.dataset[self.subsample[idx].item()]["answer"])
        prompt = self.create_prompt(question)
        return prompt, question, answer

    def collate_fn(self, batch):
        import torch
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        # Front-channel mode: insert register slots followed by tail placeholders after BOS.
        if self.prompt_removal and self.front_token_ids:
            new_ids = []
            for ids in input_ids:
                new_ids.append(self._insert_front_channels_after_bos(ids))
            input_ids = torch.stack(new_ids)
            # Update displayed prompts to include registers (reflects what the model actually sees)
            prompts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in input_ids]
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}


class GSMHardDataset(GSM8KDataset):
    """GSM-Hard: GSM8K-style questions with larger, less common numbers."""

    def load_test_dataset(self):
        self.dataset_revision = GSM_HARD_REVISION
        self.dataset = load_dataset(
            "reasoning-machines/gsm-hard",
            split="train",
            revision=GSM_HARD_REVISION,
        )

    def load_few_shot_examples(self):
        return []

    @staticmethod
    def _format_target(target):
        try:
            value = float(target)
            if value.is_integer():
                return str(int(value))
            return str(value)
        except (TypeError, ValueError):
            return str(target)

    def __getitem__(self, idx):
        row = self.dataset[self.subsample[idx].item()]
        question = row["input"]
        answer = self._format_target(row["target"])
        prompt = self.create_prompt(question)
        return prompt, question, answer


class SVAMPDataset(GSM8KDataset):
    """SVAMP: short arithmetic word problems with numeric answers."""

    def load_test_dataset(self):
        self.dataset = load_dataset("ChilleD/SVAMP", split="test")

    def load_few_shot_examples(self):
        return []

    @staticmethod
    def _format_answer(answer):
        text = str(answer).strip().replace(",", "")
        try:
            value = float(text)
            if value.is_integer():
                return str(int(value))
            return str(value)
        except ValueError:
            return text

    @staticmethod
    def _format_question(row):
        question = str(row.get("question_concat", "")).strip()
        if question:
            return question
        body = str(row.get("Body", "")).strip()
        prompt = str(row.get("Question", "")).strip()
        return f"{body} {prompt}".strip()

    def __getitem__(self, idx):
        row = self.dataset[self.subsample[idx].item()]
        question = self._format_question(row)
        answer = self._format_answer(row["Answer"])
        prompt = self.create_prompt(question)
        return prompt, question, answer
