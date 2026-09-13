from datasets import load_dataset

from gsm8k import GSM8KDataset
from math500 import MATH500_SYSTEM_PROMPT

OMNI_MATH_REVISION = "40ba231d8f16e29ecd40e6407e2c8640145a8f62"


class OmniMathDataset(GSM8KDataset):
    """Omni-MATH: 4428 Olympiad-level problems from KbsdJames/Omni-MATH.

    Answer format is free-form string (like MATH500). Scoring uses the shared
    first-answer extractor and the existing is_equiv comparison.
    """

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MATH500_SYSTEM_PROMPT,
        subsample=-1,
        num_registers=0,
        prompt_removal=False,
        use_mask_token_for_registers=False,
        channel_mode="registers",
        tail_length=0,
    ):
        super().__init__(
            tokenizer,
            num_examples=num_examples,
            add_reasoning=add_reasoning,
            system_prompt=system_prompt,
            subsample=subsample,
            num_registers=num_registers,
            prompt_removal=prompt_removal,
            use_mask_token_for_registers=use_mask_token_for_registers,
            channel_mode=channel_mode,
            tail_length=tail_length,
        )

    def load_test_dataset(self):
        self.dataset_revision = OMNI_MATH_REVISION
        self.dataset = load_dataset(
            "KbsdJames/Omni-MATH",
            split="test",
            revision=OMNI_MATH_REVISION,
        )

    def load_few_shot_examples(self):
        return []

    def __getitem__(self, idx):
        row = self.dataset[self.subsample[idx].item()]
        question = row["problem"]
        answer = row["answer"]
        prompt = self.create_prompt(question)
        return prompt, question, answer


class OmniMathLowMidDataset(OmniMathDataset):
    """Omni-MATH restricted to low/mid difficulty problems.

    The upstream dataset uses numeric difficulty labels. We keep examples with
    difficulty <= 5.0 to avoid the AIME-like tail while retaining a broad pool.
    """

    max_difficulty = 5.0
    split_label = "low/mid"

    def load_test_dataset(self):
        self.dataset_revision = OMNI_MATH_REVISION
        dataset = load_dataset(
            "KbsdJames/Omni-MATH",
            split="test",
            revision=OMNI_MATH_REVISION,
        )
        self.dataset = dataset.filter(
            lambda row: row["difficulty"] is not None and float(row["difficulty"]) <= self.max_difficulty
        )
        print(
            f"Omni-MATH {self.split_label} filter: difficulty <= {self.max_difficulty}, "
            f"{len(self.dataset)}/{len(dataset)} examples"
        )


class OmniMathEasyDataset(OmniMathLowMidDataset):
    """Omni-MATH restricted to the easiest difficulty bucket used for paper evals."""

    max_difficulty = 2.5
    split_label = "easy"
