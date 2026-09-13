"""CPU-only parser and live-evaluator wiring tests; no model dependencies."""
import ast
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from math_answers import (MATH_DATASETS, first_math_answer, first_answer_in_chunks,
                          score_math_answer, score_math_generation, score_math_chunk_prefixes)


class FirstMathAnswerTest(unittest.TestCase):
    def test_supported_formats(self):
        for text in (r"\boxed{20}", r"\fbox{20}", "<answer>20</answer>",
                     "<ANSWER>20</ANSWER>", "#### 20", r"\boxed 20",
                     r"<answer>\boxed{20}</answer>", r"<answer>$\boxed{20}$</answer>",
                     r"<answer>\boxed{20}"):
            with self.subTest(text=text):
                self.assertEqual(score_math_generation(text, "20", "gsm8k"), (1.0, "20"))

    def test_whole_numeric_normalization(self):
        for text, target in (("70,000", "70000"), ("$70,000", "70000"),
                             ("$70,000$", "70000"), (r"\$70,000", "70000"),
                             ("-2.5", "-2.50"), ("+2", "2.0"),
                             (".25", "0.25"), ("2e3", "2000")):
            with self.subTest(text=text):
                self.assertEqual(score_math_generation(f"<answer>{text}</answer>", target, "gsm8k")[0], 1)

    def test_wrong_first_answer_is_never_rescued(self):
        for text in (r"\boxed{60} then \boxed{16.00}",
                     r"<answer>60</answer> then \boxed{16}",
                     r"\boxed{60} then #### 16", "#### 60\n<answer>16</answer>",
                     "<answer>60</answer><answer>16</answer>"):
            with self.subTest(text=text):
                self.assertEqual(score_math_generation(text, 16, "gsm8k")[0], 0)

    def test_numeric_substring_is_not_an_answer(self):
        for text in ("6x + x = 3(x + 9)", "6 or 9", "16", "boxed{6}", "nan", "inf"):
            with self.subTest(text=text):
                self.assertEqual(score_math_generation(f"<answer>{text}</answer>", 6, "gsm8k")[0], 0)
        self.assertEqual(score_math_generation("#### 6 + 3 = 9", 6, "gsm8k")[0], 0)

    def test_no_answer(self):
        for text in ("The calculation is 20.", r"\boxed{20", "<answer>20", "#### "):
            with self.subTest(text=text):
                self.assertIsNone(first_math_answer(text))
                self.assertEqual(score_math_generation(text, 20, "gsm8k"), (0, None))

    def test_empty_answer_does_not_skip_to_later_answer(self):
        for text in (r"\boxed{} \boxed{20}", r"<answer></answer>\boxed{20}"):
            self.assertEqual(score_math_generation(text, 20, "gsm8k")[0], 0)

    def test_interrupted_first_tag_does_not_span_eos_or_nested_tag(self):
        for separator in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", ""):
            text = f"<answer>77{separator}<answer>259</answer>"
            answer = first_math_answer(text)
            self.assertFalse(answer.valid)
            self.assertEqual(score_math_answer(answer, 259, "gsm8k")[0], 0)

    def test_interrupted_first_box_is_not_skipped(self):
        answer = first_math_answer(r"\boxed{77 <|im_end|>\boxed{259}")
        self.assertFalse(answer.valid)
        self.assertEqual(score_math_answer(answer, 259, "gsm8k")[0], 0)

    def test_nested_latex_braces(self):
        text = r"<answer>$\boxed{\frac{1}{2}}$</answer>"
        self.assertEqual(score_math_generation(text, r"\frac{1}{2}", "math"), (1, r"\frac{1}{2}"))

    def test_plain_answer_before_inner_box_is_not_discarded(self):
        self.assertEqual(score_math_generation(r"<answer>60, actually \boxed{16}</answer>", 16, "gsm8k")[0], 0)

    def test_chunk_replay_follows_live_stopping(self):
        answer, index = first_answer_in_chunks(["<ans", "wer>20</ans", "wer>", r"\boxed{40}"])
        self.assertEqual((answer.text, index), ("20", 2))
        answer, index = first_answer_in_chunks(["#### 2", "0"])
        self.assertEqual((answer.text, index), ("2", 0))

    def test_equivalence_unchanged_for_symbolic_math(self):
        self.assertEqual(score_math_generation(r"\boxed{\frac12}", r"\frac{1}{2}", "math")[0], 1)
        self.assertEqual(score_math_generation(r"\boxed{x^2}", "x^2", "omni_math_easy")[0], 1)

    def test_batch_or_timing_continuation_cannot_change_answer(self):
        self.assertEqual(score_math_chunk_prefixes(["#### 2", "0"], 20, "gsm8k"), [0, 0])
        self.assertEqual(score_math_chunk_prefixes(["#### 2", "0"], 2, "gsm8k"), [1, 1])
        self.assertEqual(score_math_chunk_prefixes(["<answer>", "20</answer>", "<answer>40</answer>"],
                                                  20, "gsm8k"), [0, 1, 1])

    def test_extraction_does_not_depend_on_target(self):
        text = "<answer>20</answer><answer>40</answer>"
        self.assertEqual(score_math_generation(text, "20", "gsm8k")[1],
                         score_math_generation(text, "40", "gsm8k")[1])

    def test_live_evaluator_uses_same_parser_for_stop_and_score(self):
        # Load only these pure functions, avoiding torch/transformers imports.
        tree = ast.parse(Path(__file__).with_name("eval.py").read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("first_answer_end_char", "has_answer", "score_generation")]
        ns = dict(re=re, MATH_DATASETS=MATH_DATASETS, first_math_answer=first_math_answer,
                  score_math_generation=score_math_generation, SYNTHETIC_STATE_DATASETS=())
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "eval.py", "exec"), ns)
        for text in ("<answer>20</answer>", r"\fbox{20}", r"\boxed{20}", "#### 20"):
            self.assertTrue(ns["has_answer"](text, "gsm8k"))
            self.assertEqual(ns["score_generation"](text, 20, "gsm8k"), (1, "20"))
        self.assertEqual(ns["first_answer_end_char"]("<code>pass</code>", "mbpp", "code_tags"), 17)


if __name__ == "__main__":
    unittest.main()
