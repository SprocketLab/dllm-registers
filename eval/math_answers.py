"""Ground-truth-independent first-answer extraction for saved and live math eval.

Policy v1: recognize <answer>, \\boxed, \\fbox, and #### in text order.
A wrong/empty/malformed first declaration cannot be rescued by a later answer.
Tags may wrap a box (including an unfinished outer tag); plain tags require a
closing tag. Boxes require balanced braces. Bare boxes and #### end at a line,
control token, another answer declaration, or the current chunk's end. Therefore
saved runs must be replayed chunk by chunk, just like live early stopping.

Only answer extraction changes: math uses the existing symbolic equivalence;
GSM uses whole-answer numeric comparison, never a search for a matching number.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re


METRIC_VERSION = "format-aware-first-answer-v1"
MATH_DATASETS = frozenset((
    "gsm8k", "gsm_hard", "svamp", "math", "omni_math",
    "omni_math_lowmid", "omni_math_easy",
))
_OPEN = re.compile(r"<answer>|\\(?:boxed|fbox)(?=\s|\{)|####", re.IGNORECASE)
_CONTROL = re.compile(r"<\|[^>]*\|>|</?answer>", re.IGNORECASE)
_NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class MathAnswer:
    start: int
    end: int
    text: str
    format: str
    valid: bool = True
    reason: str = "complete"


def _unframe(text):
    text = text.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
            return text[len(left):-len(right)].strip()
    return text


def _answer(start, end, text, kind):
    text = _unframe(text)
    return MathAnswer(start, end, text, kind, bool(text), "complete" if text else "empty")


def first_math_answer(text):
    """Return the first answer event, or None while its delimiter is unfinished.

    This function has no target/model argument. An interrupted first declaration
    produces an invalid event, instead of skipping forward to a subsequent one.
    """
    opening = _OPEN.search(text)
    if opening is None:
        return None
    start, body = opening.span()
    marker = opening.group().lower()
    if marker == "<answer>":
        boundary = _CONTROL.search(text, body)
        limit = boundary.start() if boundary else len(text)
        inner = _OPEN.search(text, body, limit)
        if inner:
            # The usual <answer> $\\boxed{...}$ </answer> wrapper. Do not
            # discard a preceding plain answer to find a later boxed correction.
            prefix = text[body:inner.start()].strip()
            if prefix in ("", "$", "$$", r"\(", r"\["):
                nested = first_math_answer(text[inner.start():limit])
                if nested is not None:
                    return MathAnswer(start, inner.start() + nested.end, nested.text,
                                      "answer_tag/" + nested.format, nested.valid, nested.reason)
        if boundary and boundary.group().lower() == "</answer>":
            return _answer(start, boundary.end(), text[body:limit], "answer_tag")
        if boundary:
            return MathAnswer(start, boundary.end(), text[body:limit].strip(),
                              "answer_tag", False, "interrupted")
        return None

    kind = "hash" if marker == "####" else marker[1:]
    while body < len(text) and text[body].isspace():
        body += 1
    if kind != "hash" and body < len(text) and text[body] == "{":
        depth = 1
        i = body + 1
        while i < len(text):
            boundary = _CONTROL.match(text, i) or _OPEN.match(text, i)
            if boundary:
                return MathAnswer(start, boundary.end(), text[body + 1:i].strip(),
                                  kind, False, "interrupted")
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return _answer(start, i + 1, text[body + 1:i], kind)
            i += 1
        return None

    # A bare box or hash marker declares the whole line, not every number in it.
    boundaries = [len(text)]
    for pattern in (_CONTROL, _OPEN, re.compile(r"\n")):
        match = pattern.search(text, body)
        if match:
            boundaries.append(match.start())
    if kind != "hash":
        dollar = text.find("$", body)
        if dollar >= 0:
            boundaries.append(dollar)
    end = min(boundaries)
    if end == len(text) and not text[body:end].strip():
        return None
    return _answer(start, end, text[body:end], kind)


def _numeric(text):
    text = _unframe(str(text)).strip()
    if text.startswith(r"\$"):
        text = text[2:].strip()
    elif text.startswith("$"):
        text = text[1:].strip()
    if not _NUMBER.fullmatch(text):
        return None
    try:
        value = Decimal(text.replace(",", ""))
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def score_math_answer(answer, ground_truth, dataset):
    if dataset not in MATH_DATASETS:
        raise ValueError(f"Unsupported math dataset: {dataset}")
    if answer is None:
        return 0.0, None
    if not answer.valid:
        return 0.0, answer.text
    prediction = answer.text
    if dataset in ("gsm8k", "gsm_hard", "svamp"):
        pred_number, true_number = _numeric(prediction), _numeric(ground_truth)
        if pred_number is not None and true_number is not None:
            return float(abs(pred_number - true_number) < Decimal("0.000001")), prediction
    from parsers import is_equiv
    return float(is_equiv(prediction, str(ground_truth))), prediction


def score_math_generation(text, ground_truth, dataset):
    return score_math_answer(first_math_answer(text), ground_truth, dataset)


def first_answer_in_chunks(chunks):
    """Replay model-visible stopping on saved chunks; chunk indices are zero-based."""
    prefix = ""
    for index, text in enumerate(chunks):
        prefix += text
        answer = first_math_answer(prefix)
        if answer is not None:
            return answer, index
    return None, None


def score_math_chunk_prefixes(chunks, ground_truth, dataset):
    """Freeze the first answer even when a batch/timing run continues decoding."""
    answer, index = first_answer_in_chunks(chunks)
    if answer is None:
        return [0.0] * len(chunks)
    score, _ = score_math_answer(answer, ground_truth, dataset)
    return [0.0] * index + [score] * (len(chunks) - index)
