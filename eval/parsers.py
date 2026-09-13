import os
import re
import numpy as np
import ast
import resource
import secrets
import signal
import subprocess
import sys
import tempfile


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        final_answer = matches[-1].strip()
        # Strip \boxed{} wrapper if present
        boxed_match = re.match(r'\\boxed\{(.*)\}', final_answer, re.DOTALL)
        if boxed_match:
            final_answer = boxed_match.group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    if "wait" in solution_str:
        do_print = True
    do_print = np.random.rand() < 0.4
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


class Parser:
    @classmethod
    def extract_answer_gsm8k(cls, generated_text):
        """Extract the first numerical answer following '####' in the generated text."""
        try:
            # Use regex to find the first occurrence of #### followed by a number.
            # Handle formats like: #### 18, #### $18, #### -18.0
            match = re.search(r"####\s*\$?([-+]?\d[\d,]*(?:\.\d+)?)", generated_text)
            if match:
                return float(match.group(1).replace(",", ""))
        except Exception as e:
            print(f"Error extracting answer: {e}, Text: {generated_text[:100]}")
        return None

    @classmethod
    def extract_answer_boxed(cls, generated_text):
        """Extract the last complete \\boxed{...} content. Returns None if not found."""
        # Find the LAST complete \boxed{...} by scanning for balanced braces.
        # Skip truncated \boxed at end of generation (no matching braces).
        best = None
        i = 0
        while True:
            idx = generated_text.find("\\boxed{", i)
            if idx < 0:
                break
            # Scan for matching closing brace
            j = idx + len("\\boxed{")
            depth = 1
            while j < len(generated_text) and depth > 0:
                if generated_text[j] == "{":
                    depth += 1
                elif generated_text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                # Matched — extract content
                best = generated_text[idx + len("\\boxed{"):j - 1]
            i = idx + 1
        return best

    @classmethod
    def extract_answer_letter(cls, generated_text):
        """Extract a single A-D letter answer. Prefers the last \\boxed{...}.

        Falls back to scanning common phrasings: "answer is X", "\\boxed(X)", "(X)".
        Returns the uppercase letter or None.
        """
        boxed = cls.extract_answer_boxed(generated_text)
        if boxed is not None:
            m = re.search(r"[A-D]", boxed)
            if m:
                return m.group(0).upper()
        m = re.search(r"answer\s*(?:is|:)\s*\(?([A-D])\)?", generated_text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"\\boxed\s*\{?\s*\(?([A-D])\)?", generated_text)
        if m:
            return m.group(1).upper()
        return None

    @classmethod
    def extract_answer_boxed_ctd(cls, generated_text):
        """Extract the first numerical answer following '####' in the generated text."""
        pred = Parser.extract_answer_boxed(generated_text)
        pred = pred.replace(r"\div", "/").replace("\times", "*").replace(r"\cdot", "*")
        return pred

    @classmethod
    def extract_answer_grpo_ctd(cls, generated_text):
        """Extract the first numerical answer following '####' in the generated text."""
        pred = extract_solution(generated_text)
        print(generated_text)
        print(pred)
        if pred is not None:

            pred = pred.replace(r"\div", "/").replace("\times", "*").replace(r"\cdot", "*")

        return pred

    @classmethod
    def extract_answer_sudoku(cls, solution_str):
        """Extract the Sudoku solution from the generated text."""
        answer_pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(answer_pattern, solution_str, re.DOTALL)
        if matches:
            # Get the last answer tag content, strip whitespace and newlines
            final_answer = re.sub(r"\s", "", matches[-1].strip())
            return final_answer
        return None

    @classmethod
    def extract_answer_code_tags(cls, solution_str):
        """Extract Python code wrapped in <code>...</code> tags.

        During chunked generation the model may not emit the closing tag before
        the token budget expires, so an open code span is treated as the answer
        up to the first known special/control marker or the end of text.
        """
        start = solution_str.find("<code>")
        if start < 0:
            return None
        start += len("<code>")

        end = solution_str.find("</code>", start)
        if end < 0:
            candidates = []
            for marker in (
                "<|eot_id|>",
                "<|endoftext|>",
                "<reasoning>",
                "</reasoning>",
                "<answer>",
                "</answer>",
                "<code>",
            ):
                pos = solution_str.find(marker, start)
                if pos >= 0:
                    candidates.append(pos)
            end = min(candidates) if candidates else len(solution_str)

        code = solution_str[start:end].strip("\n")
        return code if code.strip() else None

    @classmethod
    def extract_answer_code(cls, solution_str, code_target_format="default"):
        """Extract the coding solution from the generated text."""
        if code_target_format == "code_tags":
            code = cls.extract_answer_code_tags(solution_str)
            if code is not None:
                return code
        matches = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(matches):
            return matches[0]
        return None


def is_equiv(str1, str2, verbose=False):
    if type(str1) == float or type(str2) == float:
        try:
            return abs(float(str1) - float(str2)) < 1e-6
        except:
            return False
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]
    except:
        return s


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str.strip(), {"__builtins__": None}, {})
        return result
    except Exception as e:
        return float("Inf")


def test_solution(code_str, output_dir=None):
    def _sandbox_child():
        # Avoid address-space limits: they caused false failures with ROCm/Python
        # library mappings. Bound CPU, file output, descriptors, and subprocesses.
        resource.setrlimit(resource.RLIMIT_CPU, (10, 11))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))

    sentinel = f"__CODE_EVAL_SUCCESS_{secrets.token_hex(16)}__"
    wrapped_code = (
        code_str
        + "\n\nimport os as __code_eval_os\n"
        + f"__code_eval_os.write(1, {sentinel.encode()!r})\n"
    )
    process = None
    try:
        sandbox_parent = output_dir if output_dir and os.path.isdir(output_dir) else None
        with tempfile.TemporaryDirectory(
            prefix="code-eval-",
            dir=sandbox_parent,
        ) as sandbox_dir:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", wrapped_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=sandbox_dir,
                env={
                    key: os.environ[key]
                    for key in ("PATH", "PYTHONPATH", "LANG", "LC_ALL")
                    if key in os.environ
                },
                start_new_session=True,
                preexec_fn=_sandbox_child,
            )
            stdout, stderr = process.communicate(timeout=10)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return 1 if process.returncode == 0 and sentinel in stdout else 0
    except subprocess.TimeoutExpired:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        return 0
    except Exception:
        return 0


def extract_human_eval_prompt(prompt):
    in_docstring = False
    docstring_lines = []
    triple_quote = None

    for line in prompt.splitlines():
        stripped = line.strip()

        # Detect opening triple quotes
        if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            triple_quote = stripped[:3]
            if stripped.count(triple_quote) == 2 and len(stripped) > 6:
                # Docstring starts and ends on same line
                return stripped[3:-3].strip()
            in_docstring = True
            docstring_lines.append(stripped[3:])
        elif in_docstring:
            if stripped.endswith(triple_quote):
                docstring_lines.append(stripped[:-3])
                break
            docstring_lines.append(stripped)

    return "\n".join(docstring_lines).strip()
