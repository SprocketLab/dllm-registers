from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import pytest

from train import build_command

ROOT = Path(__file__).resolve().parents[1]


def args(**overrides):
    return Namespace(**dict(dict(gpus=1, fsdp=False, method="registers", model=None,
                                backbone="llada", trust_remote_code=False,
                                data=Path("data/example.jsonl"), output=Path("outputs/test"),
                                code=False, seed=42, report_to="none"), **overrides))


@pytest.mark.parametrize("method", ["registers", "discrete", "memory", "full-sft"])
def test_training_commands_are_offline_dry_runs(method):
    run = subprocess.run([sys.executable, "train.py", "--method", method,
                          "--data", "nonexistent.jsonl", "--output", "outputs/test", "--dry-run"],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "--report_to none" in run.stdout
    assert not (ROOT / "outputs/test").exists()


def test_memory_detaches_primary_bridge_and_adds_reconstruction():
    command = build_command(args(method="memory"))
    assert "--detach_primary_register_bridge" in command
    assert "--aux_recon_loss" in command
    assert command[command.index("--aux_recon_weight") + 1] == "0.05"
    assert "--force_full_mask_first_pass" in command


def test_discrete_has_no_continuous_slots():
    command = build_command(args(method="discrete"))
    assert command[command.index("--num_registers") + 1] == "0"
    assert command[command.index("--tail_length") + 1] == "4"


def test_fsdp_only_exposed_for_supported_trainer():
    with pytest.raises(ValueError, match="not FSDP"):
        build_command(args(gpus=2, fsdp=True))
    command = build_command(args(gpus=2, fsdp=True, method="full-sft"))
    assert any(value.endswith("accelerate_fsdp.yaml") for value in command)


def test_dream_requires_remote_code_opt_in():
    with pytest.raises(ValueError, match="trust-remote-code"):
        build_command(args(backbone="dream"))


def test_code_execution_requires_opt_in_before_model_loading(tmp_path):
    run = subprocess.run([sys.executable, "eval/eval.py", "--dataset", "humaneval",
                          "--checkpoint", "does-not-exist", "--output", str(tmp_path / "result.json")],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 2
    assert "credential-free" in run.stderr
    assert not (tmp_path / "result.json").exists()
