import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_train_chunked import (  # noqa: E402
    is_complete_model_checkpoint,
    is_complete_training_checkpoint,
    save_pretrained_atomic,
    save_training_checkpoint_atomic,
    validate_resume_semantics,
)


class _FakeModel:
    def save_pretrained(self, path):
        path = Path(path)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"weights")


class _FakeTokenizer:
    unk_token_id = 0

    def save_pretrained(self, path):
        Path(path, "tokenizer_config.json").write_text("{}")

    def convert_tokens_to_ids(self, _token):
        return self.unk_token_id


class _FakeAccelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        return None


class _FakeStateful:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}


class CheckpointSafetyTest(unittest.TestCase):
    def test_atomic_save_publishes_success_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "checkpoint-4"
            save_pretrained_atomic(_FakeModel(), _FakeTokenizer(), str(destination))
            self.assertTrue(is_complete_model_checkpoint(str(destination)))
            self.assertEqual((destination / "_SUCCESS").read_text(), "complete\n")
            self.assertFalse(list(Path(tmp).glob("checkpoint-4.tmp-*")))

    def test_atomic_training_save_includes_trajectory_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "checkpoint-4"
            save_training_checkpoint_atomic(
                accelerator=_FakeAccelerator(),
                model=_FakeModel(),
                tokenizer=_FakeTokenizer(),
                optimizer=_FakeStateful("optimizer"),
                scheduler=_FakeStateful("scheduler"),
                destination=str(destination),
                training_progress={
                    "global_step": 4,
                    "epoch": 0,
                    "next_iteration": 4,
                    "chunked_pos": 4,
                    "vanilla_pos": 0,
                },
                hybrid_rng=random.Random(42),
            )
            self.assertTrue(is_complete_training_checkpoint(str(destination), 1))
            self.assertFalse(Path(f"{destination}.tmp").exists())

    def test_missing_indexed_shard_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"one")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors",
                        }
                    }
                )
            )
            self.assertFalse(is_complete_model_checkpoint(str(checkpoint)))

    def test_orphaned_shard_without_index_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model-00001-of-00006.safetensors").write_bytes(b"one")
            self.assertFalse(is_complete_model_checkpoint(str(checkpoint)))

    def test_exact_resume_requires_every_rank_rng_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            (checkpoint / "_SUCCESS").write_text("complete\n")
            (checkpoint / "training_state.pt").write_bytes(b"state")
            (checkpoint / "rng_state_rank0.pt").write_bytes(b"rng0")
            self.assertFalse(is_complete_training_checkpoint(str(checkpoint), 2))
            (checkpoint / "rng_state_rank1.pt").write_bytes(b"rng1")
            self.assertTrue(is_complete_training_checkpoint(str(checkpoint), 2))

    def test_memory_resume_semantics_accept_exact_match(self):
        requested = {"world_size": 2, "train_data_sha256": "abc"}
        config = SimpleNamespace(
            d1_detach_primary_register_bridge=True,
            d1_aux_recon_loss=True,
            d1_aux_recon_task_mode=False,
            d1_aux_recon_condition="completion_mask_offset_v1",
            d1_training_semantics=requested,
        )
        args = SimpleNamespace(
            detach_primary_register_bridge=True,
            aux_recon_loss=True,
            aux_recon_task_mode=False,
        )
        validate_resume_semantics(config, _FakeTokenizer(), args, requested)

    def test_memory_resume_semantics_reject_mismatch(self):
        requested = {"world_size": 2}
        config = SimpleNamespace(
            d1_detach_primary_register_bridge=True,
            d1_aux_recon_loss=True,
            d1_aux_recon_task_mode=False,
            d1_aux_recon_condition="completion_mask_offset_v1",
            d1_training_semantics={"world_size": 1},
        )
        args = SimpleNamespace(
            detach_primary_register_bridge=True,
            aux_recon_loss=True,
            aux_recon_task_mode=False,
        )
        with self.assertRaisesRegex(ValueError, "world_size"):
            validate_resume_semantics(config, _FakeTokenizer(), args, requested)


if __name__ == "__main__":
    unittest.main()
