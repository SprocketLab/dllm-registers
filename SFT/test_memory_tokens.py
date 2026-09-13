import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_trainer import (
    apply_reconstruction_read_condition,
    build_completion_mask,
    mask_invalid_attention_keys,
    route_primary_register_embeds,
    scale_loss_for_active_ddp_ranks,
)


class MemoryTokenGradientRoutingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.writer = nn.Linear(8, 8, bias=False)
        self.reader = nn.Linear(8, 1, bias=False)
        self.source = torch.randn(1, 4, 8)

    def test_detached_primary_preserves_values_and_blocks_writer_gradient(self):
        live_bridge = self.writer(self.source)
        primary_bridge = route_primary_register_embeds(
            live_bridge,
            detach_primary_register_bridge=True,
        )

        self.assertEqual(primary_bridge.shape, (1, 4, 8))
        self.assertTrue(torch.equal(primary_bridge, live_bridge))
        self.assertFalse(primary_bridge.requires_grad)

        primary_loss = self.reader(primary_bridge).square().mean()
        primary_loss.backward()

        self.assertIsNone(self.writer.weight.grad)
        self.assertIsNotNone(self.reader.weight.grad)
        self.assertGreater(self.reader.weight.grad.abs().sum().item(), 0.0)

    def test_live_primary_reaches_writer(self):
        live_bridge = self.writer(self.source)
        primary_bridge = route_primary_register_embeds(
            live_bridge,
            detach_primary_register_bridge=False,
        )
        self.reader(primary_bridge).square().mean().backward()
        self.assertIsNotNone(self.writer.weight.grad)
        self.assertGreater(self.writer.weight.grad.abs().sum().item(), 0.0)

    def test_reconstruction_still_trains_writer_when_primary_is_detached(self):
        live_bridge = self.writer(self.source)
        primary_bridge = route_primary_register_embeds(
            live_bridge,
            detach_primary_register_bridge=True,
        )
        target = torch.zeros_like(live_bridge)

        primary_loss = self.reader(primary_bridge).square().mean()
        reconstruction_loss = (live_bridge - target).square().mean()
        (primary_loss + 0.05 * reconstruction_loss).backward()

        self.assertIsNotNone(self.writer.weight.grad)
        self.assertGreater(self.writer.weight.grad.abs().sum().item(), 0.0)
        self.assertEqual(live_bridge.shape[1], 4)

    def test_reconstruction_condition_changes_only_completion_queries(self):
        embeds = torch.zeros(1, 6, 8)
        prompt_mask = torch.tensor([[True, True, True, False, False, False]])
        completion_mask = torch.tensor([[False, False, False, True, True, False]])
        condition = torch.ones(1, 1, 8)
        conditioned = apply_reconstruction_read_condition(
            embeds, prompt_mask, condition, completion_mask=completion_mask
        )
        self.assertTrue(torch.equal(conditioned[:, :3], embeds[:, :3]))
        self.assertTrue(torch.equal(conditioned[:, 3:5], torch.ones(1, 2, 8)))
        self.assertTrue(torch.equal(conditioned[:, 5:], embeds[:, 5:]))
        self.assertEqual(conditioned.shape, embeds.shape)

    def test_completion_mask_tracks_real_tokens_not_pad_token_id(self):
        mask = build_completion_mask(
            seq_len=8,
            prompt_length=3,
            completion_length=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(
            mask.tolist(),
            [[False, False, False, True, True, False, False, False]],
        )
        with self.assertRaises(ValueError):
            build_completion_mask(8, 3, 6, torch.device("cpu"))

    def test_padding_tokens_are_blocked_as_attention_keys(self):
        valid_keys = torch.tensor([[True, True, True, True, True, False]])
        existing = torch.zeros(1, 1, 6, 6)
        existing[..., 3:, 1] = -1e4
        masked = mask_invalid_attention_keys(existing, valid_keys)
        self.assertTrue(torch.equal(masked[..., 3:, 1], existing[..., 3:, 1]))
        self.assertTrue(torch.all(masked[..., 5] == -1e4))
        self.assertTrue(torch.all(masked[..., :5] <= 0))

    def test_padding_loss_scale_is_identity_without_distributed_runtime(self):
        loss = torch.tensor(3.0, requires_grad=True)
        self.assertIs(scale_loss_for_active_ddp_ranks(loss, is_padding=False), loss)
        self.assertIs(scale_loss_for_active_ddp_ranks(loss, is_padding=True), loss)


if __name__ == "__main__":
    unittest.main()
