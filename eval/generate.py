import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
import inspect
from contextlib import nullcontext


def _model_accepts_kwarg(model, kwarg):
    target = getattr(model, "module", model)
    try:
        return kwarg in inspect.signature(target.forward).parameters
    except (TypeError, ValueError):
        return False


def _forward_with_attention_bias(model, attention_bias=None, **kwargs):
    if attention_bias is None:
        return model(**kwargs)
    if _model_accepts_kwarg(model, "attention_bias"):
        return model(**kwargs, attention_bias=attention_bias)
    if _model_accepts_kwarg(model, "attention_mask"):
        return model(**kwargs, attention_mask=attention_bias)
    return model(**kwargs)


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    register_embeds=None,
    register_positions=None,
    loop_registers=False,
    attention_bias=None,
    autocast_dtype=torch.float16,
):
    """
    Optimized version of the generate function.

    Args:
        register_embeds: Optional tensor (bs, num_registers, hidden_dim) to inject
            at register_positions during each forward pass.
        register_positions: Optional tensor of register token positions in the sequence.
        loop_registers: If True, update register embeddings from hidden states at each denoising step.
        attention_bias: Optional float tensor of shape (1, 1, seq_len, seq_len) or
            (batch_size, 1, seq_len, seq_len). Added to attention scores before softmax —
            0 = allowed, large negative = blocked. Used to enforce structural attention masks
            (e.g. block chunk-completion → prompt-text attention). The seq_len of the bias
            must match prompt.shape[1] + gen_length.
        autocast_dtype: Mixed-precision dtype used during denoising. Pass this
            explicitly when comparing evaluation campaigns across hardware.

    Returns:
        x: Generated token IDs (bs, prompt_len + gen_length)
        final_register_embeds: Register hidden states after generation (bs, num_registers, hidden_dim),
            or None if register_positions is None.
    """
    # Track current register embeddings (updated each step if loop_registers=True)
    current_register_embeds = register_embeds

    # Helper to run a forward pass, optionally injecting register embeddings
    def _forward(model, input_ids, output_hidden=False):
        if current_register_embeds is not None and register_positions is not None:
            embeds = model.get_input_embeddings()(input_ids)
            embeds[:, register_positions, :] = current_register_embeds.to(embeds.dtype)
            out = _forward_with_attention_bias(
                model,
                inputs_embeds=embeds,
                output_hidden_states=output_hidden,
                attention_bias=attention_bias,
            )
        else:
            out = _forward_with_attention_bias(
                model,
                input_ids=input_ids,
                output_hidden_states=output_hidden,
                attention_bias=attention_bias,
            )
        if output_hidden:
            return out.logits, out.hidden_states[-1]
        return out.logits

    # Use mixed precision for faster computation
    autocast = (torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if prompt.device.type == "cuda" else nullcontext())
    with autocast:
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        # Prompt positions are fixed by layout, even when the prompt intentionally
        # contains mask-token placeholders for registers/tail slots.
        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        prompt_index[:, : prompt.shape[1]] = True

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=(dist.is_initialized() and dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            need_hidden = (loop_registers and register_positions is not None)

            for i in range(steps_per_block):
                mask_index = (x == mask_id) & (~prompt_index)

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    logits = _forward(model, x_)
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    if need_hidden:
                        logits, hidden = _forward(model, x, output_hidden=True)
                        current_register_embeds = hidden[:, register_positions, :].clone()
                        del hidden
                    else:
                        logits = _forward(model, x)

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]

        # Extract final register embeddings if registers are active
        final_register_embeds = None
        if register_positions is not None:
            if current_register_embeds is not None:
                embeds = model.get_input_embeddings()(x)
                embeds[:, register_positions, :] = current_register_embeds.to(embeds.dtype)
                hidden = _forward_with_attention_bias(
                    model,
                    inputs_embeds=embeds,
                    output_hidden_states=True,
                    attention_bias=attention_bias,
                ).hidden_states[-1]
            else:
                hidden = _forward_with_attention_bias(
                    model,
                    input_ids=x,
                    output_hidden_states=True,
                    attention_bias=attention_bias,
                ).hidden_states[-1]
            final_register_embeds = hidden[:, register_positions, :].clone()

        return x, final_register_embeds
