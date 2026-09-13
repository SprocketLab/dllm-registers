"""dllm's LLaDA model with gradient checkpointing support.

Importing this module registers LLaDAModelLM with transformers AutoModel,
so AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct") will use dllm's
implementation (which supports gradient_checkpointing_enable()).
"""
from .configuration_llada import LLaDAConfig
from .modeling_llada import LLaDAModelLM

from transformers import AutoConfig, AutoModel

AutoConfig.register("llada", LLaDAConfig)
AutoModel.register(LLaDAConfig, LLaDAModelLM)
