# Upstream work and licenses

The repository retains the project's Apache License 2.0 in `LICENSE`. This does not replace licenses applying to upstream code, model weights, or datasets.

- **LLaDA**: the local configuration/model implementation and denoising sampler build on [LLaDA](https://github.com/ML-GSAI/LLaDA). The [LLaDA-8B-Base model card](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) declares the MIT license. Local changes include register-state injection support, attention-mask handling, gradient checkpointing, and CPU-first initialization.
- **Dream**: the alternative model is loaded from [Dream-v0-Base-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B). Its [upstream implementation](https://github.com/DreamLM/Dream) is Apache-2.0 licensed. Dream model code and weights are downloaded separately, not bundled here.
- **d1**: the diffusion SFT implementation builds on [d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning](https://github.com/dllm-reasoning/d1). The Apache-2.0 project license is preserved in this release.
- **Training data**: [mix60k](https://huggingface.co/datasets/albertge/mix60k-math-code-sft) combines [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) and [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct). Their respective upstream licenses govern use and redistribution. Data is downloaded separately.
- **Benchmarks**: GSM8K, GSM-Hard, MATH500, Omni-MATH, HumanEval, and MBPP are obtained through their dataset repositories. Follow their individual terms and cite the original benchmark authors.

Please cite the underlying model and dataset work as well as the register-token paper when using this release. Python dependencies are installed separately and retain their own licenses.
