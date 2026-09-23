# Third-party materials

This repository includes the Qwen Image 2.1 prompt-rewrite system prompt templates in `prompts/`. They originate from [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite/prompts) and remain subject to the [Qwen Research License Agreement](LICENSE-QWEN-RESEARCH). The templates are included to run the corresponding Qwen prompt-enhancer checkpoints and are not relicensed as part of this node's own code.

Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.

The GGUF weights are **not** included in the GitHub or ComfyUI Registry package. The separate [Hugging Face mirror](https://huggingface.co/t8star/qwen-image-2.1-comfy) credits the original Qwen checkpoints, quantizers, and the Heretic derivative; its model card carries the applicable license and provenance.

The GitHub Release also distributes a ComfyUI key-converted version of [Viggle's Qwen-Image-2.1 turbo LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo). Viggle trained the adapter; T8mars changed only the tensor key names. See [VIGGLE-LORA-NOTICE.txt](VIGGLE-LORA-NOTICE.txt) for attribution, hashes, and the modification notice. The LoRA remains under the Qwen Research License and is not included in the source tree or Registry package.
