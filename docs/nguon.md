---
title: Nguồn tham khảo
description: Toàn bộ URL đã dùng trên website (cả hai route), kèm trang dùng nguồn và ngày truy cập.
---

# Nguồn tham khảo

Danh sách gồm toàn bộ URL được trích dẫn trên website, cả route Unsloth lẫn route Kiến thức nền LLM, sinh tự động từ nội dung các trang. Link chỉ nằm trong code block (URL trong lệnh cài đặt) và link notebook/tải xuống chép từ docs không được tính là nguồn.

**Ngày truy cập:** toàn bộ nguồn được đọc ngày **2026-09-24**. Docs Unsloth được đọc qua bản Markdown (thêm đuôi `.md` vào URL), theo mục lục [llms.txt](https://unsloth.ai/docs/llms.txt).

Tổng cộng **200** nguồn.

::: info Phạm vi nguồn
- Route Unsloth: chỉ dùng [docs Unsloth](https://unsloth.ai/docs) và [GitHub unslothai/unsloth](https://github.com/unslothai/unsloth).
- Route Kiến thức nền: ưu tiên docs Unsloth; nguồn ngoài giới hạn ở paper arXiv, tài liệu và blog chính thức của Hugging Face, PyTorch, llama.cpp, NVIDIA, cùng model card chính thức của hãng phát hành model. Nội dung lấy từ các nguồn này được gắn nhãn **[Nguồn ngoài]**.
:::

## Docs Unsloth (93)

| Nguồn | Dùng ở trang |
| --- | --- |
| [Unsloth Docs](https://unsloth.ai/docs) | [/](/), [/cai-dat](/cai-dat), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan) |
| [How to use Unsloth as an API endpoint](https://unsloth.ai/docs/basics/api) | [/](/), [/export-deploy](/export-deploy), [/inference](/inference), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan), [/ung-dung-rag](/ung-dung-rag) |
| [Chat Templates](https://unsloth.ai/docs/basics/chat-templates) | [/du-lieu](/du-lieu), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [How to Run Local LLMs with Claude Code](https://unsloth.ai/docs/basics/claude-code) | [/inference](/inference), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [How to Run Local LLMs with OpenAI Codex](https://unsloth.ai/docs/basics/codex) | [/inference](/inference) |
| [Continued Pretraining](https://unsloth.ai/docs/basics/continued-pretraining) | [/fine-tuning](/fine-tuning), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [Unsloth Dynamic 3.0 GGUFs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) | [/export-deploy](/export-deploy), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Fine-tuning Embedding Models with Unsloth Guide](https://unsloth.ai/docs/basics/embedding-finetuning) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/ung-dung-rag](/ung-dung-rag) |
| [Fine-tune MoE Models 12x Faster with Unsloth](https://unsloth.ai/docs/basics/faster-moe) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [Finetuning from Last Checkpoint](https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint) | [/fine-tuning](/fine-tuning), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [How to Serve Local LLMs Anywhere: Secure Remote Access with Cloudflare and Unsloth](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth) | [/export-deploy](/export-deploy), [/ung-dung-rag](/ung-dung-rag) |
| [Inference & Deployment](https://unsloth.ai/docs/basics/inference-and-deployment) | [/export-deploy](/export-deploy), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [llama-server & OpenAI endpoint Deployment Guide](https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint) | [/export-deploy](/export-deploy) |
| [Deploying models to LM Studio](https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio) | [/export-deploy](/export-deploy) |
| [Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf) | [/export-deploy](/export-deploy), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Saving models to Ollama](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama) | [/export-deploy](/export-deploy) |
| [Troubleshooting Inference](https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference) | [/export-deploy](/export-deploy), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [Unsloth Inference](https://unsloth.ai/docs/basics/inference-and-deployment/unsloth-inference) | [/tong-quan](/tong-quan) |
| [vLLM Deployment & Inference Guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide) | [/export-deploy](/export-deploy), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [LoRA Hot Swapping Guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/lora-hot-swapping-guide) | [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [How to Serve Local AI Models from Any Device on Your Network with Unsloth LAN Access](https://unsloth.ai/docs/basics/lan) | [/export-deploy](/export-deploy), [/ung-dung-rag](/ung-dung-rag) |
| [How to Use MCP Servers with Local LLMs](https://unsloth.ai/docs/basics/mcp) | [/inference](/inference) |
| [Multi-GPU Fine-tuning with Unsloth](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) | [/fine-tuning](/fine-tuning) |
| [Run Unsloth Dynamic NVFP4 Guide](https://unsloth.ai/docs/basics/nvfp4) | [/export-deploy](/export-deploy), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [Text-to-Speech (TTS) Fine-tuning Guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh) |
| [Tool Calling Guide for Local LLMs](https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms) | [/inference](/inference), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [Unsloth Benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks) | [/fine-tuning](/fine-tuning), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [Vision Fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning) | [/du-lieu](/du-lieu), [/fine-tuning](/fine-tuning), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh) |
| [500K Context Length Fine-tuning](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning) | [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [Quantization-Aware Training (QAT)](https://unsloth.ai/docs/blog/quantization-aware-training-qat) | [/fine-tuning](/fine-tuning), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [Introducing Unsloth Desktop](https://unsloth.ai/docs/desktop) | [/](/), [/lo-trinh-hoc](/lo-trinh-hoc), [/model-catalog](/model-catalog), [/tong-quan](/tong-quan) |
| [Fine-tuning for Beginners](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners) | [/](/), [/fine-tuning](/fine-tuning), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [FAQ + Is Fine-tuning Right For Me?](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me) | [/fine-tuning](/fine-tuning), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/ung-dung-rag](/ung-dung-rag) |
| [Unsloth Requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | [/cai-dat](/cai-dat), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) | [/du-lieu](/du-lieu), [/fine-tuning](/fine-tuning), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide) | [/du-lieu](/du-lieu), [/fine-tuning](/fine-tuning), [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [LoRA fine-tuning Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) | [/du-lieu](/du-lieu), [/fine-tuning](/fine-tuning), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Tutorial: How to Finetune Llama-3 and Use In Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama) | [/du-lieu](/du-lieu), [/fine-tuning](/fine-tuning), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [What Model Should I Use for Fine-tuning?](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use) | [/du-lieu](/du-lieu), [/fine-tuning](/fine-tuning), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Unsloth Installation](https://unsloth.ai/docs/get-started/install) | [/](/), [/cai-dat](/cai-dat), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan) |
| [Fine-tuning LLMs on AMD GPUs with Unsloth Guide](https://unsloth.ai/docs/get-started/install/amd) | [/cai-dat](/cai-dat) |
| [Install Unsloth via Docker](https://unsloth.ai/docs/get-started/install/docker) | [/cai-dat](/cai-dat) |
| [Google Colab](https://unsloth.ai/docs/get-started/install/google-colab) | [/cai-dat](/cai-dat), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [Fine-tuning LLMs on Intel GPUs with Unsloth](https://unsloth.ai/docs/get-started/install/intel) | [/cai-dat](/cai-dat), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [Install Unsloth on Linux](https://unsloth.ai/docs/get-started/install/linux) | [/cai-dat](/cai-dat) |
| [Install Unsloth on MacOS](https://unsloth.ai/docs/get-started/install/mac) | [/cai-dat](/cai-dat) |
| [Install Unsloth via pip and uv](https://unsloth.ai/docs/get-started/install/pip-install) | [/cai-dat](/cai-dat), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Updating Unsloth](https://unsloth.ai/docs/get-started/install/updating) | [/cai-dat](/cai-dat) |
| [Install Unsloth on Windows](https://unsloth.ai/docs/get-started/install/windows-installation) | [/cai-dat](/cai-dat), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [Reinforcement Learning (RL) Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc), [/reinforcement-learning](/reinforcement-learning) |
| [Advanced Reinforcement Learning Documentation](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [RL Reward Hacking](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/rl-reward-hacking) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/reinforcement-learning](/reinforcement-learning) |
| [FP8 Reinforcement Learning](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning) | [/export-deploy](/export-deploy), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/reinforcement-learning](/reinforcement-learning) |
| [Memory Efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl) | [/reinforcement-learning](/reinforcement-learning) |
| [Preference Optimization Training - DPO, ORPO & KTO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/reinforcement-learning](/reinforcement-learning) |
| [Training AI Agents with RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl) | [/reinforcement-learning](/reinforcement-learning) |
| [Tutorial: Train your own Reasoning model with GRPO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/reinforcement-learning](/reinforcement-learning) |
| [Unsloth Model Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/model-catalog](/model-catalog) |
| [Unsloth Notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks) | [/fine-tuning](/fine-tuning) |
| [Connect Curl & HTTP to Unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth) | [/inference](/inference) |
| [Connect Python SDK to Unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth) | [/inference](/inference) |
| [Connect API Providers & Model Servers to Unsloth](https://unsloth.ai/docs/integrations/connections) | [/inference](/inference) |
| [Connect Anthropic to Unsloth: Run Claude Models in Local Chat](https://unsloth.ai/docs/integrations/connections/anthropic-claude) | [/inference](/inference) |
| [Connect llama.cpp to Unsloth: Run GGUFs with llama-server](https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server) | [/inference](/inference) |
| [How to Connect Ollama to Unsloth](https://unsloth.ai/docs/integrations/connections/ollama) | [/inference](/inference) |
| [Connect OpenAI to Unsloth: Run GPT Models in Local Chat](https://unsloth.ai/docs/integrations/connections/openai) | [/inference](/inference) |
| [How to Connect OpenRouter to Unsloth: API Key & Model Setup](https://unsloth.ai/docs/integrations/connections/openrouter) | [/inference](/inference) |
| [Connect vLLM to Unsloth for Local Chat Inference](https://unsloth.ai/docs/integrations/connections/vllm) | [/inference](/inference) |
| [How to Run Local AI Models with OpenCode](https://unsloth.ai/docs/integrations/opencode) | [/inference](/inference) |
| [Run Coding Agents with Local LLMs using Unsloth Start](https://unsloth.ai/docs/integrations/unsloth-start) | [/inference](/inference), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [DeepSeek-V4: How to Run Locally](https://unsloth.ai/docs/models/deepseek-v4) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Gemma 4 - How to Run Locally](https://unsloth.ai/docs/models/gemma-4) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [GLM-5.3 - How to Run Locally](https://unsloth.ai/docs/models/glm-5.3) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [gpt-oss: How to Run Guide](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [IBM Granite 4.1 - How to Run Locally](https://unsloth.ai/docs/models/ibm-granite-4.1) | [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Kimi K3 - How to Run Locally](https://unsloth.ai/docs/models/kimi-k3) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Mistral 3.5 - How To Run Locally](https://unsloth.ai/docs/models/mistral-3.5) | [/model-catalog](/model-catalog) |
| [NVIDIA Nemotron 3 Nano - How To Run Guide](https://unsloth.ai/docs/models/nemotron-3) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/model-catalog](/model-catalog) |
| [Qwen3.5 - How to Run Locally](https://unsloth.ai/docs/models/qwen3.5) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Qwen3.5 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe) |
| [Qwen3.5 GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [Qwen3.8 - How to Run Locally](https://unsloth.ai/docs/models/qwen3.8) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Large language model (LLMs) Tutorials](https://unsloth.ai/docs/models/tutorials) | [/model-catalog](/model-catalog) |
| [DeepSeek-OCR 2: How to Run & Fine-tune Guide](https://unsloth.ai/docs/models/tutorials/deepseek-ocr-2) | [/model-catalog](/model-catalog) |
| [Llama 4: How to Run & Fine-tune](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune) | [/model-catalog](/model-catalog) |
| [Qwen3-2507: Run Locally Guide](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-2507) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe) |
| [Qwen3-VL: How to Run Guide](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/model-catalog](/model-catalog) |
| [Introducing Unsloth Studio](https://unsloth.ai/docs/new/studio) | [/](/), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan) |
| [How to Run models with Unsloth Studio](https://unsloth.ai/docs/new/studio/chat) | [/inference](/inference), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Unsloth Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe) | [/du-lieu](/du-lieu), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Export models with Unsloth Studio](https://unsloth.ai/docs/new/studio/export) | [/export-deploy](/export-deploy), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Unsloth Studio Installation](https://unsloth.ai/docs/new/studio/install) | [/cai-dat](/cai-dat), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Get started with Unsloth Studio](https://unsloth.ai/docs/new/studio/start) | [/du-lieu](/du-lieu), [/fine-tuning](/fine-tuning), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan) |

## GitHub Unsloth (4)

| Nguồn | Dùng ở trang |
| --- | --- |
| [unslothai/notebooks](https://github.com/unslothai/notebooks) | [/fine-tuning](/fine-tuning) |
| [unslothai/notebooks/tree/main/python_scripts](https://github.com/unslothai/notebooks/tree/main/python_scripts) | [/fine-tuning](/fine-tuning) |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | [/](/), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan) |
| [unslothai/unsloth/blob/main/unsloth/chat_templates.py](https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py) | [/du-lieu](/du-lieu) |

## Paper arXiv (29)

| Nguồn | Dùng ở trang |
| --- | --- |
| [Adam: A Method for Stochastic Optimization (arXiv 1412.6980)](https://arxiv.org/abs/1412.6980) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [Deep Residual Learning for Image Recognition (arXiv 1512.03385)](https://arxiv.org/abs/1512.03385) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [Training Deep Nets with Sublinear Memory Cost (arXiv 1604.06174)](https://arxiv.org/abs/1604.06174) | [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [Gaussian Error Linear Units (GELUs) (arXiv 1606.08415)](https://arxiv.org/abs/1606.08415) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [Layer Normalization (arXiv 1607.06450)](https://arxiv.org/abs/1607.06450) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (arXiv 1701.06538)](https://arxiv.org/abs/1701.06538) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe) |
| [Attention Is All You Need (arXiv 1706.03762)](https://arxiv.org/abs/1706.03762) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [Proximal Policy Optimization Algorithms (arXiv 1707.06347)](https://arxiv.org/abs/1707.06347) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [Decoupled Weight Decay Regularization (arXiv 1711.05101)](https://arxiv.org/abs/1711.05101) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [The Curious Case of Neural Text Degeneration (arXiv 1904.09751)](https://arxiv.org/abs/1904.09751) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [Root Mean Square Layer Normalization (arXiv 1910.07467)](https://arxiv.org/abs/1910.07467) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [GLU Variants Improve Transformer (arXiv 2002.05202)](https://arxiv.org/abs/2002.05202) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (arXiv 2101.03961)](https://arxiv.org/abs/2101.03961) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe) |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv 2104.09864)](https://arxiv.org/abs/2104.09864) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [LoRA: Low-Rank Adaptation of Large Language Models (arXiv 2106.09685)](https://arxiv.org/abs/2106.09685) | [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [8-bit Optimizers via Block-wise Quantization (arXiv 2110.02861)](https://arxiv.org/abs/2110.02861) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [Training language models to follow instructions with human feedback (arXiv 2203.02155)](https://arxiv.org/abs/2203.02155) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [Reducing Activation Recomputation in Large Transformer Models (arXiv 2205.05198)](https://arxiv.org/abs/2205.05198) | [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (arXiv 2208.07339)](https://arxiv.org/abs/2208.07339) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [FP8 Formats for Deep Learning (arXiv 2209.05433)](https://arxiv.org/abs/2209.05433) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (arXiv 2305.13245)](https://arxiv.org/abs/2305.13245) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [QLoRA: Efficient Finetuning of Quantized LLMs (arXiv 2305.14314)](https://arxiv.org/abs/2305.14314) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (arXiv 2305.18290)](https://arxiv.org/abs/2305.18290) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [Efficient Memory Management for Large Language Model Serving with PagedAttention (arXiv 2309.06180)](https://arxiv.org/abs/2309.06180) | [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [Microscaling Data Formats for Deep Learning (arXiv 2310.10537)](https://arxiv.org/abs/2310.10537) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (arXiv 2312.03732)](https://arxiv.org/abs/2312.03732) | [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [KTO: Model Alignment as Prospect Theoretic Optimization (arXiv 2402.01306)](https://arxiv.org/abs/2402.01306) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (arXiv 2402.03300)](https://arxiv.org/abs/2402.03300) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [ORPO: Monolithic Preference Optimization without Reference Model (arXiv 2403.07691)](https://arxiv.org/abs/2403.07691) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |

## Hugging Face: docs, LLM Course, blog chính thức (38)

| Nguồn | Dùng ở trang |
| --- | --- |
| [blog/moe](https://huggingface.co/blog/moe) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe) |
| [docs/hub/gguf](https://huggingface.co/docs/hub/gguf) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [docs/peft/conceptual_guides/lora](https://huggingface.co/docs/peft/conceptual_guides/lora) | [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [docs/peft/developer_guides/lora](https://huggingface.co/docs/peft/developer_guides/lora) | [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [docs/peft/main/en/package_reference/lora](https://huggingface.co/docs/peft/main/en/package_reference/lora) | [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [docs/transformers/cache_explanation](https://huggingface.co/docs/transformers/cache_explanation) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [docs/transformers/chat_extras](https://huggingface.co/docs/transformers/chat_extras) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [docs/transformers/chat_templating](https://huggingface.co/docs/transformers/chat_templating) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [docs/transformers/en/model_doc/llama](https://huggingface.co/docs/transformers/en/model_doc/llama) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [docs/transformers/en/model_doc/mixtral](https://huggingface.co/docs/transformers/en/model_doc/mixtral) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [docs/transformers/en/model_doc/qwen3_moe](https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [docs/transformers/en/modular_transformers](https://huggingface.co/docs/transformers/en/modular_transformers) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [docs/transformers/generation_strategies](https://huggingface.co/docs/transformers/generation_strategies) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [docs/transformers/glossary](https://huggingface.co/docs/transformers/glossary) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [docs/transformers/grad_checkpointing](https://huggingface.co/docs/transformers/grad_checkpointing) | [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [docs/transformers/kv_cache](https://huggingface.co/docs/transformers/kv_cache) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [docs/transformers/llm_tutorial](https://huggingface.co/docs/transformers/llm_tutorial) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [docs/transformers/main_classes/optimizer_schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [docs/transformers/mixed_precision_training](https://huggingface.co/docs/transformers/mixed_precision_training) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [docs/transformers/model_memory_anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy) | [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [docs/transformers/perf_train_gpu_one](https://huggingface.co/docs/transformers/perf_train_gpu_one) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [docs/transformers/quantization/bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [docs/transformers/quantization/overview](https://huggingface.co/docs/transformers/quantization/overview) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [docs/transformers/tokenizer_summary](https://huggingface.co/docs/transformers/tokenizer_summary) | [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [docs/trl/dpo_trainer](https://huggingface.co/docs/trl/dpo_trainer) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [docs/trl/grpo_trainer](https://huggingface.co/docs/trl/grpo_trainer) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [docs/trl/kto_trainer](https://huggingface.co/docs/trl/kto_trainer) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [docs/trl/main/en/dpo_trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) | [/reinforcement-learning](/reinforcement-learning) |
| [docs/trl/orpo_trainer](https://huggingface.co/docs/trl/orpo_trainer) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [docs/trl/reward_trainer](https://huggingface.co/docs/trl/reward_trainer) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [docs/trl/sft_trainer](https://huggingface.co/docs/trl/sft_trainer) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [learn/llm-course/chapter1/4](https://huggingface.co/learn/llm-course/chapter1/4) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [learn/llm-course/chapter1/6](https://huggingface.co/learn/llm-course/chapter1/6) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh) |
| [learn/llm-course/chapter12/2](https://huggingface.co/learn/llm-course/chapter12/2) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference) |
| [learn/llm-course/chapter2/4](https://huggingface.co/learn/llm-course/chapter2/4) | [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [learn/llm-course/chapter3/5](https://huggingface.co/learn/llm-course/chapter3/5) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [learn/llm-course/chapter6/1](https://huggingface.co/learn/llm-course/chapter6/1) | [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [learn/llm-course/chapter6/2](https://huggingface.co/learn/llm-course/chapter6/2) | [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |

## Model card và config chính thức (14)

| Nguồn | Dùng ở trang |
| --- | --- |
| [Qwen/Qwen3-14B/blob/main/config.json](https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json) | [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [Qwen/Qwen3-32B/blob/main/config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) | [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh) |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh) |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | [/reinforcement-learning](/reinforcement-learning) |
| [datasets/vicgalle/alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) | [/du-lieu](/du-lieu) |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh) |
| [settings/tokens](https://huggingface.co/settings/tokens) | [/fine-tuning](/fine-tuning) |
| [unsloth/Llama-3.2-3B-Instruct/blob/main/config.json](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct/blob/main/config.json) | [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [unsloth/Llama-3.3-70B-Instruct/blob/main/config.json](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/config.json) | [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [unsloth/Meta-Llama-3.1-8B-Instruct/blob/main/config.json](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/blob/main/config.json) | [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |

## PyTorch (13)

| Nguồn | Dùng ở trang |
| --- | --- |
| [docs/stable/amp.html](https://docs.pytorch.org/docs/stable/amp.html) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [docs/stable/checkpoint.html](https://docs.pytorch.org/docs/stable/checkpoint.html) | [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [docs/stable/generated/torch.nn.CrossEntropyLoss.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [docs/stable/generated/torch.nn.GELU.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [docs/stable/generated/torch.nn.LayerNorm.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [docs/stable/generated/torch.nn.RMSNorm.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [docs/stable/generated/torch.nn.functional.softmax.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer) |
| [docs/stable/generated/torch.optim.AdamW.html](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [docs/stable/generated/torch.optim.SGD.html](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [docs/stable/optim.html](https://docs.pytorch.org/docs/stable/optim.html) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [docs/stable/tensor_attributes.html](https://docs.pytorch.org/docs/stable/tensor_attributes.html) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [tutorials/beginner/basics/optimization_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [tutorials/beginner/blitz/autograd_tutorial.html](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |

## llama.cpp / ggml (4)

| Nguồn | Dùng ở trang |
| --- | --- |
| [ggml-org/ggml/blob/master/docs/gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [ggml-org/llama.cpp/blob/master/tools/quantize/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [ggml-org/llama.cpp/blob/master/tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |

## NVIDIA (5)

| Nguồn | Dùng ở trang |
| --- | --- |
| [developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/examples/fp8_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/examples/fp8_primer.html) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
