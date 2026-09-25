---
title: Nguồn tham khảo
description: Toàn bộ URL đã dùng trên website (cả hai route), kèm trang dùng nguồn và ngày truy cập.
---

# Nguồn tham khảo

Danh sách gồm toàn bộ URL được trích dẫn trên website, cả route Unsloth lẫn route Kiến thức nền LLM, sinh tự động từ nội dung các trang. Link chỉ nằm trong code block (URL trong lệnh cài đặt) và link notebook/tải xuống chép từ docs không được tính là nguồn.

**Ngày truy cập:** toàn bộ nguồn được đọc ngày **2026-09-24**. Docs Unsloth được đọc qua bản Markdown (thêm đuôi `.md` vào URL), theo mục lục [llms.txt](https://unsloth.ai/docs/llms.txt).

Tổng cộng **267** nguồn.

::: info Phạm vi nguồn
- Route Unsloth: chỉ dùng [docs Unsloth](https://unsloth.ai/docs) và [GitHub unslothai/unsloth](https://github.com/unslothai/unsloth).
- Route Kiến thức nền: ưu tiên docs Unsloth; nguồn ngoài giới hạn ở paper arXiv, tài liệu và blog chính thức của Hugging Face, PyTorch, llama.cpp, NVIDIA, cùng model card chính thức của hãng phát hành model. Nội dung lấy từ các nguồn này được gắn nhãn **[Nguồn ngoài]**.
- Route Bài toán 2×L40S: ngoài docs Unsloth còn dùng văn bản pháp luật, bảng xếp hạng, docs vLLM, Ollama, dataset và báo chí. Nội dung từ các nguồn này cũng gắn nhãn **[Nguồn ngoài]**.
:::

## Docs Unsloth (93)

| Nguồn | Dùng ở trang |
| --- | --- |
| [Unsloth Docs](https://unsloth.ai/docs) | [/](/), [/cai-dat](/cai-dat), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan) |
| [How to use Unsloth as an API endpoint](https://unsloth.ai/docs/basics/api) | [/](/), [/export-deploy/lan-remote](/export-deploy/lan-remote), [/inference/](/inference/), [/inference/api](/inference/api), [/inference/coding-agent](/inference/coding-agent), [/inference/studio-chat](/inference/studio-chat), [/inference/tool-calling](/inference/tool-calling), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan), [/ung-dung-rag](/ung-dung-rag) |
| [Chat Templates](https://unsloth.ai/docs/basics/chat-templates) | [/du-lieu/chat-template](/du-lieu/chat-template), [/du-lieu/checklist](/du-lieu/checklist), [/du-lieu/dinh-dang](/du-lieu/dinh-dang), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [How to Run Local LLMs with Claude Code](https://unsloth.ai/docs/basics/claude-code) | [/inference/api](/inference/api), [/inference/coding-agent](/inference/coding-agent), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [How to Run Local LLMs with OpenAI Codex](https://unsloth.ai/docs/basics/codex) | [/inference/api](/inference/api), [/inference/coding-agent](/inference/coding-agent), [/inference/studio-chat](/inference/studio-chat) |
| [Continued Pretraining](https://unsloth.ai/docs/basics/continued-pretraining) | [/fine-tuning/mo-rong](/fine-tuning/mo-rong), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [Unsloth Dynamic 3.0 GGUFs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) | [/export-deploy/gguf](/export-deploy/gguf), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Fine-tuning Embedding Models with Unsloth Guide](https://unsloth.ai/docs/basics/embedding-finetuning) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/ung-dung-rag](/ung-dung-rag) |
| [Fine-tune MoE Models 12x Faster with Unsloth](https://unsloth.ai/docs/basics/faster-moe) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [Finetuning from Last Checkpoint](https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint) | [/fine-tuning/danh-gia](/fine-tuning/danh-gia), [/fine-tuning/mo-rong](/fine-tuning/mo-rong), [/fine-tuning/quy-trinh](/fine-tuning/quy-trinh), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [How to Serve Local LLMs Anywhere: Secure Remote Access with Cloudflare and Unsloth](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth) | [/export-deploy/lan-remote](/export-deploy/lan-remote), [/export-deploy/loi-thuong-gap](/export-deploy/loi-thuong-gap), [/ung-dung-rag](/ung-dung-rag) |
| [Inference & Deployment](https://unsloth.ai/docs/basics/inference-and-deployment) | [/bai-toan/](/bai-toan/), [/bai-toan/hosting](/bai-toan/hosting), [/export-deploy/](/export-deploy/), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [llama-server & OpenAI endpoint Deployment Guide](https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint) | [/bai-toan/hosting](/bai-toan/hosting), [/export-deploy/chay-model](/export-deploy/chay-model) |
| [Deploying models to LM Studio](https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio) | [/export-deploy/chay-model](/export-deploy/chay-model), [/export-deploy/gguf](/export-deploy/gguf), [/export-deploy/loi-thuong-gap](/export-deploy/loi-thuong-gap) |
| [Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf) | [/export-deploy/](/export-deploy/), [/export-deploy/gguf](/export-deploy/gguf), [/export-deploy/loi-thuong-gap](/export-deploy/loi-thuong-gap), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Saving models to Ollama](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama) | [/bai-toan/fine-tune](/bai-toan/fine-tune), [/bai-toan/hosting](/bai-toan/hosting), [/export-deploy/](/export-deploy/), [/export-deploy/chay-model](/export-deploy/chay-model), [/export-deploy/gguf](/export-deploy/gguf) |
| [Troubleshooting Inference](https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference) | [/export-deploy/](/export-deploy/), [/export-deploy/loi-thuong-gap](/export-deploy/loi-thuong-gap), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [Unsloth Inference](https://unsloth.ai/docs/basics/inference-and-deployment/unsloth-inference) | [/tong-quan](/tong-quan) |
| [vLLM Deployment & Inference Guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide) | [/bai-toan/](/bai-toan/), [/bai-toan/hosting](/bai-toan/hosting), [/export-deploy/](/export-deploy/), [/export-deploy/chay-model](/export-deploy/chay-model), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [LoRA Hot Swapping Guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/lora-hot-swapping-guide) | [/bai-toan/hosting](/bai-toan/hosting), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [How to Serve Local AI Models from Any Device on Your Network with Unsloth LAN Access](https://unsloth.ai/docs/basics/lan) | [/export-deploy/lan-remote](/export-deploy/lan-remote), [/export-deploy/loi-thuong-gap](/export-deploy/loi-thuong-gap), [/ung-dung-rag](/ung-dung-rag) |
| [How to Use MCP Servers with Local LLMs](https://unsloth.ai/docs/basics/mcp) | [/inference/mcp](/inference/mcp) |
| [Multi-GPU Fine-tuning with Unsloth](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) | [/bai-toan/](/bai-toan/), [/bai-toan/fine-tune](/bai-toan/fine-tune), [/fine-tuning/mo-rong](/fine-tuning/mo-rong) |
| [Run Unsloth Dynamic NVFP4 Guide](https://unsloth.ai/docs/basics/nvfp4) | [/bai-toan/hosting](/bai-toan/hosting), [/export-deploy/](/export-deploy/), [/export-deploy/nvfp4-fp8](/export-deploy/nvfp4-fp8), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [Text-to-Speech (TTS) Fine-tuning Guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh) |
| [Tool Calling Guide for Local LLMs](https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms) | [/inference/tool-calling](/inference/tool-calling), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [Unsloth Benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks) | [/fine-tuning/benchmark](/fine-tuning/benchmark), [/fine-tuning/loi-thuong-gap](/fine-tuning/loi-thuong-gap), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [Vision Fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning) | [/du-lieu/checklist](/du-lieu/checklist), [/du-lieu/dinh-dang](/du-lieu/dinh-dang), [/fine-tuning/mo-rong](/fine-tuning/mo-rong), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh) |
| [500K Context Length Fine-tuning](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning) | [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora) |
| [Quantization-Aware Training (QAT)](https://unsloth.ai/docs/blog/quantization-aware-training-qat) | [/fine-tuning/mo-rong](/fine-tuning/mo-rong), [/fine-tuning/quy-trinh](/fine-tuning/quy-trinh), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [Introducing Unsloth Desktop](https://unsloth.ai/docs/desktop) | [/](/), [/lo-trinh-hoc](/lo-trinh-hoc), [/model-catalog](/model-catalog), [/tong-quan](/tong-quan) |
| [Fine-tuning for Beginners](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners) | [/](/), [/bai-toan/fine-tune](/bai-toan/fine-tune), [/fine-tuning/](/fine-tuning/), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [FAQ + Is Fine-tuning Right For Me?](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me) | [/bai-toan/](/bai-toan/), [/bai-toan/fine-tune](/bai-toan/fine-tune), [/fine-tuning/](/fine-tuning/), [/fine-tuning/chon-cach-train](/fine-tuning/chon-cach-train), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/ung-dung-rag](/ung-dung-rag) |
| [Unsloth Requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | [/bai-toan/fine-tune](/bai-toan/fine-tune), [/cai-dat](/cai-dat), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) | [/bai-toan/](/bai-toan/), [/bai-toan/fine-tune](/bai-toan/fine-tune), [/du-lieu/](/du-lieu/), [/du-lieu/checklist](/du-lieu/checklist), [/du-lieu/synthetic](/du-lieu/synthetic), [/fine-tuning/](/fine-tuning/), [/fine-tuning/benchmark](/fine-tuning/benchmark), [/fine-tuning/chon-cach-train](/fine-tuning/chon-cach-train), [/fine-tuning/danh-gia](/fine-tuning/danh-gia), [/fine-tuning/hyperparameter](/fine-tuning/hyperparameter), [/fine-tuning/loi-thuong-gap](/fine-tuning/loi-thuong-gap), [/fine-tuning/notebooks](/fine-tuning/notebooks), [/fine-tuning/quy-trinh](/fine-tuning/quy-trinh), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh), [/du-lieu/](/du-lieu/), [/du-lieu/chat-template](/du-lieu/chat-template), [/du-lieu/checklist](/du-lieu/checklist), [/du-lieu/dinh-dang](/du-lieu/dinh-dang), [/du-lieu/studio](/du-lieu/studio), [/du-lieu/synthetic](/du-lieu/synthetic), [/fine-tuning/chon-cach-train](/fine-tuning/chon-cach-train), [/fine-tuning/loi-thuong-gap](/fine-tuning/loi-thuong-gap), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [LoRA fine-tuning Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) | [/du-lieu/chat-template](/du-lieu/chat-template), [/fine-tuning/chon-cach-train](/fine-tuning/chon-cach-train), [/fine-tuning/danh-gia](/fine-tuning/danh-gia), [/fine-tuning/hyperparameter](/fine-tuning/hyperparameter), [/fine-tuning/loi-thuong-gap](/fine-tuning/loi-thuong-gap), [/fine-tuning/quy-trinh](/fine-tuning/quy-trinh), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Tutorial: How to Finetune Llama-3 and Use In Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama) | [/du-lieu/chat-template](/du-lieu/chat-template), [/du-lieu/checklist](/du-lieu/checklist), [/du-lieu/dinh-dang](/du-lieu/dinh-dang), [/fine-tuning/benchmark](/fine-tuning/benchmark), [/fine-tuning/chon-cach-train](/fine-tuning/chon-cach-train), [/fine-tuning/danh-gia](/fine-tuning/danh-gia), [/fine-tuning/hyperparameter](/fine-tuning/hyperparameter), [/fine-tuning/loi-thuong-gap](/fine-tuning/loi-thuong-gap), [/fine-tuning/notebooks](/fine-tuning/notebooks), [/fine-tuning/quy-trinh](/fine-tuning/quy-trinh), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [What Model Should I Use for Fine-tuning?](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use) | [/du-lieu/](/du-lieu/), [/fine-tuning/chon-cach-train](/fine-tuning/chon-cach-train), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/lo-trinh-hoc](/lo-trinh-hoc) |
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
| [Reinforcement Learning (RL) Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) | [/fine-tuning/](/fine-tuning/), [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc), [/reinforcement-learning/](/reinforcement-learning/), [/reinforcement-learning/grpo](/reinforcement-learning/grpo), [/reinforcement-learning/loi-thuong-gap](/reinforcement-learning/loi-thuong-gap), [/reinforcement-learning/memory-efficient](/reinforcement-learning/memory-efficient), [/reinforcement-learning/reward-function](/reinforcement-learning/reward-function) |
| [Advanced Reinforcement Learning Documentation](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [RL Reward Hacking](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/rl-reward-hacking) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/reinforcement-learning/loi-thuong-gap](/reinforcement-learning/loi-thuong-gap), [/reinforcement-learning/reward-function](/reinforcement-learning/reward-function) |
| [FP8 Reinforcement Learning](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning) | [/export-deploy/](/export-deploy/), [/export-deploy/nvfp4-fp8](/export-deploy/nvfp4-fp8), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/reinforcement-learning/grpo](/reinforcement-learning/grpo), [/reinforcement-learning/loi-thuong-gap](/reinforcement-learning/loi-thuong-gap), [/reinforcement-learning/memory-efficient](/reinforcement-learning/memory-efficient) |
| [Memory Efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl) | [/reinforcement-learning/grpo](/reinforcement-learning/grpo), [/reinforcement-learning/loi-thuong-gap](/reinforcement-learning/loi-thuong-gap), [/reinforcement-learning/memory-efficient](/reinforcement-learning/memory-efficient) |
| [Preference Optimization Training - DPO, ORPO & KTO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/reinforcement-learning/](/reinforcement-learning/), [/reinforcement-learning/dpo-orpo-kto](/reinforcement-learning/dpo-orpo-kto) |
| [Training AI Agents with RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl) | [/reinforcement-learning/agent](/reinforcement-learning/agent) |
| [Tutorial: Train your own Reasoning model with GRPO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo) | [/kien-thuc-nen/rl-va-preference](/kien-thuc-nen/rl-va-preference), [/reinforcement-learning/](/reinforcement-learning/), [/reinforcement-learning/grpo](/reinforcement-learning/grpo), [/reinforcement-learning/loi-thuong-gap](/reinforcement-learning/loi-thuong-gap), [/reinforcement-learning/reward-function](/reinforcement-learning/reward-function) |
| [Unsloth Model Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/model-catalog](/model-catalog) |
| [Unsloth Notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks) | [/fine-tuning/benchmark](/fine-tuning/benchmark), [/fine-tuning/notebooks](/fine-tuning/notebooks) |
| [Connect Curl & HTTP to Unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth) | [/inference/api](/inference/api), [/inference/tool-calling](/inference/tool-calling) |
| [Connect Python SDK to Unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth) | [/inference/api](/inference/api), [/inference/coding-agent](/inference/coding-agent), [/inference/tool-calling](/inference/tool-calling) |
| [Connect API Providers & Model Servers to Unsloth](https://unsloth.ai/docs/integrations/connections) | [/inference/connections](/inference/connections) |
| [Connect Anthropic to Unsloth: Run Claude Models in Local Chat](https://unsloth.ai/docs/integrations/connections/anthropic-claude) | [/inference/connections](/inference/connections) |
| [Connect llama.cpp to Unsloth: Run GGUFs with llama-server](https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server) | [/inference/connections](/inference/connections) |
| [How to Connect Ollama to Unsloth](https://unsloth.ai/docs/integrations/connections/ollama) | [/inference/connections](/inference/connections) |
| [Connect OpenAI to Unsloth: Run GPT Models in Local Chat](https://unsloth.ai/docs/integrations/connections/openai) | [/inference/connections](/inference/connections) |
| [How to Connect OpenRouter to Unsloth: API Key & Model Setup](https://unsloth.ai/docs/integrations/connections/openrouter) | [/inference/connections](/inference/connections) |
| [Connect vLLM to Unsloth for Local Chat Inference](https://unsloth.ai/docs/integrations/connections/vllm) | [/inference/connections](/inference/connections) |
| [How to Run Local AI Models with OpenCode](https://unsloth.ai/docs/integrations/opencode) | [/inference/api](/inference/api), [/inference/coding-agent](/inference/coding-agent) |
| [Run Coding Agents with Local LLMs using Unsloth Start](https://unsloth.ai/docs/integrations/unsloth-start) | [/inference/coding-agent](/inference/coding-agent), [/inference/studio-chat](/inference/studio-chat), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [DeepSeek-V4: How to Run Locally](https://unsloth.ai/docs/models/deepseek-v4) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Gemma 4 - How to Run Locally](https://unsloth.ai/docs/models/gemma-4) | [/bai-toan/chon-model](/bai-toan/chon-model), [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [GLM-5.3 - How to Run Locally](https://unsloth.ai/docs/models/glm-5.3) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [gpt-oss: How to Run Guide](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [IBM Granite 4.1 - How to Run Locally](https://unsloth.ai/docs/models/ibm-granite-4.1) | [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Kimi K3 - How to Run Locally](https://unsloth.ai/docs/models/kimi-k3) | [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Mistral 3.5 - How To Run Locally](https://unsloth.ai/docs/models/mistral-3.5) | [/model-catalog](/model-catalog) |
| [NVIDIA Nemotron 3 Nano - How To Run Guide](https://unsloth.ai/docs/models/nemotron-3) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/model-catalog](/model-catalog) |
| [Qwen3.5 - How to Run Locally](https://unsloth.ai/docs/models/qwen3.5) | [/bai-toan/chon-model](/bai-toan/chon-model), [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Qwen3.5 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) | [/bai-toan/](/bai-toan/), [/bai-toan/chon-model](/bai-toan/chon-model), [/bai-toan/fine-tune](/bai-toan/fine-tune), [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe) |
| [Qwen3.5 GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [Qwen3.8 - How to Run Locally](https://unsloth.ai/docs/models/qwen3.8) | [/bai-toan/chon-model](/bai-toan/chon-model), [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe), [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/model-catalog](/model-catalog) |
| [Large language model (LLMs) Tutorials](https://unsloth.ai/docs/models/tutorials) | [/model-catalog](/model-catalog) |
| [DeepSeek-OCR 2: How to Run & Fine-tune Guide](https://unsloth.ai/docs/models/tutorials/deepseek-ocr-2) | [/model-catalog](/model-catalog) |
| [Llama 4: How to Run & Fine-tune](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune) | [/model-catalog](/model-catalog) |
| [Qwen3-2507: Run Locally Guide](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-2507) | [/kien-thuc-nen/dense-va-moe](/kien-thuc-nen/dense-va-moe) |
| [Qwen3-VL: How to Run Guide](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/model-catalog](/model-catalog) |
| [Introducing Unsloth Studio](https://unsloth.ai/docs/new/studio) | [/](/), [/bai-toan/fine-tune](/bai-toan/fine-tune), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan) |
| [How to Run models with Unsloth Studio](https://unsloth.ai/docs/new/studio/chat) | [/inference/](/inference/), [/inference/studio-chat](/inference/studio-chat), [/inference/tool-calling](/inference/tool-calling), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Unsloth Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe) | [/du-lieu/studio](/du-lieu/studio), [/du-lieu/synthetic](/du-lieu/synthetic), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Export models with Unsloth Studio](https://unsloth.ai/docs/new/studio/export) | [/export-deploy/](/export-deploy/), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Unsloth Studio Installation](https://unsloth.ai/docs/new/studio/install) | [/cai-dat](/cai-dat), [/lo-trinh-hoc](/lo-trinh-hoc) |
| [Get started with Unsloth Studio](https://unsloth.ai/docs/new/studio/start) | [/bai-toan/fine-tune](/bai-toan/fine-tune), [/du-lieu/checklist](/du-lieu/checklist), [/du-lieu/dinh-dang](/du-lieu/dinh-dang), [/du-lieu/studio](/du-lieu/studio), [/fine-tuning/chon-cach-train](/fine-tuning/chon-cach-train), [/fine-tuning/danh-gia](/fine-tuning/danh-gia), [/fine-tuning/hyperparameter](/fine-tuning/hyperparameter), [/fine-tuning/loi-thuong-gap](/fine-tuning/loi-thuong-gap), [/fine-tuning/mo-rong](/fine-tuning/mo-rong), [/fine-tuning/quy-trinh](/fine-tuning/quy-trinh), [/kien-thuc-nen/lora-va-qlora](/kien-thuc-nen/lora-va-qlora), [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan) |

## GitHub Unsloth (4)

| Nguồn | Dùng ở trang |
| --- | --- |
| [unslothai/notebooks](https://github.com/unslothai/notebooks) | [/fine-tuning/notebooks](/fine-tuning/notebooks) |
| [unslothai/notebooks/tree/main/python_scripts](https://github.com/unslothai/notebooks/tree/main/python_scripts) | [/fine-tuning/mo-rong](/fine-tuning/mo-rong) |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | [/](/), [/kien-thuc-nen/kien-truc-transformer](/kien-thuc-nen/kien-truc-transformer), [/lo-trinh-hoc](/lo-trinh-hoc), [/tong-quan](/tong-quan) |
| [unslothai/unsloth/blob/main/unsloth/chat_templates.py](https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py) | [/du-lieu/chat-template](/du-lieu/chat-template) |

## Paper arXiv (30)

| Nguồn | Dùng ở trang |
| --- | --- |
| [Adam: A Method for Stochastic Optimization (arXiv 1412.6980)](https://arxiv.org/abs/1412.6980) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
| [Distilling the Knowledge in a Neural Network (arXiv 1503.02531)](https://arxiv.org/abs/1503.02531) | [/kien-thuc-nen/qua-trinh-huan-luyen](/kien-thuc-nen/qua-trinh-huan-luyen) |
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
| [docs/trl/main/en/dpo_trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) | [/reinforcement-learning/dpo-orpo-kto](/reinforcement-learning/dpo-orpo-kto) |
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
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | [/bai-toan/chon-model](/bai-toan/chon-model), [/kien-thuc-nen/tham-so-va-bo-nho](/kien-thuc-nen/tham-so-va-bo-nho) |
| [datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | [/reinforcement-learning/grpo](/reinforcement-learning/grpo) |
| [datasets/vicgalle/alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) | [/du-lieu/dinh-dang](/du-lieu/dinh-dang) |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh), [/kien-thuc-nen/token-va-context](/kien-thuc-nen/token-va-context) |
| [google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt) | [/kien-thuc-nen/phan-loai-mo-hinh](/kien-thuc-nen/phan-loai-mo-hinh) |
| [settings/tokens](https://huggingface.co/settings/tokens) | [/fine-tuning/quy-trinh](/fine-tuning/quy-trinh) |
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
| [ggml-org/llama.cpp/blob/master/tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | [/bai-toan/hosting](/bai-toan/hosting), [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |

## NVIDIA (5)

| Nguồn | Dùng ở trang |
| --- | --- |
| [developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization) | [/kien-thuc-nen/suy-luan-va-sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| [docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/examples/fp8_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/examples/fp8_primer.html) | [/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |

## Bài toán 2×L40S (66)

Nguồn riêng của phần [Bài toán 2×L40S](/bai-toan/), truy cập ngày **2026-09-25**: văn bản pháp luật, bảng xếp hạng model, docs vLLM và Ollama, dataset Hugging Face, báo chí.

| Nguồn | Dùng ở trang |
| --- | --- |
| [aclanthology.org/2025.vlsp-1.21.pdf](https://aclanthology.org/2025.vlsp-1.21.pdf) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [ai.google.dev/gemma/docs/core/model_card_4](https://ai.google.dev/gemma/docs/core/model_card_4) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [arxiv.org/html/2512.14554v5](https://arxiv.org/html/2512.14554v5) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [chinhphu.vn/?pageid=27160&docid=214590&classid=1&typegroupid=3](https://chinhphu.vn/?pageid=27160&docid=214590&classid=1&typegroupid=3) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [dantri.com.vn/noi-vu/cong-chuc-giam-tai-nho-tro-ly-ao-ai-20260829063246268.htm](https://dantri.com.vn/noi-vu/cong-chuc-giam-tai-nho-tro-ly-ao-ai-20260829063246268.htm) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [discuss.vllm.ai/t/performance-tuning-for-l40s-with-qwen36-35b/2848](https://discuss.vllm.ai/t/performance-tuning-for-l40s-with-qwen36-35b/2848) | [/bai-toan/hosting](/bai-toan/hosting) |
| [docs.ollama.com/api/openai-compatibility](https://docs.ollama.com/api/openai-compatibility) | [/bai-toan/hosting](/bai-toan/hosting) |
| [docs.ollama.com/faq](https://docs.ollama.com/faq) | [/bai-toan/hosting](/bai-toan/hosting) |
| [docs.ollama.com/gpu](https://docs.ollama.com/gpu) | [/bai-toan/hosting](/bai-toan/hosting) |
| [docs.vllm.ai/en/latest](https://docs.vllm.ai/en/latest/) | [/bai-toan/hosting](/bai-toan/hosting) |
| [docs.vllm.ai/en/latest/configuration/engine_args](https://docs.vllm.ai/en/latest/configuration/engine_args/) | [/bai-toan/hosting](/bai-toan/hosting) |
| [docs.vllm.ai/en/latest/features/quantization/gguf](https://docs.vllm.ai/en/latest/features/quantization/gguf/) | [/bai-toan/hosting](/bai-toan/hosting) |
| [docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/) | [/bai-toan/hosting](/bai-toan/hosting) |
| [docs.vllm.ai/en/latest/features/quantization/quantized_kvcache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) | [/bai-toan/hosting](/bai-toan/hosting) |
| [docs.vllm.ai/en/latest/serving/parallelism_scaling.html](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html) | [/bai-toan/hosting](/bai-toan/hosting) |
| [english.luatvietnam.vn/law-no-134-2025-qh15-dated-december-10-2025-of-the-national-assembly-on-artificial-intelligence-422299-doc1.html](https://english.luatvietnam.vn/law-no-134-2025-qh15-dated-december-10-2025-of-the-national-assembly-on-artificial-intelligence-422299-doc1.html) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [fpt.com/vi/tin-tuc/tin-fpt/fpt-song-hanh-cung-bo-tu-phap-ra-mat-cong-phap-luat-quoc-gia](https://fpt.com/vi/tin-tuc/tin-fpt/fpt-song-hanh-cung-bo-tu-phap-ra-mat-cong-phap-luat-quoc-gia) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [github.com/ollama/ollama/blob/main/envconfig/config.go](https://github.com/ollama/ollama/blob/main/envconfig/config.go) | [/bai-toan/hosting](/bai-toan/hosting) |
| [huggingface.co/CMC-OPENAI/CMC-AI-Legal-32B](https://huggingface.co/CMC-OPENAI/CMC-AI-Legal-32B) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [huggingface.co/NaverHustQA/LawVinaLlama](https://huggingface.co/NaverHustQA/LawVinaLlama) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [huggingface.co/Qwen/Qwen3-32B/resolve/main/config.json](https://huggingface.co/Qwen/Qwen3-32B/resolve/main/config.json) | [/bai-toan/hosting](/bai-toan/hosting) |
| [huggingface.co/Qwen/Qwen3.5-27B/resolve/main/config.json](https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/config.json) | [/bai-toan/hosting](/bai-toan/hosting) |
| [huggingface.co/Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [huggingface.co/Viet-Mistral/Vistral-7B-Chat](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [huggingface.co/aisingapore/Qwen-SEA-LION-v4.5-27B-IT](https://huggingface.co/aisingapore/Qwen-SEA-LION-v4.5-27B-IT) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [huggingface.co/arcee-ai/Arcee-VyLinh](https://huggingface.co/arcee-ai/Arcee-VyLinh) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [huggingface.co/datasets/VLSP2025-LegalSML/Public-Test](https://huggingface.co/datasets/VLSP2025-LegalSML/Public-Test) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [huggingface.co/datasets/duyet/vietnamese-legal-instruct](https://huggingface.co/datasets/duyet/vietnamese-legal-instruct) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [huggingface.co/datasets/hirine/dataset-thu-tuc-hanh-chinh-5733-samples](https://huggingface.co/datasets/hirine/dataset-thu-tuc-hanh-chinh-5733-samples) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [huggingface.co/datasets/th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [huggingface.co/meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [huggingface.co/sail/Sailor2-20B-Chat](https://huggingface.co/sail/Sailor2-20B-Chat) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [huggingface.co/unsloth/Llama-3.3-70B-Instruct/resolve/main/config.json](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/resolve/main/config.json) | [/bai-toan/hosting](/bai-toan/hosting) |
| [huggingface.co/vinai/PhoGPT-4B-Chat](https://huggingface.co/vinai/PhoGPT-4B-Chat) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [leaderboard.sea-lion.ai](https://leaderboard.sea-lion.ai/) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [leaderboard.sea-lion.ai/detailed/VI](https://leaderboard.sea-lion.ai/detailed/VI) | [/bai-toan/](/bai-toan/), [/bai-toan/chon-model](/bai-toan/chon-model) |
| [lsvn.vn/doi-tuong-nao-khong-thuoc-pham-vi-bao-ho-quyen-tac-gia-1683458018-a130123.html](https://lsvn.vn/doi-tuong-nao-khong-thuoc-pham-vi-bao-ho-quyen-tac-gia-1683458018-a130123.html) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [lsvn.vn/mot-so-diem-dang-chu-y-tai-luat-so-huu-tri-tue-sua-doi-nam-2025-a167372.html](https://lsvn.vn/mot-so-diem-dang-chu-y-tai-luat-so-huu-tri-tue-sua-doi-nam-2025-a167372.html) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [luatvietnam.vn/hanh-chinh/cong-van-221-cvtltnn-nv-2026-tra-loi-phan-anh-kien-nghi-ve-cong-tac-van-thu-quoc-te-427073-d6.html](https://luatvietnam.vn/hanh-chinh/cong-van-221-cvtltnn-nv-2026-tra-loi-phan-anh-kien-nghi-ve-cong-tac-van-thu-quoc-te-427073-d6.html) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [luatvietnam.vn/khoa-hoc/quyet-dinh-350-qd-ttg-2026-phe-duyet-de-an-chuyen-doi-so-linh-vuc-noi-vu-den-2030-427085-d1.html](https://luatvietnam.vn/khoa-hoc/quyet-dinh-350-qd-ttg-2026-phe-duyet-de-an-chuyen-doi-so-linh-vuc-noi-vu-den-2030-427085-d1.html) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [luatvietnam.vn/tin-van-ban-moi/mo-hinh-ngon-ngu-lon-tieng-viet-va-tro-ly-ao-thuoc-danh-muc-san-pham-cong-nghe-so-trong-diem-186-112245-article.html](https://luatvietnam.vn/tin-van-ban-moi/mo-hinh-ngon-ngu-lon-tieng-viet-va-tro-ly-ao-thuoc-danh-muc-san-pham-cong-nghe-so-trong-diem-186-112245-article.html) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [nhandan.vn/cong-phap-luat-quoc-gia-nen-tang-phap-ly-so-hien-dai-va-toan-dien-post921622.html](https://nhandan.vn/cong-phap-luat-quoc-gia-nen-tang-phap-ly-so-hien-dai-va-toan-dien-post921622.html) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [nhandan.vn/thi-diem-su-dung-tro-ly-ao-ho-tro-can-bo-cong-chuc-tinh-hung-yen-post919132.html](https://nhandan.vn/thi-diem-su-dung-tro-ly-ao-ho-tro-can-bo-cong-chuc-tinh-hung-yen-post919132.html) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [nrl.ai/en/bench](https://www.nrl.ai/en/bench) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [nvidia.com/en-us/data-center/l40s](https://www.nvidia.com/en-us/data-center/l40s/) | [/bai-toan/](/bai-toan/), [/bai-toan/hosting](/bai-toan/hosting) |
| [unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp) | [/bai-toan/fine-tune](/bai-toan/fine-tune) |
| [unsloth.ai/docs/models/gemma-4/train](https://unsloth.ai/docs/models/gemma-4/train) | [/bai-toan/](/bai-toan/), [/bai-toan/chon-model](/bai-toan/chon-model), [/bai-toan/fine-tune](/bai-toan/fine-tune) |
| [unsloth.ai/docs/models/muse-glimmer](https://unsloth.ai/docs/models/muse-glimmer) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [unsloth.ai/docs/models/muse-glimmer/train](https://unsloth.ai/docs/models/muse-glimmer/train) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [unsloth.ai/docs/models/qwen3.6](https://unsloth.ai/docs/models/qwen3.6) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [unsloth.ai/docs/models/qwen3.8/train](https://unsloth.ai/docs/models/qwen3.8/train) | [/bai-toan/chon-model](/bai-toan/chon-model), [/bai-toan/fine-tune](/bai-toan/fine-tune) |
| [vanban.chinhphu.vn](https://vanban.chinhphu.vn/) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [vanban.chinhphu.vn/?pageid=27160&docid=213327](https://vanban.chinhphu.vn/?pageid=27160&docid=213327) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [vanban.chinhphu.vn/?pageid=27160&docid=214592](https://vanban.chinhphu.vn/?pageid=27160&docid=214592) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [vanban.chinhphu.vn/?pageid=27160&docid=216334&classid=1&typegroupid=3](https://vanban.chinhphu.vn/?pageid=27160&docid=216334&classid=1&typegroupid=3) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [vanban.chinhphu.vn/?pageid=27160&docid=216387](https://vanban.chinhphu.vn/?pageid=27160&docid=216387) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [vanban.chinhphu.vn/default.aspx?pageid=27160&docid=199378](https://vanban.chinhphu.vn/default.aspx?pageid=27160&docid=199378) | [/bai-toan/](/bai-toan/), [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [vbpl.vn](https://vbpl.vn/) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [vlsp.org.vn/vlsp2025/eval/legalSLM](https://vlsp.org.vn/vlsp2025/eval/legalSLM) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [vmlu.ai/leaderboard](https://vmlu.ai/leaderboard) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [vnexpress.net/mo-hinh-ngon-ngu-lon-tieng-viet-voi-120-ty-tham-so-5082057.html](https://vnexpress.net/mo-hinh-ngon-ngu-lon-tieng-viet-voi-120-ty-tham-so-5082057.html) | [/bai-toan/chon-model](/bai-toan/chon-model) |
| [xaydungchinhsach.chinhphu.vn/huong-dan-su-dung-chatbot-ai-ho-tro-can-bo-cong-chuc-nguoi-lao-dong-trong-cong-viec-119250405175138265.htm](https://xaydungchinhsach.chinhphu.vn/huong-dan-su-dung-chatbot-ai-ho-tro-can-bo-cong-chuc-nguoi-lao-dong-trong-cong-viec-119250405175138265.htm) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [xaydungchinhsach.chinhphu.vn/nhung-diem-moi-quan-trong-cua-luat-bao-ve-bi-mat-nha-nuoc-co-hieu-luc-tu-1-3-2026-119260301170433415.htm](https://xaydungchinhsach.chinhphu.vn/nhung-diem-moi-quan-trong-cua-luat-bao-ve-bi-mat-nha-nuoc-co-hieu-luc-tu-1-3-2026-119260301170433415.htm) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
| [xaydungchinhsach.chinhphu.vn/quy-dinh-ve-cong-tac-van-thu-quan-ly-van-ban-the-thuc-ky-thuat-trinh-bay-van-ban-ban-sao-van-ban-viet-hoa-mau-trinh-bay-van-ban-hanh-chinh-119251023153016706.htm](https://xaydungchinhsach.chinhphu.vn/quy-dinh-ve-cong-tac-van-thu-quan-ly-van-ban-the-thuc-ky-thuat-trinh-bay-van-ban-ban-sao-van-ban-viet-hoa-mau-trinh-bay-van-ban-hanh-chinh-119251023153016706.htm) | [/bai-toan/van-ban-hanh-chinh](/bai-toan/van-ban-hanh-chinh) |
