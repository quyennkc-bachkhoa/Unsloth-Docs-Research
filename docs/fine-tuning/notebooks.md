---
title: Notebooks
description: "Notebook Colab dựng sẵn để chạy thử fine-tune: SFT, hội thoại, vision, CPT, QAT."
---

# Notebooks

Notebook dựng sẵn là cách nhanh nhất để người mới chạy thử một lượt fine-tune. Docs khuyên bắt đầu từ đây.

Cách dùng:

1. Mở notebook, bấm **Run all**, hoặc lưu về máy.
2. Thay dataset của bạn vào.
3. Train rồi deploy.

Bạn có thể đổi sang model bất kỳ trong notebook. Nếu chạy từng cell trên Colab, phải chạy đúng thứ tự và không bỏ cell nào. Sau đó bạn có thể chép notebook về chạy local (cần Linux, WSL hoặc Windows).

Một số notebook tiêu biểu (Colab):

| Mục đích | Notebook |
| --- | --- |
| SFT cơ bản, Alpaca | [Llama 3.1 (8B) Alpaca](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_\(8B\)-Alpaca.ipynb) |
| Hội thoại (conversational) | [Llama 3.2 (1B + 3B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_\(1B_and_3B\)-Conversational.ipynb) |
| Fine-tune rồi chạy trên Ollama | [Llama 3 (8B) Ollama](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_\(8B\)-Ollama.ipynb) |
| Vision | [Qwen3-VL (8B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_\(8B\)-Vision.ipynb), [Llama 3.2 Vision (11B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_\(11B\)-Vision.ipynb) |
| Continued pretraining | [Mistral v0.3 (7B) CPT](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_\(7B\)-CPT.ipynb) |
| QAT | [Qwen3 (4B) QAT](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(4B\)_Instruct-QAT.ipynb) |
| Sinh dữ liệu tổng hợp | [Synthetic Data Llama 3.2 (3B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_\(3B\).ipynb) |

**Model lớn.** Trang Notebooks ghi Colab miễn phí có 15 GB VRAM; tutorial Llama-3 lại ghi GPU miễn phí 16GB (xem hộp "Docs chưa thống nhất" ở trang [Hiệu năng và benchmark](/fine-tuning/benchmark)). Model vượt mức này cần gói Colab trả phí hoặc credits. Docs nói với GPU 80 GB trên Colab, bạn có thể fine-tune model 120B.

Danh sách đầy đủ, gồm cả Kaggle, GRPO, TTS, embedding, có ở [trang Unsloth Notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks) và [repo GitHub](https://github.com/unslothai/notebooks/).

**Nguồn:** https://unsloth.ai/docs/get-started/unsloth-notebooks, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama
