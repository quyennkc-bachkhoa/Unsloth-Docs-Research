---
title: "Chạy model trong Studio Chat"
description: "Tải và chọn model, tính năng của trang Chat, tham số sampling và lệnh unsloth run."
---

# Chạy model local trong Studio Chat

Đây là cách đơn giản nhất: bạn tải một model về máy rồi chat với nó ngay trong Studio. Trang này đi qua việc chọn model, các tính năng của trang Chat và cách chỉnh tham số sampling.

## Tải và chọn model

- Mở dropdown **Select model** ở góc trên bên trái trang Chat, hoặc mở tab **Model hub**. Chọn model và mức quantization vừa với máy. Quantization (lượng tử hóa) là nén trọng số để model nhẹ hơn. Tải về xong là chat được ngay.
- Studio tìm và tải model từ Hugging Face, hoặc dùng file có sẵn trên máy. Các định dạng được hỗ trợ:
  - GGUF: định dạng file model của llama.cpp, thường đã lượng tử hóa.
  - safetensors: định dạng lưu trọng số chuẩn của Hugging Face.
  - LoRA adapter.
  - Model vision-language và model text-to-speech.
- Tài liệu lấy ví dụ model `unsloth/gemma-4-26B-A4B-it-GGUF` với quant khuyến nghị `UD-Q4_K_XL`.

::: tip Kiến thức nền
Đọc tên model/GGUF như `26B-A4B`, `UD-Q4_K_XL`: xem [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) và [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).
:::

**Model GGUF đã tải sẵn từ trước:** bạn không cần tải lại.

- Studio tự phát hiện model cũ đã tải qua Hugging Face, LM Studio… (cập nhật 27/3).
- Bạn cũng chọn được một thư mục có sẵn để Studio quét (cập nhật 1/4).
- Studio đọc model từ Hugging Face Hub cache. File GGUF của LM Studio nằm ở thư mục riêng, mặc định llama.cpp không nhìn thấy. Vì vậy bạn cần chép hoặc chuyển các file `.gguf` sang HF cache, hoặc sang một đường dẫn llama.cpp đọc được.

::: warning Docs chưa thống nhất
Docs Unsloth viết đường dẫn cache trên Windows theo nhiều cách, ngay trong cùng một trang [new/studio/chat](https://unsloth.ai/docs/new/studio/chat):

| Thư mục | Cách viết 1 | Cách viết 2 |
| --- | --- | --- |
| Hugging Face Hub cache | `C:\Users{your_username}.cache\huggingface\hub` (mục "Using old / existing GGUF models") | `C:\Users\<username>\.cache\huggingface\hub\` và `%USERPROFILE%\.cache\huggingface\hub\` (mục "Deleting model files") |
| Model của LM Studio | `C:\Users\{your_username}.cache\lm-studio\models` | `C:\Users{your_username}\lm-studio\models` |
:::

**Xóa model:** bấm biểu tượng thùng rác trong model search. Cách khác là xóa thư mục của model trong HF cache:

- macOS, Linux, WSL: `~/.cache/huggingface/hub/`
- Windows: `%USERPROFILE%\.cache\huggingface\hub\`

Nếu bạn đã đặt biến `HF_HUB_CACHE` hoặc `HF_HOME`, cache nằm ở thư mục đó. Trên Linux và WSL, biến `XDG_CACHE_HOME` cũng đổi được thư mục gốc của cache.

**Nguồn:** https://unsloth.ai/docs/new/studio/chat, https://unsloth.ai/docs/basics/api

## Tính năng của Chat

Trang Chat không chỉ để hỏi đáp. Model còn được chạy code, đọc web và dùng file bạn đính kèm.

| Tính năng | Mô tả ngắn |
| --- | --- |
| Code execution | Model chạy được Bash và Python trong sandbox, không chỉ JavaScript. Model dùng nó để thử code, tạo file, kiểm chứng đáp án bằng tính toán thật |
| Auto-healing tool calling | Tự sửa tool call (lời gọi công cụ) sai định dạng. Docs Unsloth ghi giảm lỗi 50% |
| Advanced web search | Vào thẳng trang web để đọc nội dung, thay vì chỉ đọc tóm tắt. Dùng API của DuckDuckGo |
| Chat Workspace | Nơi nhập prompt và đính kèm tài liệu, ảnh (webp, png), file code, txt, audio. Có nút bật tắt **Thinking** và **Web search** |
| Thêm file làm context | Đính kèm PDF, ảnh chụp màn hình, DOCX… File được xử lý ngay trên máy |
| Model Arena | So sánh 2 model cạnh nhau với cùng prompt, ví dụ base model và LoRA adapter sau fine-tune. Hiện Studio nạp lần lượt từng model. Chạy song song "đang được phát triển" |
| Multi-GPU | Máy có nhiều GPU thì Chat tự dùng hết khi inference |
| Connect Providers | Dùng model server local (llama.cpp, vLLM, Ollama), hoặc model cloud, trong cùng giao diện (xem trang [Connections](/inference/connections)) |

Unsloth có công bố kết quả thử nghiệm với `unsloth/Qwen3.5-4B-GGUF (UD-Q4_K_XL)`, khi bật web search, code execution và thinking:

| Chỉ số | Tool calling thường | Tool calling của Unsloth |
| --- | --- | --- |
| XML lọt vào câu trả lời | 10/10 | 0/10 |
| Số lần fetch URL | 0 | 4/10 lần chạy |
| Lần chạy ra đúng tên bài hát | 0/10 | 2/10 |
| Số tool call trung bình | 5.5 | 3.8 |
| Thời gian phản hồi trung bình | 12.3s | 9.8s |

**Nguồn:** https://unsloth.ai/docs/new/studio/chat, https://unsloth.ai/docs/basics/api

## Tham số sampling

Tham số sampling quyết định cách model chọn token tiếp theo. **[Nhận định]** Khi mới bắt đầu, bạn có thể để Studio và `unsloth run` tự chọn, chỉ ghi đè khi cần.

- Với model mới như Qwen3.5, Studio **tự đặt sẵn** temperature, top-p, top-k và MTP để ra kết quả tốt. Bạn vẫn chỉnh tay được. Bạn cũng sửa được system prompt và chat template.
- Bạn không cần chỉnh context length (độ dài ngữ cảnh). Lý do là llama.cpp có "smart auto context": nó chỉ dùng phần context thật sự cần. Context length là gì và vì sao nó quyết định KV cache: xem [Token & context](/kien-thuc-nen/token-va-context), [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho).
- Giá trị mặc định cụ thể của từng tham số trong giao diện Studio: **cần kiểm tra lại** (trang nguồn không ghi số).

::: warning Docs chưa thống nhất
Có cần chỉnh context length hay không? Mỗi trang docs trả lời một kiểu:

- [new/studio/chat](https://unsloth.ai/docs/new/studio/chat): "Context length adjustment is no longer necessary" nhờ smart auto context của llama.cpp.
- [integrations/unsloth-start](https://unsloth.ai/docs/integrations/unsloth-start): có cờ `--context-length` / `--max-seq-length`, ví dụ `--context-length 32768`; không truyền cờ thì Unsloth tự chọn context length.
- [basics/api](https://unsloth.ai/docs/basics/api): ví dụ tăng context bằng `-c 131072`.
- [basics/codex](https://unsloth.ai/docs/basics/codex): khi OOM giữa chừng, "Reduce context in Unsloth **Settings → Inference**".
:::

Khi chạy bằng dòng lệnh `unsloth run` mà không truyền cờ sampling nào, Unsloth tự chọn cấu hình khuyến nghị cho model, như context length, temperature… Muốn ghi đè thì bạn truyền cờ. Các cờ này được chuyển thẳng xuống `llama-server`. Ba ví dụ:

```bash
# Lower randomness and improve reproducibility
unsloth run \
  --model unsloth/Qwen3-1.7B-GGUF \
  --temp 0.6 \
  --seed 42
```

```bash
# Tune token selection and repetition behavior
unsloth run \
  --model unsloth/Qwen3-1.7B-GGUF \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.05 \
  --repeat-penalty 1.1
```

```bash
# Use a larger context window and more CPU threads
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  -c 131072 \
  --threads 32
```

::: details Unsloth không dùng GPU (Docker)
Kiểm tra lần lượt:

- Kéo image mới nhất bằng `docker pull unsloth/unsloth:latest`.
- Chạy container có cấp GPU: thêm `--gpus all` nếu dùng `docker run`, hoặc `capabilities: [gpu]` nếu dùng Docker Compose.
- Trên Linux, cần cài NVIDIA Container Toolkit.
- Trên Windows, kiểm tra `nvcc --version` khớp với CUDA version trong `nvidia-smi`.
:::

**Nguồn:** https://unsloth.ai/docs/new/studio/chat, https://unsloth.ai/docs/basics/api
