---
title: LoRA & QLoRA
description: Full fine-tuning và PEFT, cách LoRA hoạt động, ý nghĩa r / lora_alpha / lora_dropout / target_modules / bias, QLoRA, gradient checkpointing và việc merge adapter trong Unsloth.
---

# LoRA & QLoRA

LoRA và QLoRA là hai phương pháp mặc định khi bạn fine-tune bằng Unsloth. Trang này giải thích cách chúng hoạt động, và ý nghĩa từng tham số trong lời gọi `FastLanguageModel.get_peft_model(...)`. Nên đọc trang này trước khi chọn phương pháp train trong Studio (QLoRA, LoRA hoặc Full Fine-tuning), hoặc trước khi sửa tham số LoRA trong notebook. Các khái niệm loss, learning rate, overfitting được giải thích ở [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen).

## Full fine-tuning và PEFT

Có hai hướng fine-tune: sửa toàn bộ model, hoặc chỉ train thêm một phần nhỏ. Hướng thứ hai tiết kiệm bộ nhớ hơn nhiều, và là hướng Unsloth khuyên dùng trước.

**Khái niệm.**
- **Full fine-tuning** (FFT — tinh chỉnh toàn bộ): cập nhật **mọi** trọng số của model. Cách này phải lưu gradient và optimizer state cho toàn bộ tham số, nên rất tốn bộ nhớ.
- **PEFT** (Parameter-Efficient Fine-Tuning — tinh chỉnh tiết kiệm tham số): giữ nguyên (đóng băng) trọng số gốc, chỉ train một lượng nhỏ tham số thêm vào. LoRA là phương pháp PEFT phổ biến nhất. **[Nguồn ngoài]** https://huggingface.co/docs/peft/conceptual_guides/lora

**Ví dụ.** Docs Unsloth lấy Llama 70B làm ví dụ: model này có 70 tỷ con số. LoRA không sửa cả 70 tỷ con số đó. Thay vào đó, nó thêm các ma trận "mỏng" A và B vào mỗi trọng số và chỉ tối ưu chúng. Phần này chỉ khoảng 1% số trọng số.

**Ảnh hưởng khi dùng Unsloth.**
- Unsloth hỗ trợ FFT (`full_finetuning = True`). Tuy vậy, docs nói FFT thường không cần: nếu làm đúng cách, LoRA có thể ngang FFT. Docs khuyên thử LoRA hoặc QLoRA trước. Nếu LoRA không đạt thì FFT gần như chắc chắn cũng không đạt.
- Trong Studio, bạn chọn phương pháp bằng nút:
  - **QLoRA**: VRAM thấp nhất.
  - **LoRA**: VRAM trung bình.
  - **Full Fine-tuning**: VRAM cao nhất.
- Chỉ được bật một phương pháp tại một thời điểm (`load_in_4bit`, `load_in_16bit`, `load_in_8bit`, `full_finetuning`).
- Vì sao PEFT tiết kiệm: xem ví dụ optimizer state ở [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) và cách ước tính VRAM ở [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho).

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning/lora-qlora-full) (mục LoRA vs QLoRA vs full fine-tuning); docs gốc [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [Unsloth Studio](https://unsloth.ai/docs/new/studio/start).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/new/studio/start, https://huggingface.co/docs/peft/conceptual_guides/lora

## LoRA: cập nhật bằng hai ma trận hạng thấp

LoRA không sửa trực tiếp trọng số của model. Nó học một phần "chỉnh sửa" nhỏ, rồi cộng phần đó vào trọng số gốc.

**Khái niệm.** Mỗi lớp linear trong model có một ma trận trọng số `W` kích thước `d × k`. **LoRA** (Low-Rank Adaptation) giữ `W` đóng băng. Phần "chỉnh sửa" được tách thành tích của hai ma trận nhỏ:
- `B`, kích thước `d × r`.
- `A`, kích thước `r × k`.

Ở đây `r` nhỏ hơn rất nhiều so với `d` và `k`. Tích `B × A` có cùng kích thước với `W`, nhưng số tham số phải học ít hơn hẳn. **[Nguồn ngoài]** https://arxiv.org/abs/2106.09685

Công thức (theo paper LoRA, ở dạng Unsloth dùng trong LoRA Hyperparameters Guide):

```text
W_mới = W + (lora_alpha / r) × B × A
```

- `A` được khởi tạo bằng giá trị ngẫu nhiên nhỏ (Gaussian). `B` được khởi tạo bằng 0. Vì vậy lúc bắt đầu `B × A = 0`: model hành xử y như model gốc, rồi học dần. **[Nguồn ngoài]** https://arxiv.org/abs/2106.09685, https://huggingface.co/docs/peft/developer_guides/lora
- Hệ số `lora_alpha / r` chỉnh "độ mạnh" của phần cập nhật (xem mục `lora_alpha` bên dưới).
- Khi triển khai, bạn có thể cộng hẳn `B × A` vào `W` (merge). Nhờ vậy LoRA không làm chậm inference. **[Nguồn ngoài]** https://arxiv.org/abs/2106.09685

**Ví dụ.** **[Ước tính]** Số tham số LoRA tính theo công thức `r × (d + k)`. Docs Unsloth ghi `m*r + r*n` cho ma trận `m × n`; paper LoRA dùng cùng kích thước B, A.
- Một ma trận `4096 × 4096`: full fine-tuning phải học `4096 × 4096 = 16,777,216` tham số.
- LoRA với `r = 16`: `16 × (4096 + 4096) = 131,072` tham số, khoảng 0.78% so với ma trận gốc.
- Tăng `r` lên 32 thì số tham số LoRA gấp đôi (262,144).

Docs Unsloth cũng đưa ví dụ lớp MLP với `m ≈ 4096`, `n ≈ 12k`, `r ≈ 64`. Khi đó LoRA có khoảng 1 triệu tham số, so với khoảng 48 triệu tham số gốc (~2%).

**Ảnh hưởng khi dùng Unsloth.**
- Kết quả train chỉ là một **adapter** nhỏ (docs: "a small 100MB file called a LoRA adapter"). Khi chạy, bạn vẫn cần model gốc, trừ khi đã merge (mục cuối trang).
- Muốn kiểm tra adapter có thay đổi sau train không, docs Unsloth lưu ý đừng dùng `np.allclose()`. Hàm này có thể bỏ sót thay đổi nhỏ, nhất là ở ma trận A (khởi tạo bằng giá trị Gaussian nhỏ). Hãy dùng một trong các cách: so sánh hash hoặc checksum, tổng chênh lệch tuyệt đối, hoặc `np.array_equal()`.

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning/); docs gốc [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [Faster MoE](https://unsloth.ai/docs/basics/faster-moe).

**Nguồn:** https://arxiv.org/abs/2106.09685, https://huggingface.co/docs/peft/conceptual_guides/lora, https://huggingface.co/docs/peft/developer_guides/lora, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/basics/faster-moe, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama

## Tham số LoRA trong Unsloth

Mục này đi qua từng tham số trong đoạn code gắn LoRA vào model. Đoạn code dưới đây chép nguyên văn từ docs Unsloth ([Google Colab install](https://unsloth.ai/docs/get-started/install/google-colab); bản giống hệt ở [Windows install](https://unsloth.ai/docs/get-started/install/windows-installation)):

```python
# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```

Trong Studio, các tham số này nằm ở nhóm **LoRA Settings**: Rank, Alpha, Dropout, LoRA Variant (`LoRA` / `RS-LoRA` / `LoftQ`), Target Modules. Nhóm này bị ẩn khi bạn chọn Full Fine-tuning.

**Nguồn:** https://unsloth.ai/docs/get-started/install/google-colab, https://unsloth.ai/docs/get-started/install/windows-installation, https://unsloth.ai/docs/new/studio/start

### Rank `r`

**Khái niệm.** `r` (rank — hạng) là "bề rộng" của hai ma trận A và B. Nó quyết định số tham số train được trong adapter. `r` càng lớn thì adapter càng "có sức chứa", nhưng càng tốn bộ nhớ. **[Nguồn ngoài]** https://huggingface.co/docs/peft/main/en/package_reference/lora

**Ví dụ.** Với ma trận `4096 × 4096`: `r = 16` cho 131,072 tham số. `r = 128` cho `128 × 8192 = 1,048,576` tham số **[Ước tính]**, tức gấp 8 lần.

**Ảnh hưởng khi dùng Unsloth.**
- Rank lớn dùng nhiều bộ nhớ hơn và chậm hơn, nhưng có thể chính xác hơn với tác vụ phức tạp. Rank quá lớn có thể gây overfitting.
- Docs gợi ý 8 hoặc 16 cho fine-tune nhanh, tối đa 128. Bảng khuyến nghị ghi "chọn 16 hoặc 32". Studio mặc định `16`.
- Khi underfitting, docs gợi ý tăng rank, nhất là với model nhỏ hoặc dataset phức tạp.
- Docs ghi khoảng rank không thống nhất. Xem hộp cảnh báo cuối mục này.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/new/studio/start, https://huggingface.co/docs/peft/main/en/package_reference/lora

### `lora_alpha` và rsLoRA

**Khái niệm.** `lora_alpha` là hệ số chỉnh độ mạnh của phần cập nhật LoRA. Phần cập nhật được nhân với `lora_alpha / r`.

**rsLoRA** (Rank-Stabilized LoRA) đổi hệ số này thành `lora_alpha / sqrt(r)`. Bạn bật nó bằng `use_rslora = True`. Paper rsLoRA chỉ ra rằng chia cho `r` làm adapter rank cao học chậm và kém hơn. Vì vậy paper đề xuất chia cho căn bậc hai của rank. Kết quả là adapter ổn định hơn và tận dụng tốt hơn rank cao. **[Nguồn ngoài]** https://huggingface.co/docs/peft/developer_guides/lora, https://huggingface.co/docs/peft/main/en/package_reference/lora, https://arxiv.org/abs/2312.03732

**Ví dụ.** **[Ước tính]**
- `r = 16`, `lora_alpha = 16` → hệ số `16 / 16 = 1`.
- `r = 16`, `lora_alpha = 32` → hệ số `2` (cập nhật mạnh gấp đôi).
- `r = 16`, `lora_alpha = 16`, bật rsLoRA → hệ số `16 / sqrt(16) = 16 / 4 = 4`.
- Giữ `lora_alpha = 16` mà tăng `r` từ 16 lên 64: với LoRA thường, hệ số giảm từ 1 xuống 0.25; với rsLoRA, hệ số giảm từ 4 xuống 2.

**Ảnh hưởng khi dùng Unsloth.**
- Docs khuyên hai mốc: `lora_alpha = r` (mốc an toàn) hoặc `lora_alpha = r * 2` (học "mạnh tay" hơn). Tức là `alpha / rank` bằng 1 hoặc 2. Alpha lớn làm model học nhiều hơn từ dataset, nhưng dễ overfit.
- Mẹo chống overfitting sau khi train: giảm alpha lúc inference (ví dụ nhân 0.5). Docs nói việc này tương đương lấy trung bình trọng số của model gốc và model fine-tune.
- Studio: slider Alpha 4–256, mặc định `32`. Chọn LoRA Variant = `RS-LoRA` để bật rsLoRA.
- Các trang docs ghi giá trị mặc định, giá trị khuyến nghị và mô tả `use_rslora` không thống nhất. Xem hộp cảnh báo cuối mục này.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/new/studio/start, https://huggingface.co/docs/peft/developer_guides/lora, https://huggingface.co/docs/peft/main/en/package_reference/lora, https://arxiv.org/abs/2312.03732

### `lora_dropout`

**Khái niệm.** Dropout (bỏ ngẫu nhiên) cho các lớp LoRA. Trong lúc train, một tỉ lệ activation của LoRA bị đặt ngẫu nhiên về 0. Mục đích là giảm học thuộc (regularization). **[Nguồn ngoài]** https://huggingface.co/docs/peft/main/en/package_reference/lora

**Ví dụ.** `lora_dropout = 0.1`: mỗi bước, khoảng 10% activation của nhánh LoRA bị tắt.

**Ảnh hưởng khi dùng Unsloth.**
- Code của Unsloth được tối ưu khi `lora_dropout = 0`, nên train nhanh hơn một chút.
- Docs dẫn nghiên cứu cho thấy với run ngắn, dropout có thể là cách điều chuẩn không đáng tin. Tuy vậy, nếu nghi overfitting thì nên dùng giá trị khác 0 (ví dụ `0.1`).
- Khoảng khuyến nghị là 0–0.1. Studio mặc định khác code mẫu (xem hộp cảnh báo cuối mục này).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/new/studio/start, https://huggingface.co/docs/peft/main/en/package_reference/lora

### `target_modules`

**Khái niệm.** `target_modules` là danh sách các lớp linear được gắn LoRA. Trong các model họ Llama, tên module gồm:
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- MLP: `gate_proj`, `up_proj`, `down_proj`.
Các lớp này là gì trong Transformer: xem [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer).

**Ví dụ.** Thư viện PEFT mặc định chỉ gắn LoRA vào query và value của attention. Paper QLoRA gắn vào mọi lớp linear và đạt hiệu năng ngang full fine-tuning. PEFT hỗ trợ viết tắt `target_modules="all-linear"`. **[Nguồn ngoài]** https://huggingface.co/docs/peft/developer_guides/lora

**Ảnh hưởng khi dùng Unsloth.**
- Docs khuyên gắn vào **cả 7 module** (attention và MLP). Theo biểu đồ trong docs, QLoRA-All cho kết quả tốt nhất, hơn chỉ FFN hoặc chỉ Attention. Bạn có thể bỏ bớt module để tiết kiệm bộ nhớ, nhưng docs "strongly advise against" vì lượng tiết kiệm rất nhỏ.
- Studio mặc định bật cả 7 module. Với model vision, Studio có thêm các lựa chọn Vision Layers / Language Layers / Attention Modules / MLP Modules.
- Continued pretraining: thêm `lm_head` và `embed_tokens` (xem [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen)).
- Với model MoE, docs nói fine-tune lớp router không tốt, nên Unsloth tắt nó theo mặc định (xem [Dense & MoE](/kien-thuc-nen/dense-va-moe)).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/continued-pretraining, https://unsloth.ai/docs/basics/faster-moe, https://huggingface.co/docs/peft/developer_guides/lora

### `bias` và các tham số còn lại

**Khái niệm.** `bias` quyết định có train các hệ số bias của lớp linear hay không. PEFT cho phép ba giá trị: `"none"` (mặc định), `"all"`, `"lora_only"`. Nếu bạn train bias, thì khi tắt adapter, model cũng không còn cho kết quả y như model gốc. **[Nguồn ngoài]** https://huggingface.co/docs/peft/main/en/package_reference/lora

**Ví dụ.** `bias = "none"`: chỉ ma trận A và B được train.

**Ảnh hưởng khi dùng Unsloth.**
- `bias = "none"`: train nhanh hơn và ít bộ nhớ hơn. Train bias thêm tham số mà gần như không có lợi.
- `random_state = 3407`: seed (hạt giống ngẫu nhiên), để các lần chạy tái lập được.
- `loftq_config = None`: LoftQ khởi tạo A và B từ `r` vector kỳ dị lớn nhất của trọng số gốc. Cách này có thể tăng độ chính xác, nhưng bộ nhớ tăng vọt lúc bắt đầu train.

::: warning Docs chưa thống nhất
Các trang docs Unsloth, và đôi khi hai mục trong cùng một trang, ghi giá trị khác nhau cho các tham số LoRA:

| Tham số | Nguồn A | Nguồn B | Nguồn C |
| --- | --- | --- | --- |
| Khoảng rank `r` | 8, 16, 32, 64, 128 — chọn 16 hoặc 32 (bảng khuyến nghị, [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)) | "usually is between 4 and 64" (mục Underfitting, cùng trang [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)) | Slider 4–128, mặc định 16 — [Unsloth Studio](https://unsloth.ai/docs/new/studio/start) |
| `lora_alpha` | `32` — [Unsloth Studio](https://unsloth.ai/docs/new/studio/start) | `16` — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), code [Google Colab install](https://unsloth.ai/docs/get-started/install/google-colab) | `r` hoặc `r * 2` — [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide); code `lora_alpha = lora_rank*2, # *2 speeds up training` — [Faster MoE](https://unsloth.ai/docs/basics/faster-moe) |
| Quan hệ alpha và rank | "keep alpha/rank at least = 1" (tức alpha ≥ rank) — mục LoRA Alpha and Rank relationship, [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) | "Rank should at least equal to the alpha number" (tức rank ≥ alpha) — mục Underfitting, cùng trang [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) | — |
| `lora_dropout` | `0.05` — [Unsloth Studio](https://unsloth.ai/docs/new/studio/start) | `0` — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), code [Google Colab install](https://unsloth.ai/docs/get-started/install/google-colab) | `0` (mặc định) đến `0.1` — [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) |
| `use_rslora` | "Advanced feature to set the `lora_alpha = 16` automatically" — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama) | Nếu `True`, scaling thành `lora_alpha / sqrt(r)` thay vì `lora_alpha / r` — [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) (khớp với PEFT docs) | — |
| `bias` | Code `bias = "none"`, nhưng lời giải thích ghi "Leave this as 0" — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama) | "Leave this as `"none"`" — [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) | — |
:::

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning/quy-trinh) (Code Core tiêu biểu, Hyperparameter khuyến nghị, So sánh mặc định), [Cài đặt](/cai-dat) (code mẫu); docs gốc [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/get-started/install/google-colab, https://unsloth.ai/docs/basics/faster-moe, https://unsloth.ai/docs/basics/continued-pretraining, https://huggingface.co/docs/peft/main/en/package_reference/lora, https://huggingface.co/docs/peft/developer_guides/lora, https://arxiv.org/abs/2312.03732

## QLoRA

QLoRA là LoRA chạy trên một model gốc đã được nén xuống 4-bit. Bạn dùng nó khi VRAM hạn chế, và docs Unsloth khuyên bắt đầu bằng QLoRA.

**Khái niệm.** **QLoRA** (Quantized LoRA) = LoRA + model gốc được lượng tử hóa (quantize) xuống **4-bit**. Model gốc 4-bit bị đóng băng. Gradient được truyền ngược "xuyên qua" nó vào các adapter LoRA. Bản thân adapter vẫn ở độ chính xác cao (16-bit). **[Nguồn ngoài]** https://arxiv.org/abs/2305.14314

Paper QLoRA giới thiệu ba kỹ thuật:
- **NF4** (4-bit NormalFloat): kiểu dữ liệu tối ưu cho trọng số phân bố chuẩn.
- **Double quantization**: lượng tử hóa cả các hằng số lượng tử hóa.
- **Paged optimizers**: xử lý đột biến bộ nhớ.

Nhờ ba kỹ thuật này, paper fine-tune được model 65B trên một GPU 48GB.

4-bit, 16-bit, NF4 nghĩa là gì: xem [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).

**Ví dụ.** Docs Unsloth ghi QLoRA giảm VRAM hơn 75%, tức khoảng 4 lần ít hơn LoRA 16-bit. Llama 70B chạy QLoRA trong Unsloth vừa dưới 48GB VRAM. **[Ước tính]** Chỉ tính riêng trọng số model 8B: 16-bit ≈ `8 tỷ × 2 byte = 16 GB`, 4-bit ≈ `8 tỷ × 0.5 byte = 4 GB`. Con số này chưa tính activation, adapter, optimizer state (cách tính ở [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho)).

**Ảnh hưởng khi dùng Unsloth.**
- Cách bật:
  - `load_in_4bit = True` → QLoRA.
  - Tắt đi (hoặc đặt `load_in_16bit = True`) → LoRA 16-bit.
- Cách đọc tên model:
  - Tên kết thúc bằng `unsloth-bnb-4bit` là Unsloth Dynamic 4-bit. Bản này tốn VRAM hơn BitsAndBytes 4-bit thường một chút, nhưng chính xác hơn đáng kể.
  - `bnb-4bit` là BitsAndBytes 4-bit chuẩn.
- LoRA 16-bit nhanh hơn và chính xác hơn một chút. QLoRA chậm hơn một chút nhưng ít VRAM hơn nhiều. Docs Unsloth khuyên bắt đầu bằng QLoRA.
- Docs lưu ý: train và serve cùng độ chính xác giúp giữ độ chính xác. Muốn serve 4-bit thì train 4-bit, và ngược lại.
- Docs Unsloth có nhắc tới NF4 trong trang gpt-oss (dùng BitsAndBytes NF4 để mô phỏng MXFP4). Tuy vậy, các trang QLoRA chính không nói rõ Unsloth Dynamic 4-bit có dùng NF4 hay không — cần kiểm tra lại.

::: warning Docs chưa thống nhất
Các trang docs đánh giá khác nhau về mức giảm độ chính xác khi dùng QLoRA:

| Tham số | Nguồn A | Nguồn B |
| --- | --- | --- |
| Mức giảm độ chính xác của QLoRA so với LoRA | 4-bit gây "1-2% accuracy degradation" — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama) | Nhờ dynamic 4-bit, mức giảm "largely recovered" — [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide); "now negligible" — [FAQ](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me) |
:::

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning/lora-qlora-full) (LoRA vs QLoRA vs full fine-tuning; đọc hậu tố tên model), [Model catalog](/model-catalog); docs gốc [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).

**Nguồn:** https://arxiv.org/abs/2305.14314, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune

## Gradient checkpointing

Gradient checkpointing giúp bạn train với ít VRAM hơn, đổi lại phải tính toán nhiều hơn một chút. Nó đặc biệt hữu ích khi context dài.

**Khái niệm.** Khi train, lượt backward cần các **activation** (giá trị trung gian) từ lượt forward. Bình thường, tất cả activation được giữ trong VRAM. **Gradient checkpointing** (còn gọi là activation checkpointing) không giữ các activation này. Đến lượt backward, nó tính lại phần forward cần thiết. Tức là đổi thêm tính toán lấy ít bộ nhớ hơn. **[Nguồn ngoài]** https://docs.pytorch.org/docs/stable/checkpoint.html. Theo paper gốc, bộ nhớ cho mạng `n` lớp giảm xuống cỡ `sqrt(n)`. Đổi lại, mỗi mini-batch tốn thêm khoảng một lượt forward. **[Nguồn ngoài]** https://arxiv.org/abs/1604.06174

Đừng nhầm với **checkpoint** (bản lưu tiến trình train) — xem [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen).

**Ví dụ.** Theo blog 500K context của docs Unsloth, với context dài, activation của một lớp decoder có thể vượt 2 GB. Bản `"unsloth"` chuyển (offload) activation từ GPU sang RAM hệ thống ngay khi chúng được tạo ra. Docs ghi cách này giúp context dài hơn khoảng 10 lần, thêm tối đa 0.1% thời gian train, và không ảnh hưởng độ chính xác.

**Ảnh hưởng khi dùng Unsloth.**
- `use_gradient_checkpointing` nhận `True`, `False` hoặc `"unsloth"`. Docs khuyên dùng `"unsloth"`: giảm thêm 30% bộ nhớ và hỗ trợ fine-tune context rất dài. Comment trong code mẫu ghi: "uses 30% less VRAM, fits 2x larger batch sizes".
- Studio mặc định Gradient Checkpointing = `unsloth`.
- Chỉ nên đặt `False` để train nhanh hơn khi VRAM còn dư nhiều. Nếu không, bạn dễ gặp OOM, nhất là với `max_seq_length` lớn **[Nhận định]**.
- `"unsloth"` dùng RAM hệ thống để chứa activation. Vì vậy máy ít RAM có thể thành điểm nghẽn **[Nhận định]**, cần kiểm tra lại.

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning/hyperparameter) (bảng Hyperparameter khuyến nghị); docs gốc [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [500K Context Length Fine-tuning](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning), [Unsloth Studio](https://unsloth.ai/docs/new/studio/start).

**Nguồn:** https://docs.pytorch.org/docs/stable/checkpoint.html, https://arxiv.org/abs/1604.06174, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/blog/500k-context-length-fine-tuning, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/get-started/install/google-colab

## Merge adapter vào model gốc

Sau khi train, bạn chọn giữ adapter riêng hay gộp nó vào model gốc. Mục này giải thích merge là gì, cách lưu từng kiểu, và khi nào không nên merge.

**Khái niệm.** **Merge** (gộp) là cộng hẳn phần cập nhật `(lora_alpha / r) × B × A` vào trọng số gốc `W`. Kết quả là một model độc lập, không cần adapter riêng. Sau khi merge, inference không còn độ trễ do phải tính thêm nhánh adapter. Trong thư viện PEFT, việc này làm bằng `merge_and_unload()`. **[Nguồn ngoài]** https://huggingface.co/docs/peft/main/en/package_reference/lora, https://arxiv.org/abs/2106.09685

**Ví dụ.** Các cách lưu trong Unsloth (theo docs):

| Cách lưu | Lệnh | Dùng khi |
| --- | --- | --- |
| Chỉ adapter | `model.save_pretrained("finetuned_lora")` + `tokenizer.save_pretrained("finetuned_lora")`, hoặc `save_method = "lora"` | Muốn file nhỏ (~100MB), giữ model gốc riêng |
| Merge 16-bit | `model.save_pretrained_merged("finetuned_model", tokenizer, save_method = "merged_16bit")` | Chạy bằng vLLM; bước đầu tiên trước khi chuyển sang GGUF |
| Merge 4-bit | `save_method = "merged_4bit"`, rồi `merged_4bit_forced` nếu chắc chắn | Docs **không khuyến khích**, trừ khi biết rõ mục đích (ví dụ DPO training, inference engine online của Hugging Face) |
| Đẩy lên Hub | `model.push_to_hub_merged("hf/model", tokenizer, save_method = ..., token = "")` | Chia sẻ lên Hugging Face |

**Ảnh hưởng khi dùng Unsloth.**
- Xuất GGUF (cho Ollama, llama.cpp, LM Studio) thực chất đi qua bản merge 16-bit trước, rồi mới lượng tử hóa sang GGUF.
- Nếu merge hoặc lưu 16-bit bị OOM, docs gợi ý giảm `maximum_memory_usage` (mặc định `0.75`, thử `0.5`).
- Model sau merge phải dùng đúng chat template và EOS token như lúc train. Sai template là nguyên nhân phổ biến nhất khiến model chạy tốt trong Unsloth nhưng kém trên engine khác.

**Khi nào không merge.**
- **Phục vụ nhiều adapter trên một model gốc**: docs Unsloth có hướng dẫn **LoRA hot swapping** trong vLLM. Cách này nạp và gỡ adapter lúc server đang chạy, không cần khởi động lại. Các bước theo ví dụ của docs:
  - Đặt `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.
  - Chạy `vllm serve` với `--enable-lora`, `--max-loras 4`, `--max-lora-rank 64`.
  - Gọi endpoint `/v1/load_lora_adapter` và `/v1/unload_lora_adapter`.
  
  Adapter train với `r` lớn hơn `--max-lora-rank` có thể không nạp được — **[Nhận định]**, cần kiểm tra lại.
- **Còn muốn train tiếp**: load lại adapter đã lưu để continued finetuning (docs Continued Pretraining).
- **Muốn chỉnh độ mạnh fine-tune sau khi train** (giảm alpha, ví dụ nhân 0.5): cần adapter còn tách riêng **[Nhận định]**.
- **Tiết kiệm dung lượng**: giữ nhiều adapter ~100MB thay vì nhiều bản model đầy đủ **[Nhận định]**.
- Thư viện PEFT cũng cho nạp nhiều adapter và chuyển qua lại (`load_adapter`, `set_adapter`), hoặc merge tạm rồi gỡ (`merge_adapter` / `unmerge_adapter`). **[Nguồn ngoài]** https://huggingface.co/docs/peft/developer_guides/lora

**Gặp ở đâu trong Unsloth.** [Export & deploy](/export-deploy) (bảng định dạng export, Deploy: vLLM), [Inference](/inference/); docs gốc [vLLM Deployment & Inference Guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide), [Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf), [LoRA Hot Swapping Guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/lora-hot-swapping-guide), [Inference & Deployment](https://unsloth.ai/docs/basics/inference-and-deployment).

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf, https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/lora-hot-swapping-guide, https://unsloth.ai/docs/basics/inference-and-deployment, https://unsloth.ai/docs/basics/continued-pretraining, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://huggingface.co/docs/peft/main/en/package_reference/lora, https://huggingface.co/docs/peft/developer_guides/lora, https://arxiv.org/abs/2106.09685

## Gặp ở đâu trong Unsloth

| Khái niệm | Trang Unsloth trên website | Docs gốc |
| --- | --- | --- |
| Full fine-tuning vs PEFT | [Fine-tuning](/fine-tuning/) | [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) |
| LoRA (A × B) | [Fine-tuning](/fine-tuning/) | [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [Faster MoE](https://unsloth.ai/docs/basics/faster-moe) |
| `r`, `lora_alpha`, `lora_dropout`, `bias`, `use_rslora`, `loftq_config` | [Fine-tuning](/fine-tuning/), [Cài đặt](/cai-dat) | [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama) |
| `target_modules` | [Fine-tuning](/fine-tuning/) | [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [Continued Pretraining](https://unsloth.ai/docs/basics/continued-pretraining) |
| QLoRA, `load_in_4bit` | [Fine-tuning](/fine-tuning/), [Model catalog](/model-catalog) | [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [FAQ](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me) |
| Gradient checkpointing | [Fine-tuning](/fine-tuning/) | [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [500K Context Length Fine-tuning](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning) |
| Merge adapter, `save_method` | [Export & deploy](/export-deploy) | [vLLM Deployment & Inference Guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide), [Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf) |
| LoRA hot swapping | [Export & deploy](/export-deploy) | [LoRA Hot Swapping Guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/lora-hot-swapping-guide) |
