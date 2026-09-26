---
title: Fine-tune bằng Unsloth
description: "Fine-tune mang lại gì cho việc soạn văn bản hành chính, khi nào nên dùng, kết hợp với RAG thế nào, và cách chạy Unsloth trên 2×L40S."
---

# Fine-tune bằng Unsloth

Trang này trả lời hai câu: **fine-tune để làm gì** khi soạn văn bản hành chính, và **làm thế nào** với Unsloth trên máy 2×L40S. Các bước chi tiết đã có ở phần [Fine-tuning](/fine-tuning/). Trang này chỉ nối chúng lại theo đúng dự án này và dẫn link tới từng bước.

::: tip Tóm tắt
- **Dùng khi:** đã chọn được model nền và muốn model viết đúng văn phong, thể thức và nghiệp vụ của đơn vị.
- **Kết quả:** biết phần nào nên giao cho fine-tune, phần nào cho RAG, và có lộ trình 6 bước kèm cấu hình cho 1 hoặc 2 card L40S.
- **Nên biết trước:** [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen), [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora), [Chọn model cho tiếng Việt](/tro-ly-van-ban/chon-model).
:::

## Fine-tune mang lại gì

Fine-tune là **dạy thêm** cho một model có sẵn bằng ví dụ của chính đơn vị. Model đã biết tiếng Việt từ trước. Fine-tune dạy nó **viết như cán bộ văn thư của đơn vị viết**.

Docs Unsloth nêu ba mục tiêu của fine-tuning: cập nhật kiến thức, tùy biến hành vi (giọng điệu, phong cách) và tối ưu cho một tác vụ cụ thể. Áp vào văn bản hành chính:

| Mục tiêu theo docs | Với văn bản hành chính | Ví dụ |
| --- | --- | --- |
| Tùy biến hành vi | Văn phong công vụ: trang trọng, ngắn gọn, không cảm xúc | Viết "Đề nghị các đơn vị khẩn trương triển khai…" thay vì văn nói |
| Tối ưu cho tác vụ | Đúng thể thức, đúng cấu trúc từng loại văn bản | Công văn có đủ số ký hiệu, trích yếu, "Kính gửi", nơi nhận |
| Cập nhật kiến thức | Thuật ngữ, quy trình, tên đơn vị nội bộ | Biết tên phòng ban, mẫu biểu riêng của đơn vị |

**[Nhận định]** Model gốc hỏi gì cũng trả lời được, nhưng thường viết như "trợ lý chat": có lời chào, có gạch đầu dòng markdown, sai thứ tự thành phần thể thức. Fine-tune với vài trăm đến vài nghìn văn bản mẫu tốt sẽ sửa đúng những lỗi này. Chép tay mọi quy tắc vào prompt thì khó giữ ổn định.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners

## Fine-tune và RAG chia việc

Docs Unsloth khuyên **kết hợp** hai cách: fine-tuning cho chuyên môn và định dạng, RAG cho dữ liệu thay đổi nhanh. Văn bản pháp luật là loại dữ liệu thay đổi liên tục: văn bản mới ban hành, văn bản cũ hết hiệu lực.

<div class="dg">
  <div class="dg-title">Mỗi cách lo một phần</div>
  <div class="dg-grid" style="--cols: 2">
    <div class="dg-group">
      <div class="dg-glabel">Fine-tune (nằm trong trọng số)</div>
      <div class="dg-node">Văn phong công vụ</div>
      <div class="dg-node">Thể thức, bố cục từng loại văn bản</div>
      <div class="dg-node">Thuật ngữ, cách xưng hô nội bộ</div>
    </div>
    <div class="dg-group">
      <div class="dg-glabel">RAG (tra cứu lúc hỏi)</div>
      <div class="dg-node">Nội dung nghị định, thông tư đang hiệu lực</div>
      <div class="dg-node">Văn bản, hồ sơ của đơn vị</div>
      <div class="dg-node">Số liệu, tên người, ngày tháng</div>
    </div>
  </div>
  <div class="dg-cap">Hỏi đáp quy định thì RAG là chính. Soạn thảo thì fine-tune là chính, RAG cấp căn cứ pháp lý.</div>
</div>

**[Nhận định]** Đừng fine-tune để model "thuộc" nội dung luật. Khi luật đổi, model vẫn trả lời theo bản cũ, và không ai biết nó lấy câu đó từ đâu. Hãy để RAG đưa đúng điều khoản vào prompt, kèm trích dẫn nguồn.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

## Lộ trình 6 bước

Mỗi bước dẫn tới trang chi tiết đã có trên website.

| Bước | Làm gì | Đọc |
| --- | --- | --- |
| 1. Chọn model | Qwen3.6/3.8-27B hoặc Gemma 4 31B | [Chọn model cho tiếng Việt](/tro-ly-van-ban/chon-model), [Chọn cách train và model](/fine-tuning/chon-cach-train) |
| 2. Chuẩn bị dữ liệu | Gom văn bản mẫu, đổi thành cặp "yêu cầu → văn bản" dạng hội thoại | [Tìm và chuẩn bị dữ liệu](/tro-ly-van-ban/van-ban-hanh-chinh), [Định dạng dữ liệu](/du-lieu/dinh-dang), [Chat template](/du-lieu/chat-template) |
| 3. Cấu hình | LoRA rank, learning rate, số epoch | [Chọn hyperparameter](/fine-tuning/hyperparameter) |
| 4. Train | Studio (không code) hoặc Core (script) | [Quy trình từng bước](/fine-tuning/quy-trinh) |
| 5. Đánh giá | Eval loss, và **quan trọng hơn** là cán bộ nghiệp vụ chấm bài | [Đánh giá và overfitting](/fine-tuning/danh-gia) |
| 6. Export và host | merged 16-bit cho vLLM, GGUF cho Ollama | [Xuất file GGUF](/export-deploy/gguf), [Host model trên máy chủ](/tro-ly-van-ban/hosting) |

Với dữ liệu hội thoại, nên **chỉ tính loss trên câu trả lời** (phần văn bản model viết ra), không tính trên yêu cầu. Xem [Chỉ train trên câu trả lời](/du-lieu/chat-template#chi-train-tren-cau-tra-loi).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/new/studio/start

## Chạy trên 1 hay 2 card L40S

Nếu model vừa một card, hãy **train trên một card**. Chia qua hai card chỉ để chứa model lớn hơn, không làm train nhanh hơn.

Unsloth có hai cách dùng nhiều GPU:

| Cách | Làm gì | Được gì | Lệnh |
| --- | --- | --- | --- |
| `device_map = "balanced"` | Chia các layer của model ra 2 card | Chứa được model tới khoảng 96 GB. Hai card chạy lần lượt nên không nhanh hơn | Thêm vào `from_pretrained` |
| DDP | Mỗi card giữ một bản model, chia dữ liệu | Train nhanh hơn. Model phải vừa **một** card | `accelerate launch train.py` hoặc `torchrun --nproc_per_node 2 train.py` |

Docs ghi hỗ trợ nhiều GPU hiện vẫn phải thiết lập thủ công, bản chính thức "sẽ sớm có". Studio ghi "Multi-GPU works automatically, with a new version coming" nhưng không nói cơ chế.

**[Ước tính]** Theo bảng VRAM tối thiểu và trang từng model trong docs Unsloth:

| Model | Cách train | VRAM theo docs | Cấu hình trên 2×L40S |
| --- | --- | --- | --- |
| Gemma 4 31B | QLoRA | 22 GB | 1 card. Card còn lại để serve hoặc chạy DDP |
| Qwen3.8-27B | QLoRA | 24 GB | 1 card |
| Qwen3.8-27B | LoRA 16-bit | >36 GB | 1 card (sát khi context dài) hoặc `balanced` 2 card |
| Qwen3.5/3.6-27B | LoRA 16-bit | 56 GB | `balanced` 2 card |
| Dense 32B | LoRA 16-bit | 76 GB | `balanced` 2 card, còn ít chỗ |
| Dense 70B | QLoRA | 41 GB | 1 card, sát trần |

Đây là mức tối thiểu với batch nhỏ và context ngắn. Văn bản hành chính dài 1–3 trang (khoảng 1.500–4.000 token), nên cần thêm bộ nhớ.

**[Nhận định]** Ví dụ ghép từ các đoạn code trong docs: LoRA 16-bit Qwen trên 2 card. Tên model và `max_seq_length` là giả định, hãy đối chiếu trang model trước khi chạy.

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3.8-27B",
    max_seq_length = 4096,       # đủ cho công văn 2–3 trang
    load_in_16bit = True,        # LoRA 16-bit, docs Qwen3.5 khuyên không QLoRA
    device_map = "balanced",     # chia model ra 2 card L40S
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    use_gradient_checkpointing = "unsloth",
)
```

**Nguồn:** https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth, https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp, https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements, https://unsloth.ai/docs/models/qwen3.5/fine-tune, https://unsloth.ai/docs/models/qwen3.8/train, https://unsloth.ai/docs/models/gemma-4/train

## Khi nào không cần fine-tune

Fine-tune tốn công chuẩn bị dữ liệu và đánh giá. Nên thử cách rẻ hơn trước.

**[Nhận định]**

1. **Thử prompt trước.** Viết system prompt mô tả thể thức, kèm 1–2 văn bản mẫu. Nếu model đã viết đạt khoảng 80% yêu cầu, chỉ cần RAG và prompt tốt.
2. **Fine-tune khi** prompt không giữ được thể thức ổn định, prompt quá dài vì chứa nhiều mẫu, hoặc cần model nhỏ hơn chạy nhanh hơn.
3. **Không fine-tune để** model nhớ nội dung văn bản pháp luật. Việc đó là của RAG, xem [Ứng dụng RAG](/ung-dung-rag).

::: warning Lỗi thường gặp
- Train quá nhiều epoch trên ít dữ liệu, model chép lại nguyên văn bản mẫu. Xem [Đánh giá và overfitting](/fine-tuning/danh-gia).
- Dùng sai chat template lúc chạy so với lúc train, model trả lời lộn xộn. Docs Unsloth nhấn mạnh đây là lỗi hay gặp nhất khi đem model sang Ollama.
- Để lọt dữ liệu cá nhân hoặc tài liệu mật vào dataset. Xem [Tìm và chuẩn bị dữ liệu](/tro-ly-van-ban/van-ban-hanh-chinh).
:::

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me

## Đọc tiếp

- [Tìm và chuẩn bị dữ liệu](/tro-ly-van-ban/van-ban-hanh-chinh) — trang kế tiếp: tìm dataset trên Hugging Face.
- [Quy trình từng bước](/fine-tuning/quy-trinh) — chi tiết từng bước trong Studio và Core.
- [Kỹ thuật nâng cao](/fine-tuning/mo-rong) — multi-GPU, continued pretraining.
- [Host model trên máy chủ](/tro-ly-van-ban/hosting) — đem model đã train đi phục vụ người dùng.
