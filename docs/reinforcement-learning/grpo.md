---
title: GRPO trong Unsloth
description: "Vòng lặp train GRPO, dữ liệu cần có, yêu cầu VRAM theo kích thước model và các mẹo từ docs."
---

# GRPO trong Unsloth

GRPO là phương pháp RL chính mà Unsloth hướng dẫn. Trang này đi qua vòng lặp train, dữ liệu cần có, yêu cầu VRAM và các mẹo thực tế. Cách viết reward nằm ở trang [Reward function và verifier](/reinforcement-learning/reward-function).

## Vòng lặp GRPO

<div class="dg">
<div class="dg-flow">
<div class="dg-node">Prompt<small>câu hỏi trong dataset</small></div>
<div class="dg-node">Model sinh nhiều câu trả lời<small>vd 8 biến thể</small></div>
<div class="dg-node is-main">Reward function / verifier<small>chấm điểm từng câu</small></div>
<div class="dg-node">Tính advantage<small>(reward − trung bình) / độ lệch chuẩn</small></div>
<div class="dg-node is-end">Cập nhật trọng số policy</div>
<div class="dg-back" style="grid-column: 1 / 6"><span>Sang prompt tiếp theo, lặp lại</span></div>
</div>
</div>

Theo docs, một bước train diễn ra như sau:

1. Với mỗi cặp câu hỏi–đáp án, model sinh nhiều câu trả lời (ví dụ 8; có thể tăng lên 16).
2. Mỗi câu trả lời được chấm bằng các reward function.
3. Số bước train bằng số dòng dữ liệu: 300 dòng là 300 bước (900 bước nếu train 3 epoch).
4. Model cập nhật trọng số sau mỗi bước.

Bạn cần ít nhất **2 generation mỗi prompt**. Lý do: với 1 mẫu, độ lệch chuẩn bằng 0, nên công thức advantage (reward - mean)/std không xác định.

## Dữ liệu cho GRPO

Dataset cần ít nhất 2 cột: câu hỏi và đáp án. Đáp án **không được** chứa lập luận dẫn tới nó, vì model phải tự sinh phần lập luận. Tutorial dùng dataset [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (toán tiểu học). Xem thêm [Dữ liệu](/du-lieu/).

## Yêu cầu VRAM và kích thước model

::: tip Kiến thức nền
Chưa rõ "8B tham số" hay cách ước tính VRAM (bộ nhớ card đồ họa)? Xem [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho). QLoRA/LoRA: xem [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora).
:::

Bảng dưới gom các con số VRAM mà docs đưa ra. Mỗi con số đi kèm điều kiện riêng, nên hãy đọc cả cột điều kiện.

| Nội dung | Con số theo docs | Điều kiện |
| --- | --- | --- |
| Tối thiểu | 5GB VRAM | Model từ 1.5B tham số trở xuống |
| Model tới 17B | 15GB VRAM | Ví dụ Llama 3.1 (8B), Phi-4 (14B), Mistral (7B), Qwen2.5 (7B) |
| Colab miễn phí | GPU 16GB, train model tới 16B | Theo tutorial GRPO |
| Quy tắc chung | Số tỷ tham số ≈ số GB VRAM cần | QLoRA 4-bit; context càng dài càng tốn VRAM |
| LoRA 16-bit | Ít nhất gấp 4 lần VRAM so với QLoRA 4-bit | |
| Kích thước model khuyến nghị | Tối thiểu 1.5B tham số | Để model sinh thinking token đúng; model nhỏ hơn có thể không làm được |
| FP8 GRPO | Qwen3-1.7B chạy với 5GB VRAM | Cần GPU hỗ trợ FP8 (RTX 40, 50, H100...); T4 không hỗ trợ FP8 |

::: warning Docs chưa thống nhất
Các trang docs đưa ra con số VRAM và kích thước model không khớp nhau. Bảng dưới đặt các con số cạnh nhau để bạn tự đối chiếu:

| Thông số | Nguồn A | Nguồn B |
| --- | --- | --- |
| VRAM ~15–16GB train được model tối đa bao nhiêu | 15GB VRAM: model tới 17B ([RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)) | GPU 16GB (Colab miễn phí): model tới 16B ([Tutorial GRPO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo)) |
| Mức 5GB VRAM | 5GB đủ cho model từ 1.5B tham số trở xuống ([RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)) | Qwen3-1.7B FP8 GRPO chạy với 5GB VRAM ([FP8 RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning)) |
| Kích thước model tối thiểu | Mức VRAM tối thiểu 5GB được nêu cho model từ 1.5B trở xuống ([RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), mục "What Unsloth offers") | Khuyên dùng model tối thiểu 1.5B để sinh thinking token đúng ([RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), mục "Basics/Tips") |
:::

Trước đây GRPO chỉ hỗ trợ full fine-tuning. Unsloth đã làm cho nó chạy được với QLoRA và LoRA.

## Mẹo từ docs {#meo-tu-docs}

- **Kiên nhẫn với số bước.** Chờ ít nhất **300 bước** thì reward mới bắt đầu tăng; có khi cần 1000 bước hoặc hơn. Để có kết quả tốt có thể mất tối thiểu 12 giờ, nhưng bạn dừng lúc nào cũng được.
- **Đủ dữ liệu.** Nên có ít nhất **500 dòng dữ liệu**. Thử với 10 dòng vẫn được, nhưng càng nhiều càng tốt.
- **Base model cần chat template.** Nếu dùng base model thì phải có chat template.
- **Model chưa có trong vLLM.** Model không được vLLM hỗ trợ (vd Qwen3.5) vẫn chạy RL được bằng cách đặt `fast_inference=False`.
- **Biến thể của GRPO.** GRPOConfig hỗ trợ GSPO, Dr. GRPO, DAPO... qua tham số `loss_type`. Chi tiết xem hướng dẫn nâng cao trên docs.

::: warning Docs chưa thống nhất
Hai trang docs ghi khác nhau về thời gian train và về danh sách giá trị `loss_type`:

| Thông số | Nguồn A: [RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) | Nguồn B: [Tutorial GRPO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo) |
| --- | --- | --- |
| Số bước / thời gian | Ít nhất 300 bước để reward tăng; để có kết quả tốt có thể cần tối thiểu 12 giờ; có khi 1000 bước hoặc hơn | Ít nhất 300 bước, có thể mất 30 phút; train lâu hơn để có kết quả tối ưu |
| Giá trị `loss_type` được liệt kê | `'gspo'`, `'grpo'`, `'dr_grpo'` | `'bnpo'`, `'grpo'`, `'dr_grpo'`, `'dapo'` |
:::

Tutorial từng bước: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning
