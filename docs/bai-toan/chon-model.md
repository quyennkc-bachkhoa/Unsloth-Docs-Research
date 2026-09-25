---
title: Chọn model cho tiếng Việt
description: "Model open-weight nào mạnh tiếng Việt nhất (09/2026), cái nào vừa máy 2×L40S để chạy và fine-tune bằng Unsloth, và vì sao không nên tin một bảng điểm duy nhất."
---

# Chọn model cho tiếng Việt

Trang này trả lời câu hỏi "hiện tại model nào tốt nhất, phù hợp nhất cho tiếng Việt" trong điều kiện máy 2×L40S (96 GB). Đọc xong bạn có danh sách 2–3 ứng viên, lý do chọn, số bộ nhớ để chạy và fine-tune theo docs Unsloth.

::: tip Tóm tắt
- **Dùng khi:** cần chốt model nền (base model) trước khi fine-tune hoặc triển khai cho người dùng tiếng Việt.
- **Kết quả:** ứng viên chính là **Qwen3.6-27B / Qwen3.8-27B** và **Gemma 4 31B**. Cả hai đều Apache 2.0, nằm nhóm đầu bảng tiếng Việt và vừa 96 GB để fine-tune.
- **Nên biết trước:** [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) (base và instruct), [Dense & MoE](/kien-thuc-nen/dense-va-moe) (ký hiệu "122B-A10B"), [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho).
:::

::: warning Dữ liệu có hạn dùng
Số liệu tra ngày **2026-09-25**. Bảng xếp hạng thay đổi mỗi khi có model mới. Trước khi chốt, hãy mở lại các bảng xếp hạng dưới đây.
:::

## Đáp án ngắn

**[Nhận định]** Chọn một trong hai, tốt nhất là thử cả hai trên dữ liệu thật của đơn vị:

| Hạng | Model | Vì sao | Fine-tune trên 2×L40S |
| --- | --- | --- | --- |
| 1 | **Qwen3.6-27B** hoặc bản mới hơn **Qwen3.8-27B** | Điểm tiếng Việt 76.12 (Qwen3.6), ngang nhóm đầu. Tokenizer ít tốn token nhất với văn bản tiếng Việt. Nhiều model tiếng Việt mạnh nhất trên VMLU cũng dựng trên Qwen | LoRA 16-bit chia 2 card (56 GB theo docs Qwen3.5), hoặc 1 card theo docs Qwen3.8 |
| 2 | **Gemma 4 31B** | Điểm tiếng Việt cao nhất bảng SEA-HELM (77.09) | QLoRA 22 GB, vừa thoải mái 1 card |
| Đối chứng | Qwen-SEA-LION-v4.5-27B-IT | Tinh chỉnh riêng cho Đông Nam Á | Như Qwen3.6 (cùng nền) |

- **Qwen3.8-27B** mới ra 08/2026. Model này làm tốt hơn Qwen3.6 rõ rệt ở bài kiểm tra làm theo chỉ dẫn (IFBench 79.5 so với 69.1), điều quan trọng khi soạn văn bản theo khuôn mẫu. Nhưng nó **chưa có điểm tiếng Việt công khai**.
- **Qwen3.5-122B-A10B** (điểm tiếng Việt 76.39) quá lớn để fine-tune trên 96 GB. Có thể chạy bản 4-bit (khoảng 70 GB) để làm "model giáo viên": sinh dữ liệu mẫu và chấm điểm model nhỏ.

**Nguồn:** https://leaderboard.sea-lion.ai/detailed/VI, https://huggingface.co/Qwen/Qwen3.8-27B, https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/qwen3.5/fine-tune, https://unsloth.ai/docs/models/qwen3.8/train, https://unsloth.ai/docs/models/gemma-4/train

## Bảng điểm tiếng Việt hiện có

Hiện chỉ **SEA-HELM** (của AI Singapore) có điểm tiếng Việt cho các model đời 2026. VMLU, bảng quen thuộc ở Việt Nam, không còn dùng để so model tải về được.

### SEA-HELM, điểm tiếng Việt

**[Nguồn ngoài]** Bảng VI cập nhật ngày 18/09/2026, trích các model open-weight:

| Model | Cỡ | Điểm VI | Khoảng tin cậy 95% |
| --- | --- | --- | --- |
| Gemma 4 | 31B | **77.09** | −1.73 / +1.64 |
| Qwen 3.5 | 122B MoE | **76.39** | −1.59 / +1.56 |
| Qwen 3.6 | 27B | **76.12** | −1.46 / +1.39 |
| SEA-LION v4.5 (nền Qwen) | 27B | 73.85 | −1.40 / +1.40 |
| Muse Glimmer (Meta) | 30B | 73.52 | −1.47 / +1.38 |
| SEA-LION v4.8 (nền Nemotron) | 120B MoE | 70.36 | |
| gpt-oss | 120B MoE | 69.81 | |
| Mistral Medium 3.5 | 128B | 68.82 | |
| Nemotron 3.5 Lightning | 30B MoE | 60.97 | |

- **[Nhận định]** Ba model đầu chênh nhau chưa tới 1 điểm, nằm trong khoảng tin cậy, nên thực chất là **ngang nhau**. Đừng chọn chỉ vì đứng thứ nhất.
- SEA-LION v4.5 tinh chỉnh cho Đông Nam Á nhưng lại **thấp hơn** chính model nền Qwen 3.6 của nó.
- SEA-HELM đo năng lực chung: NLP cổ điển, chat, làm theo chỉ dẫn, ngôn ngữ và văn hóa, an toàn. Nó không đo văn bản hành chính.

### VMLU

**[Nguồn ngoài]** VMLU (Zalo AI và JAIST) gồm 10.880 câu trắc nghiệm, 58 môn.

- Bảng cho model gốc dừng cập nhật từ **13/03/2025**. Khi đó đứng đầu là QwQ-32B với 76.13. Bảng không có Qwen3.5 trở lên, Gemma 4 hay gpt-oss.
- Bảng cho model đã fine-tune (cập nhật tới 08/2026) có điểm 85–90. Nhưng **toàn bộ nhóm đầu đều là model riêng, không công bố trọng số**, ví dụ VAI-LLM, V-LLM (VinSmart Future), Vi-Qwen3.6 (Zoom AI Lab), MISA-AI, VNPT AI. Đơn vị không tải về dùng được.
- **[Nhận định]** Nhiều model trong nhóm này ghi nền là Qwen3 hoặc Qwen3.6. Đây là bằng chứng gián tiếp rằng fine-tune Qwen cho tiếng Việt cho kết quả tốt.

### Benchmark pháp luật

**[Nguồn ngoài]**
- **VLegal-Bench** (CMC, arXiv 2512.14554) có 10.450 mẫu, 22 tác vụ pháp luật tiếng Việt, nhưng chỉ đo các model đời 2024–2025. Model tụt mạnh ở tác vụ khó: 16/23 model đạt 0 điểm ở phát hiện mâu thuẫn giữa quy định.
- **VLSP 2025 LegalSLM** giới hạn model tối đa 4B tham số.

**Không tìm được benchmark nào đo soạn thảo hoặc tóm tắt văn bản hành chính nhà nước.** Đơn vị cần tự làm bộ đánh giá, xem [Dữ liệu văn bản hành chính](/bai-toan/van-ban-hanh-chinh).

**Nguồn:** https://leaderboard.sea-lion.ai/, https://leaderboard.sea-lion.ai/detailed/VI, https://vmlu.ai/leaderboard, https://www.nrl.ai/en/bench, https://arxiv.org/html/2512.14554v5, https://vlsp.org.vn/vlsp2025/eval/legalSLM

## Model "chuyên tiếng Việt" thì sao

Các model do nhóm Việt Nam hoặc Đông Nam Á train riêng hầu hết **đã cũ hoặc không tải về được**. Model đa ngôn ngữ đời 2026 đang mạnh hơn.

**[Nguồn ngoài]**

| Model | Cỡ | Nền | License | Tình trạng |
| --- | --- | --- | --- | --- |
| Vistral-7B-Chat | 7B | Mistral-7B (2023) | AFL-3.0, gated | VMLU 50.07, cũ |
| PhoGPT-4B-Chat (VinAI) | 3.7B | Train từ đầu | BSD-3-Clause | 11/2023, cũ |
| SeaLLMs-v3-7B, Sailor2-20B, Arcee-VyLinh | 3–20B | Qwen2 / Qwen2.5 | Riêng / Apache 2.0 | 2024 – đầu 2025, cũ |
| CMC-AI-Legal-32B | 32B | Qwen2 | **cc-by-nc-nd-4.0** | Mạnh về pháp luật, nhưng **cấm dùng thương mại và cấm tạo bản phái sinh**, nên không fine-tune tiếp được |
| VT-Super-120B-A12B (Viettel AI) | 120B MoE | Nemotron 3 Super | Không công bố trọng số | Có train trên dữ liệu hành chính, nhưng không tải được |
| Qwen-SEA-LION-v4.5-27B-IT | 27B | Qwen3.6-27B | MIT | Mới (05/2026), dùng làm đối chứng |

Không tìm thấy model open-weight mới nào trong 2025–2026 của FPT AI, VinAI, Zalo hoặc VNPT.

**[Nhận định]** Với dự án mới, đừng bắt đầu từ Vistral, PhoGPT hay SeaLLMs. Hãy lấy model đa ngôn ngữ mới rồi fine-tune bằng dữ liệu của đơn vị.

**Nguồn:** https://huggingface.co/Viet-Mistral/Vistral-7B-Chat, https://huggingface.co/vinai/PhoGPT-4B-Chat, https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat, https://huggingface.co/sail/Sailor2-20B-Chat, https://huggingface.co/arcee-ai/Arcee-VyLinh, https://huggingface.co/CMC-OPENAI/CMC-AI-Legal-32B, https://vnexpress.net/mo-hinh-ngon-ngu-lon-tieng-viet-voi-120-ty-tham-so-5082057.html, https://huggingface.co/aisingapore/Qwen-SEA-LION-v4.5-27B-IT

## Có vừa máy 2×L40S không

Cả hai ứng viên chính đều chạy được trên một card và fine-tune được trên máy này. Số liệu dưới đây lấy từ docs Unsloth.

**Chạy model** (bản GGUF, tổng RAM + VRAM cần):

| Model | 4-bit | 8-bit | BF16 |
| --- | --- | --- | --- |
| Qwen3.6-27B | 18 GB | 30 GB | 55 GB |
| Qwen3.8-27B | 16–19 GB | 31 GB | 56 GB |
| Gemma 4 31B | 17–20 GB | 34–38 GB | 62 GB |
| Muse Glimmer 30B | 17 GB+ | 34 GB+ | 58 GB+ |
| Qwen3.5-122B-A10B | 70 GB | 132 GB | 245 GB |

**Fine-tune:**

| Model | QLoRA 4-bit | LoRA 16-bit | Trên 2×L40S |
| --- | --- | --- | --- |
| Qwen3.5-27B | Docs khuyên **không** dùng | 56 GB | LoRA 16-bit, chia 2 card |
| Qwen3.8-27B | 24 GB | >36 GB | QLoRA 1 card, LoRA 16-bit 1 card (sát) hoặc 2 card |
| Gemma 4 31B | 22 GB | Docs không ghi | QLoRA 1 card |
| Muse Glimmer 30B | 24 GB | >40 GB | QLoRA 1 card |
| Qwen3.5-122B-A10B | Không khuyên dùng | 256 GB | **Không vừa** |

- Qwen3.6 không có số riêng, trang docs Qwen3.6 trỏ về hướng dẫn Qwen3.5.
- Chia model qua 2 card khi fine-tune: xem [Fine-tune bằng Unsloth](/bai-toan/fine-tune).

::: warning Docs chưa thống nhất: có nên QLoRA với Qwen
| Trang | Ghi |
| --- | --- |
| [Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune) | "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense" |
| [Qwen3.8 train](https://unsloth.ai/docs/models/qwen3.8/train) | QLoRA chạy với 24 GB, khuyên dùng bản `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` |
:::

**[Nhận định]** Máy có 96 GB, nên với Qwen cứ chọn LoRA 16-bit cho an toàn. Không cần tiết kiệm bộ nhớ bằng QLoRA.

**Nguồn:** https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/qwen3.6, https://unsloth.ai/docs/models/qwen3.8, https://unsloth.ai/docs/models/gemma-4, https://unsloth.ai/docs/models/muse-glimmer, https://unsloth.ai/docs/models/qwen3.5/fine-tune, https://unsloth.ai/docs/models/qwen3.8/train, https://unsloth.ai/docs/models/gemma-4/train, https://unsloth.ai/docs/models/muse-glimmer/train

## Tokenizer và license

Hai yếu tố này ít được nhắc nhưng ảnh hưởng tới chi phí và quyền sử dụng.

### Model tốn bao nhiêu token cho tiếng Việt

**[Ước tính]** Đo trên **một** văn bản mẫu dạng công văn (341 âm tiết), bằng tokenizer chính thức của từng model:

| Tokenizer | Token / âm tiết |
| --- | --- |
| Qwen3.6 / Qwen3.8 / SEA-LION v4.5 | 1.40 |
| gpt-oss | 1.49 |
| Gemma 4 | 1.50 |
| Nemotron-SEA-LION v4.8 | 1.57 |

**[Nhận định]** Tokenizer đời mới nào cũng xử lý tiếng Việt khá tốt. Qwen tiết kiệm hơn Gemma khoảng 6–7% token. Chênh lệch này không đủ để quyết định chọn model. Con số chỉ mang tính minh họa vì chỉ đo trên một mẫu.

### License

**[Nguồn ngoài]**
- Qwen3.x, Gemma 4, gpt-oss, Muse Glimmer: Apache 2.0.
- SEA-LION v4.5: MIT.
- CMC-AI-Legal-32B: cc-by-nc-nd-4.0 (phi thương mại, không phái sinh).

**[Nhận định]** Apache 2.0 và MIT cho phép fine-tune và dùng nội bộ. Với cơ quan nhà nước, còn cần xét quy định về nguồn gốc model và an ninh dữ liệu. Phần này nằm ngoài phạm vi nghiên cứu, xem thêm [Dữ liệu văn bản hành chính](/bai-toan/van-ban-hanh-chinh).

**Nguồn:** https://huggingface.co/Qwen/Qwen3.6-27B, https://huggingface.co/Qwen/Qwen3.8-27B, https://ai.google.dev/gemma/docs/core/model_card_4, https://huggingface.co/meta-models/Muse-Glimmer-30B, https://huggingface.co/aisingapore/Qwen-SEA-LION-v4.5-27B-IT, https://huggingface.co/CMC-OPENAI/CMC-AI-Legal-32B

## Đọc tiếp

- [Fine-tune bằng Unsloth](/bai-toan/fine-tune) — trang kế tiếp: dùng model đã chọn để train với dữ liệu của đơn vị.
- [Danh sách model hỗ trợ](/model-catalog) — bảng đầy đủ các họ model Unsloth cung cấp.
- [Host model trên máy chủ](/bai-toan/hosting) — model 27–35B phục vụ được bao nhiêu người trên 96 GB.
