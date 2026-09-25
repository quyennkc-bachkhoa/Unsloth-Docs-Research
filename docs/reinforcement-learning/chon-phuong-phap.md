---
title: SFT vs DPO vs GRPO
description: "So sánh SFT, DPO và GRPO theo mục tiêu tối ưu, dữ liệu cần có, lượng dữ liệu và chi phí."
---

# Khi nào dùng gì: SFT vs DPO vs GRPO

Cách chọn phụ thuộc vào dữ liệu bạn có: đáp án mẫu, cặp so sánh tốt/xấu, hay một cách chấm điểm tự động. Bảng dưới so sánh ba hướng.

| Tiêu chí | SFT | DPO (và ORPO/KTO) | GRPO |
| --- | --- | --- | --- |
| Tối ưu cái gì (theo docs) | Xác suất dự đoán từ tiếp theo | Căn chỉnh theo preference (sở thích) | Tối đa hóa reward từ reward function |
| Dữ liệu cần | Cặp đầu vào → đầu ra mẫu | Dữ liệu preference (định dạng: cần kiểm tra lại) | Câu hỏi + đáp án (không kèm lập luận) + reward function/verifier |
| Lượng dữ liệu theo docs | (xem trang [Fine-tuning](/fine-tuning/)) | Docs không nêu | Tối ưu từ 500 dòng; thử được với 10 dòng |
| Hợp với | [Nhận định] Dạy model format, phong cách, kiến thức miền khi có sẵn đáp án mẫu | [Nhận định] Khi có sẵn các cặp so sánh "câu này tốt hơn câu kia" | Tác vụ kiểm chứng được (toán, code); suy luận; email, truy vấn DB, luật, y khoa nếu có rubric tốt |
| Chi phí | [Nhận định] Rẻ nhất | [Nhận định] Trung bình (ví dụ trong docs xuất phát từ model đã SFT) | Cao: sinh nhiều câu trả lời mỗi prompt, cần tối thiểu ~300 bước |

[Nhận định] Một thứ tự thường gặp là SFT trước, rồi mới DPO hoặc GRPO. Hai chi tiết trong docs đều khớp với thứ tự này:

- Code DPO xuất phát từ model `zephyr-sft`.
- Docs khuyên làm RL trên model đã instruction-finetune để xác suất ra đáp án đúng lớn hơn 0.

Docs cũng nhắc notebook Advanced GRPO dùng "pre-finetuning" để tránh việc GRPO chỉ học định dạng.

[Nhận định] Nếu bạn không viết được reward function hay verifier đáng tin cho tác vụ, GRPO dễ gặp [reward hacking](/reinforcement-learning/reward-hacking). Khi đó SFT hoặc DPO an toàn hơn.

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo
