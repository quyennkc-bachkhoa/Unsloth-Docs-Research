---
title: Lỗi thường gặp
description: "Các lỗi hay gặp khi train RL/GRPO, gom từ nhiều trang docs: train chưa đủ, dữ liệu và model chưa đúng, cấu hình máy."
---

# Lỗi thường gặp khi làm RL

Các lỗi dưới đây gom lại từ những lưu ý rải rác trong docs, xếp theo nhóm: train chưa đủ, dữ liệu và model chưa đúng, và cấu hình máy.

::: warning Cạm bẫy
**Train chưa đủ hoặc reward chưa tốt**

- **Model không học ra suy luận:** thường do train quá ít bước, hoặc reward function và verifier chưa tốt. Docs khuyên thử notebook Advanced GRPO vì notebook này có reward function tốt hơn.
- **Dừng quá sớm:** reward thường chỉ bắt đầu tăng sau khoảng 300 bước, có khi phải 1000 bước hoặc hơn. Docs ghi thời gian khác nhau (30 phút hay tối thiểu 12 giờ); xem hộp "Docs chưa thống nhất" ở trang [GRPO trong Unsloth](/reinforcement-learning/grpo#meo-tu-docs).
- **GRPO chỉ học định dạng:** docs nhắc GRPO có xu hướng mặc định là chỉ học format. Notebook Advanced dùng pre-finetuning để tránh việc này.
- **Reward hacking:** nếu thiếu sandbox hoặc thiếu kiểm tra, model có thể sửa test, đọc biến global hay dùng thư viện có sẵn để "ăn gian".

**Model và dữ liệu chưa đúng**

- **Xác suất đáp án đúng bằng 0:** khi đó RL không bao giờ hoạt động. Hãy bắt đầu từ model đã instruction-finetune. Nếu dùng base model thì phải có chat template.
- **Model quá nhỏ:** docs khuyên dùng từ 1.5B tham số trở lên thì mới sinh thinking token ổn định.
- **Chỉ 1 generation mỗi prompt:** advantage không xác định vì độ lệch chuẩn bằng 0. Cần tối thiểu 2.
- **Đáp án trong dataset chứa lập luận:** cột đáp án chỉ được chứa kết quả, không kèm phần lập luận.

**Cấu hình máy và chi phí**

- **Chưa tính thời gian inference khi dự trù chi phí:** theo docs, trong một lần chạy RL của Unsloth, khoảng 96% thời gian là vLLM inference, còn train chưa tới 4%.
- **`gpu_memory_utilization` = 1.0:** không chạy được. Dùng 0.9–0.95 kèm Standby.
- **Lỗi khi chạy GRPO local:** docs gợi ý `pip install diffusers` và dùng vLLM bản mới nhất.
- **FP8 trên Colab miễn phí:** GPU T4 không hỗ trợ FP8, nên notebook FP8 của docs dùng L4 24GB.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/rl-reward-hacking, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning
