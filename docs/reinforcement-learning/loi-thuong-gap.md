---
title: Lỗi thường gặp
description: "Các lỗi hay gặp khi train RL/GRPO, gom từ nhiều trang docs: train chưa đủ, dữ liệu và model chưa đúng, cấu hình máy."
---

# Lỗi thường gặp

Trang này gom các lỗi hay gặp khi train RL (GRPO) từ những lưu ý rải rác trong docs, xếp theo ba nhóm: train chưa đủ, dữ liệu và model chưa đúng, và cấu hình máy.

::: tip Tóm tắt
- **Dùng khi:** reward không tăng, model không học ra suy luận, hoặc train RL bị lỗi và bạn cần tra nhanh nguyên nhân.
- **Kết quả:** nhận ra lỗi của mình thuộc nhóm nào và biết cách xử lý mà docs gợi ý.
- **Nên biết trước:** [Train bằng GRPO](/reinforcement-learning/grpo), [Tiết kiệm VRAM khi chạy RL](/reinforcement-learning/memory-efficient).
:::

## Train chưa đủ hoặc reward chưa tốt

Nhóm lỗi hay gặp nhất là dừng train quá sớm hoặc reward function chưa đủ tốt.

::: warning Lỗi thường gặp
- **Model không học ra suy luận:** thường do train quá ít bước, hoặc reward function và verifier chưa tốt. Docs khuyên thử notebook Advanced GRPO vì notebook này có reward function tốt hơn.
- **Dừng quá sớm:** reward thường chỉ bắt đầu tăng sau khoảng 300 bước, có khi phải 1000 bước hoặc hơn. Docs ghi thời gian khác nhau (30 phút hay tối thiểu 12 giờ); xem hộp "Docs chưa thống nhất" ở trang [Train bằng GRPO](/reinforcement-learning/grpo#meo-tu-docs).
- **GRPO chỉ học định dạng:** docs nhắc GRPO có xu hướng mặc định là chỉ học format. Notebook Advanced dùng pre-finetuning để tránh việc này.
- **[Reward hacking](/reinforcement-learning/reward-function#reward-hacking):** nếu thiếu sandbox hoặc thiếu kiểm tra, model có thể sửa test, đọc biến global hay dùng thư viện có sẵn để "ăn gian".
:::

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/rl-reward-hacking

## Model và dữ liệu chưa đúng

RL chỉ chạy khi model xuất phát và dataset đáp ứng vài điều kiện tối thiểu.

::: warning Lỗi thường gặp
- **Xác suất đáp án đúng bằng 0:** khi đó RL không bao giờ hoạt động. Hãy bắt đầu từ model đã instruction-finetune. Nếu dùng base model thì phải có chat template.
- **Model quá nhỏ:** docs khuyên dùng từ 1.5B tham số trở lên thì mới sinh thinking token ổn định.
- **Chỉ 1 generation mỗi prompt:** advantage không xác định vì độ lệch chuẩn bằng 0. Cần tối thiểu 2.
- **Đáp án trong dataset chứa lập luận:** cột đáp án chỉ được chứa kết quả, không kèm phần lập luận.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo

## Cấu hình máy và chi phí

Phần lớn thời gian khi làm RL nằm ở khâu inference, nên khi dự trù chi phí và chọn cấu hình máy cần tính cả phần này.

::: warning Lỗi thường gặp
- **Chưa tính thời gian inference khi dự trù chi phí:** theo docs, trong một lần chạy RL của Unsloth, khoảng 96% thời gian là vLLM inference, còn train chưa tới 4%.
- **`gpu_memory_utilization` = 1.0:** không chạy được. Dùng 0.9–0.95 kèm Standby.
- **Lỗi khi chạy GRPO local:** docs gợi ý `pip install diffusers` và dùng vLLM bản mới nhất.
- **FP8 trên Colab miễn phí:** GPU T4 không hỗ trợ FP8, nên notebook FP8 của docs dùng L4 24GB.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning

## Đọc tiếp

- [Dữ liệu](/du-lieu/) — phần kế tiếp: chuẩn bị dataset đúng định dạng cho fine-tune và RL.
- [Train bằng GRPO](/reinforcement-learning/grpo#meo-tu-docs) — các mẹo về số bước và số dòng dữ liệu giúp tránh phần lớn lỗi ở trên.
- [Viết reward function](/reinforcement-learning/reward-function#reward-hacking) — chi tiết từng kiểu reward hacking và cách chống.
