---
title: Từ RLHF, PPO tới GRPO
description: "Các tên hay gặp khi đọc về RL cho LLM: RLHF, PPO, GRPO, RLVR, và ý nghĩa của chữ Group Relative."
---

# Từ RLHF, PPO tới GRPO

Trang này điểm qua các tên bạn sẽ gặp khi đọc về RL cho LLM. Điểm cần nhớ: GRPO là bản gọn hơn của PPO, bỏ bớt hai model nên tốn ít bộ nhớ hơn.

- **RLHF** (Reinforcement Learning from Human Feedback, RL từ phản hồi của con người): OpenAI phổ biến khái niệm này. Nút thích hoặc không thích trong ChatGPT là một ví dụ về dữ liệu cho RLHF.
- **PPO** (Proximal Policy Optimization): thuật toán dùng để làm RLHF. Theo docs, PPO gồm 3 hệ thống:
  - Generating Policy: model đang train.
  - Reference Policy: model gốc.
  - Value Model: ước lượng reward trung bình.

  Ngoài ra còn một Reward Model để tính reward.
- **GRPO** (Group Relative Policy Optimization): do DeepSeek phát triển để train các model suy luận R1. GRPO khác PPO ở hai điểm:
  1. Bỏ Value Model, thay bằng thống kê thu được khi gọi hàm chấm điểm nhiều lần.
  2. Bỏ Reward Model, thay bằng reward function tự viết (có thể dùng RLVR).

  Vì bớt được hai model, GRPO tiết kiệm bộ nhớ và chạy nhanh hơn.
- **RLVR** (Reinforcement Learning with Verifiable Rewards, RL với phần thưởng kiểm chứng được): chấm điểm dựa trên những tác vụ dễ kiểm tra đúng sai. Ví dụ: phép toán (2+2=4), hoặc code có chạy đúng hay không. Docs nhấn mạnh GRPO không chỉ dùng cho toán và code. Mẹo là thiết kế một **rubric** (danh sách nhiều reward nhỏ kiểm chứng được), thay vì một reward duy nhất bao trùm tất cả.

**"Group Relative" nghĩa là gì:** GRPO so mỗi câu trả lời với cả nhóm câu trả lời cho cùng một câu hỏi. Cách làm như sau:

1. Với mỗi câu hỏi, GRPO sinh nhiều câu trả lời. Docs ví dụ sinh 4 lần cho "What is 2+2?" và được 4, 3, D, C.
2. Chấm reward cho từng câu.
3. Tính trung bình và độ lệch chuẩn, rồi chuẩn hóa Z-score.

Kết quả gọi là **advantage** (lợi thế tương đối của từng câu so với cả nhóm). Advantage được dùng thay cho Value Model.

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
