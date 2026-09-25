---
title: Train AI agent
description: "Train agent nhiều lượt bằng GRPO với ART của OpenPipe: trajectory, frontend/backend và reward RULER."
---

# Train AI agent

Agent là LLM tự làm nhiều bước liên tiếp để đạt một mục tiêu. RL, cụ thể là GRPO, giúp agent làm các chuỗi bước đó ổn định hơn; trang này giới thiệu ART của OpenPipe, công cụ docs dùng để train agent theo cách này.

::: tip Tóm tắt
- **Dùng khi:** model của bạn phải làm nhiều bước liên tiếp hoặc gọi tool, và bạn muốn nó làm việc đó ổn định hơn.
- **Kết quả:** hiểu agent multi-turn là gì, ART bổ sung gì cho GRPOTrainer của Unsloth (trajectory, frontend/backend, reward RULER) và khi nào nên chọn ART.
- **Nên biết trước:** [Train bằng GRPO](/reinforcement-learning/grpo), [Viết reward function](/reinforcement-learning/reward-function).
:::

## Agent là gì và vì sao cần RL

Trong docs, **agent** là một LLM được giao một mục tiêu tổng quát kèm bộ công cụ (tool). Agent thường **multi-turn** (nhiều lượt): thực hiện action, xem kết quả trên môi trường, rồi làm tiếp cho tới khi đạt mục tiêu hoặc thất bại. Docs cho biết ngay cả LLM mạnh cũng khó làm ổn định các tác vụ multi-turn, và train bằng GRPO giúp agent ổn định hơn nhiều.

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl

## ART của OpenPipe

Docs giới thiệu **ART** (Agent Reinforcement Trainer) của OpenPipe, xây trên GRPOTrainer của Unsloth. ART bổ sung ba thứ:

1. **Multi-turn training** qua khái niệm **trajectory** (quỹ đạo: toàn bộ lịch sử hội thoại và hành động của agent). Trajectory được chấm điểm rồi đưa vào GRPO. Cách này hỗ trợ cả tool call và gọi sub-agent.
2. **Tích hợp vào code có sẵn.** ART tách thành hai phần. "Frontend" là client nằm trong codebase của bạn. "Backend" là nơi train và phục vụ model qua API tương thích OpenAI. Hai phần có thể chạy chung một máy bằng `LocalBackend`.
3. **RULER** (Relative Universal LLM-Elicited Rewards): reward function tổng quát, dùng LLM làm giám khảo. RULER có thể thay reward function viết tay. Theo docs, agent train bằng RULER thường ngang hoặc hơn agent train bằng reward viết tay, và RULER giúp rút ngắn thời gian phát triển 2–3 lần.

Ví dụ trong docs:

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

Cài đặt theo docs:

```bash
pip install openpipe-art # or `uv add openpipe-art`
```

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl

## Khi nào nên chọn ART

Theo docs, bạn nên chọn ART khi:

- Agent phải làm nhiều bước hoặc gọi tool.
- Bạn muốn làm prototype nhanh mà chưa viết reward.
- Bạn muốn thêm RL vào một codebase agent có sẵn mà sửa ít nhất.

Các ví dụ docs nêu: agent truy xuất email (vượt o3), agent chơi game (2048, Tic Tac Toe, Codenames), tác vụ suy luận (Temporal Clue).

**[Nhận định]** RULER dựa vào một LLM giám khảo bên ngoài (ví dụ `"openai/o3"` trong code). Vì vậy bạn tốn thêm chi phí API và phụ thuộc vào dịch vụ đó.

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl

## Đọc tiếp

- [Lỗi thường gặp](/reinforcement-learning/loi-thuong-gap) — trang kế tiếp: các lỗi hay gặp khi train RL, gom theo nhóm.
- [Viết reward function](/reinforcement-learning/reward-function) — so sánh RULER với reward function viết tay.
- [Tiết kiệm VRAM khi chạy RL](/reinforcement-learning/memory-efficient) — agent vẫn train bằng GRPO nên cũng tốn VRAM như RL thường.
