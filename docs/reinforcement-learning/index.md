---
title: Reinforcement Learning
description: "RL trong Unsloth ở mức khái niệm: RL khác SFT thế nào, GRPO và reward function, DPO/ORPO/KTO, khi nào dùng gì, Memory-efficient RL, reward hacking và huấn luyện agent."
---

# Reinforcement Learning (RL)

Phần Reinforcement Learning gồm trang này và chín trang con. Nội dung giải thích RL trong Unsloth ở mức khái niệm và giúp bạn chọn phương pháp. Nếu muốn làm theo từng bước, hãy xem tutorial GRPO chính thức (link ở cuối trang [GRPO trong Unsloth](/reinforcement-learning/grpo)).

Nếu mới bắt đầu, bạn đọc theo thứ tự từ trên xuống:

- **Reinforcement Learning** (trang này): RL là gì, khác SFT ở đâu, và vì sao RL "chạy được".
  1. [Từ RLHF, PPO tới GRPO](/reinforcement-learning/rlhf-ppo-grpo): các tên bạn sẽ gặp khi đọc về RL cho LLM. GRPO là bản gọn của PPO, bỏ bớt hai model nên tốn ít bộ nhớ hơn.
  1. [GRPO trong Unsloth](/reinforcement-learning/grpo): phương pháp RL chính mà Unsloth hướng dẫn. Trang này có vòng lặp train, dữ liệu cần có, VRAM cần cho từng cỡ model và các mẹo từ docs.
  1. [Reward function và verifier](/reinforcement-learning/reward-function): cách chấm điểm câu trả lời trong GRPO, kèm các ví dụ từ docs và bộ reward GSM8K dùng trong notebook mẫu.
  1. [DPO, ORPO, KTO](/reinforcement-learning/dpo-orpo-kto): dạy model bằng cách so sánh câu trả lời tốt và câu trả lời tệ, không cần viết reward function. Có code DPO chép từ docs.
  1. [Khi nào dùng gì: SFT vs DPO vs GRPO](/reinforcement-learning/chon-phuong-phap): bảng so sánh ba hướng để chọn theo dữ liệu bạn có.
  1. [Memory-efficient RL](/reinforcement-learning/memory-efficient): vì sao RL tốn bộ nhớ GPU hơn fine-tune thường, và các kỹ thuật Unsloth dùng để giảm, như Standby và FP8 RL.
  1. [Reward hacking](/reinforcement-learning/reward-hacking): khi model tìm mẹo để tăng reward mà không làm đúng việc, và cách chống mà docs đưa ra.
  1. [Huấn luyện AI agent bằng RL](/reinforcement-learning/agent): train agent làm nhiều bước liên tiếp bằng ART của OpenPipe và reward RULER.
  1. [Lỗi thường gặp](/reinforcement-learning/loi-thuong-gap): các lỗi hay gặp khi train RL, xếp theo nhóm: train chưa đủ, dữ liệu và model chưa đúng, cấu hình máy.

## RL là gì, khác SFT thế nào

RL là cách dạy model bằng điểm thưởng thay vì bằng đáp án mẫu. Model tự thử, được chấm điểm, rồi dần nghiêng về những câu trả lời được điểm cao.

Reinforcement Learning (học tăng cường) giúp một "agent" (tác nhân) học cách ra quyết định. Ở đây agent chính là LLM. Agent tương tác với environment (môi trường) và nhận feedback dưới dạng reward (điểm thưởng) hoặc penalty (điểm phạt). Docs Unsloth chia RL thành ba thành phần:

| Thành phần | Nghĩa | Ví dụ trong docs |
| --- | --- | --- |
| Action (hành động) | Thứ model sinh ra | Một câu trả lời |
| Reward (điểm thưởng) | Tín hiệu cho biết action tốt hay tệ | Câu trả lời có làm đúng yêu cầu không, có hữu ích không |
| Environment (môi trường) | Bối cảnh hoặc tác vụ model đang làm | Trả lời câu hỏi của người dùng |

Mục tiêu của RL gói gọn trong hai ý: tăng xác suất ra kết quả "tốt" và giảm xác suất ra kết quả "xấu".

Docs lấy ví dụ câu hỏi "What is 2 + 2?". Một model chưa được căn chỉnh có thể trả lời 3, 4, C, D, -10... Bạn có thể đặt ra quy ước chấm điểm:

- Ra số thì tốt hơn ra chữ C hoặc D.
- Ra 3 thì tốt hơn ra 8.
- Ra 4 là đúng.

Bộ quy ước đó chính là một **reward function** (hàm chấm điểm).

**Khác SFT ở đâu:** SFT (Supervised Fine-Tuning, fine-tune có giám sát, tức "fine-tune thường") chỉ tối đa hóa xác suất dự đoán từ tiếp theo theo dữ liệu mẫu. GRPO thì tối ưu theo reward function. Nhờ vậy model học *cách* đi tới đáp án, thay vì chỉ ghi nhớ và lặp lại câu trả lời trong dữ liệu.

::: tip Kiến thức nền
Chưa rõ SFT vs RL, reward, policy là gì? Xem [RL & Preference](/kien-thuc-nen/rl-va-preference). Về SFT nói chung, xem [Fine-tuning](/fine-tuning/).
:::

**Vì sao RL "chạy được":** docs gọi ý này là "Patience is All You Need". Model chưa huấn luyện có thể trả lời "0, cat, -10, 1928, 3, A, B..." rồi bỗng ra "4". Reward tương ứng là 0, 0, 0... rồi 1.

RL không chỉ ngồi chờ đáp án đúng xuất hiện. Mỗi câu trả lời sai cũng là một tín hiệu, giúp đẩy phân phối đầu ra của model ra xa vùng sai.

Điều kiện bắt buộc: xác suất ra đáp án đúng phải lớn hơn 0. Nếu xác suất đó luôn bằng 0 thì RL không bao giờ hoạt động. Vì lý do này, người ta hay làm RL trên model đã instruction-finetune (đã fine-tune để làm theo chỉ dẫn).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
