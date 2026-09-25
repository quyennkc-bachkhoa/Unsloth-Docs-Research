---
title: Reinforcement Learning
description: "RL trong Unsloth ở mức khái niệm: RL khác SFT thế nào, GRPO và reward function, DPO/ORPO/KTO, khi nào dùng gì, Memory-efficient RL, reward hacking và huấn luyện agent."
---

# Reinforcement Learning

Reinforcement Learning (RL, học tăng cường) là cách dạy model bằng điểm thưởng thay vì bằng đáp án mẫu. Phần này giải thích RL trong Unsloth ở mức khái niệm và giúp bạn chọn phương pháp; muốn làm theo từng bước thì xem tutorial GRPO chính thức (link ở cuối trang [Train bằng GRPO](/reinforcement-learning/grpo)).

::: tip Tóm tắt
- **Dùng khi:** bạn đã fine-tune thường (SFT) và muốn model học theo điểm thưởng hoặc theo cặp câu trả lời tốt/xấu, hoặc đang phân vân giữa SFT, DPO và GRPO.
- **Kết quả:** hiểu RL khác SFT ở đâu, các tên RLHF, PPO, GRPO, RLVR nghĩa là gì, và biết nên chọn phương pháp nào theo dữ liệu mình có.
- **Nên biết trước:** [RL & Preference](/kien-thuc-nen/rl-va-preference) (SFT vs RL, reward, policy là gì) và [Fine-tuning](/fine-tuning/) (SFT nói chung).
:::

## Các trang trong phần này

| Trang | Giúp bạn làm gì | Đọc khi nào |
| --- | --- | --- |
| [Train bằng GRPO](/reinforcement-learning/grpo) | Hiểu vòng lặp train GRPO, dữ liệu cần có, VRAM cần cho từng cỡ model và các mẹo từ docs | Khi đã chọn GRPO, phương pháp RL chính mà Unsloth hướng dẫn |
| [Viết reward function](/reinforcement-learning/reward-function) | Chấm điểm câu trả lời trong GRPO, xem ví dụ từ docs, bộ reward GSM8K và cách chống reward hacking | Khi cần viết reward function hoặc verifier cho tác vụ của mình |
| [Train theo cặp tốt/xấu (DPO, ORPO, KTO)](/reinforcement-learning/dpo-orpo-kto) | Dạy model bằng cách so sánh câu trả lời tốt và tệ, không cần viết reward function; có code DPO chép từ docs | Khi bạn có sẵn các cặp so sánh "câu này tốt hơn câu kia" |
| [Tiết kiệm VRAM khi chạy RL](/reinforcement-learning/memory-efficient) | Hiểu vì sao RL tốn bộ nhớ GPU hơn fine-tune thường và các kỹ thuật giảm như Standby, FP8 RL | Khi RL bị thiếu VRAM hoặc muốn context dài hơn |
| [Train AI agent](/reinforcement-learning/agent) | Train agent làm nhiều bước liên tiếp bằng ART của OpenPipe và reward RULER | Khi model cần gọi tool hoặc làm tác vụ nhiều lượt |
| [Lỗi thường gặp](/reinforcement-learning/loi-thuong-gap) | Tra các lỗi hay gặp khi train RL, xếp theo nhóm: train chưa đủ, dữ liệu và model chưa đúng, cấu hình máy | Khi reward không tăng hoặc train bị lỗi |

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

### Khác SFT ở đâu

SFT (Supervised Fine-Tuning, fine-tune có giám sát, tức "fine-tune thường") chỉ tối đa hóa xác suất dự đoán từ tiếp theo theo dữ liệu mẫu. GRPO thì tối ưu theo reward function. Nhờ vậy model học *cách* đi tới đáp án, thay vì chỉ ghi nhớ và lặp lại câu trả lời trong dữ liệu.

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide

## Vì sao RL "chạy được"

RL chạy được vì mỗi câu trả lời, kể cả câu sai, đều là tín hiệu; docs gọi ý này là "Patience is All You Need". Model chưa huấn luyện có thể trả lời "0, cat, -10, 1928, 3, A, B..." rồi bỗng ra "4". Reward tương ứng là 0, 0, 0... rồi 1.

RL không chỉ ngồi chờ đáp án đúng xuất hiện. Mỗi câu trả lời sai cũng là một tín hiệu, giúp đẩy phân phối đầu ra của model ra xa vùng sai.

Điều kiện bắt buộc: xác suất ra đáp án đúng phải lớn hơn 0. Nếu xác suất đó luôn bằng 0 thì RL không bao giờ hoạt động. Vì lý do này, người ta hay làm RL trên model đã instruction-finetune (đã fine-tune để làm theo chỉ dẫn).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide

## Từ RLHF, PPO tới GRPO {#rlhf-ppo-grpo}

Khi đọc về RL cho LLM, bạn sẽ gặp bốn cái tên RLHF, PPO, GRPO và RLVR; điều cần nhớ nhất là GRPO là bản gọn hơn của PPO: nó bỏ bớt hai model (Value Model và Reward Model), nên tốn ít bộ nhớ hơn và chạy nhanh hơn.

### RLHF

**RLHF** (Reinforcement Learning from Human Feedback, RL từ phản hồi của con người) là khái niệm do OpenAI phổ biến. Model học từ đánh giá của con người.

Ví dụ: nút thích hoặc không thích trong ChatGPT tạo ra dữ liệu cho RLHF.

### PPO

**PPO** (Proximal Policy Optimization) là thuật toán dùng để làm RLHF. Theo docs, PPO cần bốn thành phần:

| Thành phần | Vai trò |
| --- | --- |
| Generating Policy | Model đang train |
| Reference Policy | Model gốc |
| Value Model | Ước lượng reward trung bình |
| Reward Model | Tính reward |

### GRPO

**GRPO** (Group Relative Policy Optimization) do DeepSeek phát triển để train các model suy luận R1. GRPO khác PPO ở hai thành phần:

| Thành phần | PPO | GRPO |
| --- | --- | --- |
| Value Model | Có | Bỏ. Thay bằng thống kê thu được khi gọi hàm chấm điểm nhiều lần |
| Reward Model | Có | Bỏ. Thay bằng reward function tự viết (có thể dùng RLVR) |

Vì bớt được hai model, GRPO tiết kiệm bộ nhớ và chạy nhanh hơn.

#### "Group Relative" nghĩa là gì

GRPO không chấm từng câu trả lời riêng lẻ. Nó so mỗi câu với cả nhóm câu trả lời cho cùng một câu hỏi:

<div class="dg">
<div class="dg-flow">
<div class="dg-node"><div><span class="dg-n">1</span>Sinh nhiều câu trả lời</div><small>vd 4 lần cho "What is 2+2?": 4, 3, D, C</small></div>
<div class="dg-node"><div><span class="dg-n">2</span>Chấm reward</div><small>cho từng câu</small></div>
<div class="dg-node"><div><span class="dg-n">3</span>Tính trung bình và độ lệch chuẩn</div><small>của cả nhóm</small></div>
<div class="dg-node is-end"><div><span class="dg-n">4</span>Chuẩn hóa Z-score</div><small>ra advantage của từng câu</small></div>
</div>
</div>

Kết quả gọi là **advantage**: lợi thế tương đối của một câu so với cả nhóm. GRPO dùng advantage thay cho Value Model.

### RLVR

**RLVR** (Reinforcement Learning with Verifiable Rewards, RL với phần thưởng kiểm chứng được) chấm điểm dựa trên những tác vụ dễ kiểm tra đúng sai. Ví dụ:

- Phép toán: 2+2=4.
- Code: chạy đúng hay không.

Docs nhấn mạnh GRPO không chỉ dùng cho toán và code. Mẹo là thiết kế một **rubric**: một danh sách nhiều reward nhỏ kiểm chứng được, thay vì một reward duy nhất bao trùm tất cả. Ví dụ cụ thể ở trang [Viết reward function](/reinforcement-learning/reward-function).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide

## Nên chọn SFT, DPO hay GRPO {#chon-phuong-phap}

Cách chọn phụ thuộc vào dữ liệu bạn có: đáp án mẫu, cặp so sánh tốt/xấu, hay một cách chấm điểm tự động. Bảng dưới so sánh ba hướng.

| Tiêu chí | SFT | DPO (và ORPO/KTO) | GRPO |
| --- | --- | --- | --- |
| Tối ưu cái gì (theo docs) | Xác suất dự đoán từ tiếp theo | Căn chỉnh theo preference (sở thích) | Tối đa hóa reward từ reward function |
| Dữ liệu cần | Cặp đầu vào → đầu ra mẫu | Dữ liệu preference (định dạng: cần kiểm tra lại) | Câu hỏi + đáp án (không kèm lập luận) + reward function/verifier |
| Lượng dữ liệu theo docs | (xem trang [Fine-tuning](/fine-tuning/)) | Docs không nêu | Tối ưu từ 500 dòng; thử được với 10 dòng |
| Hợp với | **[Nhận định]** Dạy model format, phong cách, kiến thức miền khi có sẵn đáp án mẫu | **[Nhận định]** Khi có sẵn các cặp so sánh "câu này tốt hơn câu kia" | Tác vụ kiểm chứng được (toán, code); suy luận; email, truy vấn DB, luật, y khoa nếu có rubric tốt |
| Chi phí | **[Nhận định]** Rẻ nhất | **[Nhận định]** Trung bình (ví dụ trong docs xuất phát từ model đã SFT) | Cao: sinh nhiều câu trả lời mỗi prompt, cần tối thiểu ~300 bước |

**[Nhận định]** Một thứ tự thường gặp là SFT trước, rồi mới DPO hoặc GRPO. Hai chi tiết trong docs đều khớp với thứ tự này:

- Code DPO xuất phát từ model `zephyr-sft`.
- Docs khuyên làm RL trên model đã instruction-finetune để xác suất ra đáp án đúng lớn hơn 0.

Docs cũng nhắc notebook Advanced GRPO dùng "pre-finetuning" để tránh việc GRPO chỉ học định dạng.

**[Nhận định]** Nếu bạn không viết được reward function hay verifier đáng tin cho tác vụ, GRPO dễ gặp [reward hacking](/reinforcement-learning/reward-function#reward-hacking). Khi đó SFT hoặc DPO an toàn hơn.

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo

## Đọc tiếp

- [Train bằng GRPO](/reinforcement-learning/grpo) — bước tiếp theo: vòng lặp train, dữ liệu và VRAM cho phương pháp RL chính của Unsloth.
- [Viết reward function](/reinforcement-learning/reward-function) — GRPO chỉ tốt khi cách chấm điểm tốt, trang này có ví dụ và cách chống reward hacking.
- [Train theo cặp tốt/xấu (DPO, ORPO, KTO)](/reinforcement-learning/dpo-orpo-kto) — lựa chọn khi bạn có cặp so sánh tốt/xấu thay vì reward function.
- [RL & Preference](/kien-thuc-nen/rl-va-preference) — ôn lại khái niệm reward, policy, preference nếu phần trên còn lạ.
