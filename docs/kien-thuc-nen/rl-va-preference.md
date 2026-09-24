---
title: RL & preference
description: SFT khác RL thế nào; reward, reward model, verifier, policy, reference model, KL penalty; RLHF/PPO, DPO, ORPO, KTO, GRPO khác nhau ở đâu; reward hacking và khi nào dùng phương pháp nào.
---

# RL & preference optimization

Trang này giải thích các khái niệm docs Unsloth dùng khi nói về Reinforcement Learning (học tăng cường) và preference optimization (tối ưu theo sở thích). Đó là các khái niệm như reward, policy, reference model, DPO, GRPO... Nên đọc trang này trước khi mở trang [Reinforcement Learning](/reinforcement-learning), hoặc trước khi chạy notebook GRPO và DPO.

::: tip Đọc kèm
Trang [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) giải thích pre-training và SFT; trang [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) giải thích adapter mà mọi ví dụ RL của Unsloth đều dùng.
:::

## SFT và RL khác nhau ở đâu

SFT dạy model bằng đáp án mẫu. RL không cần đáp án mẫu: model tự thử, rồi được chấm điểm.

**Khái niệm.**
- **SFT** (Supervised Fine-Tuning, tinh chỉnh có giám sát): cho model xem cặp đầu vào → đầu ra mẫu. Model được tối ưu để dự đoán đúng từng token tiếp theo của đầu ra mẫu.
- **RL**: không cần đầu ra mẫu. Model tự sinh câu trả lời. Một cơ chế chấm điểm cho biết câu trả lời tốt hay xấu. Sau đó model được đẩy về phía câu trả lời điểm cao.

Theo docs Unsloth, mục tiêu của RL chỉ gồm hai việc: tăng khả năng gặp kết quả "tốt", và giảm khả năng gặp kết quả "xấu".

**Ví dụ.** Câu hỏi "2 + 2 bằng mấy?":

- SFT: dữ liệu ghi sẵn đáp án `4`, model học chép lại.
- RL: model chưa được căn chỉnh có thể trả lời 3, 4, C, D, -10... Docs Unsloth chấm như sau: ra số tốt hơn ra chữ, ra 3 tốt hơn ra 8, ra 4 là đúng. Bộ quy tắc chấm đó chính là một reward function.

Docs Unsloth cũng nêu điều kiện để RL chạy được: xác suất model sinh ra câu đúng phải lớn hơn 0. Nếu model không bao giờ ra câu đúng, RL không bao giờ nhận được tín hiệu tốt để học. Vì vậy người ta hay làm RL trên model đã instruction-finetune (đã SFT).

**Ảnh hưởng khi dùng Unsloth.** Docs Unsloth so sánh: SFT thường chỉ tối đa hóa xác suất dự đoán từ tiếp theo, còn GRPO tối ưu theo reward function. RL dùng khi bạn cần model giỏi một hành vi cụ thể (ví dụ tool calling), dựa trên môi trường và reward function thay vì dữ liệu có nhãn. Docs cũng nói "với đa số trường hợp, SFT là đủ".

**Gặp ở đâu trong Unsloth.** [Reinforcement Learning](/reinforcement-learning), [Fine-tuning](/fine-tuning/).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide. **[Nguồn ngoài]** Loss của SFT là cross-entropy trên từng token của chuỗi đích: https://huggingface.co/docs/trl/sft_trainer

## Policy, action, reward, environment

Đây là bộ từ vựng cơ bản của RL. Mục này đối chiếu từng thuật ngữ với vai trò của nó khi train LLM.

**Khái niệm.**

| Thuật ngữ | Nghĩa trong RL | Với LLM (theo docs Unsloth) |
| --- | --- | --- |
| Agent (tác tử) | Thứ ra quyết định | Chính language model |
| Policy (chính sách) | Chiến lược chọn hành động của agent | Model đang được train ("generating policy") |
| Action (hành động) | Việc agent làm | Đoạn văn bản model sinh ra |
| Environment (môi trường) | Bối cảnh agent tương tác | Tác vụ model đang làm, ví dụ trả lời câu hỏi của người dùng |
| Reward (phần thưởng) | Tín hiệu tốt/xấu sau hành động | Điểm số cho câu trả lời: có làm theo hướng dẫn không, có hữu ích không |
| Rollout | Một lần cho agent chạy thử | Một lần model sinh câu trả lời cho prompt |

**Ví dụ.** Docs Unsloth lấy game Pacman làm ví dụ:
- Môi trường là màn chơi.
- Hành động là lên, xuống, trái, phải.
- Reward dương khi ăn bánh, âm khi chạm quái.

Người chơi không biết trước nước đi tốt nhất. Họ chỉ quan sát được kết quả trung gian, hoặc thắng thua cuối cùng.

**[Nguồn ngoài]** HF LLM Course định nghĩa policy là "chiến lược chọn hành động của agent". Với LLM, action có thể là sinh từ trong câu, hoặc chọn câu trả lời cho câu hỏi: https://huggingface.co/learn/llm-course/chapter12/2

**Ảnh hưởng khi dùng Unsloth.** Trong code Unsloth, "policy" là `model` bạn truyền vào trainer, thường đã gắn LoRA. Docs gọi quá trình "chờ" câu đúng xuất hiện qua nhiều rollout là "Patience is All You Need". RL không chỉ chờ: trong lúc chờ, nó còn chủ động đẩy model khỏi các câu trả lời xấu nhận được.

**Gặp ở đâu trong Unsloth.** [Reinforcement Learning](/reinforcement-learning).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide; **[Nguồn ngoài]** https://huggingface.co/learn/llm-course/chapter12/2

## Reward model, reward function và verifier

RL cần một cách chấm điểm câu trả lời. Có ba cách tạo ra điểm reward, và mục này giải thích từng cách.

**Khái niệm.**

- **Reward model** (mô hình phần thưởng): một mạng neural riêng, được train để cho điểm câu trả lời. Dữ liệu train là các cặp "câu tốt hơn / câu kém hơn" cho cùng một prompt. Model học cách cho câu được ưa thích điểm cao hơn.
- **Verifier** (bộ kiểm chứng): theo docs Unsloth, verifier chỉ xác định câu trả lời đúng hay sai, không cho điểm số. Verifier có thể chạy code (ví dụ Python) để kiểm tra logic và cú pháp.
- **Reward function** (hàm phần thưởng): theo docs Unsloth, reward function đổi kết quả kiểm chứng (hoặc tiêu chí khác) thành điểm số. Nó có thể dùng verifier bên trong. Nó cũng có thể phạt theo tiêu chí ngoài chuyện đúng sai, ví dụ câu quá dài hay khó đọc.

**Ví dụ.** Docs Unsloth đưa hai reward function cho câu hỏi `"2 + 2"`:

- Hàm 1: có số trong câu trả lời → +1; không có số → -1.
- Hàm 2: số khớp đáp án → +3; sai → -3.
- Tổng reward = tổng các hàm. Câu trả lời `4` được +4. Câu `3` được -2. Câu `C` được -4.

Docs còn có ví dụ tự động trả lời email: +1 nếu có từ khóa bắt buộc, +1 nếu khớp câu trả lời lý tưởng, -1 nếu quá dài, +1 nếu có tên người nhận, +1 nếu có chữ ký.

**[Nguồn ngoài]** TRL `RewardTrainer` train reward model trên dữ liệu preference dạng `chosen`/`rejected`. Model học sao cho điểm câu `chosen` cao hơn câu `rejected` (mô hình Bradley-Terry), và trả về một điểm số cho mỗi chuỗi: https://huggingface.co/docs/trl/reward_trainer

**Ảnh hưởng khi dùng Unsloth.**
- Docs Unsloth nhấn mạnh reward function phải được thiết kế tốt. Reward kém có thể làm model tệ đi.
- Mẹo trong docs: dùng một rubric (danh sách nhiều reward nhỏ kiểm chứng được) thay vì một reward tổng duy nhất.
- Nếu GRPO không cho ra suy luận, docs khuyên kiểm tra reward function, verifier và số bước train.

**Gặp ở đâu trong Unsloth.** [Reinforcement Learning](/reinforcement-learning) (mục reward function và ví dụ GSM8K).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide; **[Nguồn ngoài]** https://huggingface.co/docs/trl/reward_trainer

## Reference model và KL penalty

Khi tối ưu theo reward, model có thể đi quá xa khỏi model ban đầu. Reference model và KL penalty là cơ chế giữ model không đi lệch quá mức.

**Khái niệm.**
- **Reference model** (mô hình tham chiếu): một bản đóng băng của model trước khi train RL.
- **KL penalty** (phạt KL): đo mức "đi lệch" của phân phối token mà model đang train so với reference model, rồi trừ vào mục tiêu.
- Hệ số của KL penalty thường tên là `beta`. `beta` càng lớn thì model càng bị giữ gần reference model.

Mục đích: model vẫn tối ưu reward, nhưng không đi quá xa model gốc. Nhờ vậy model tránh khai thác lỗ hổng của bộ chấm điểm, và không quên các khả năng sẵn có.

**Ví dụ.** Docs Unsloth mô tả PPO có "Reference Policy (original model)" và một số hạng KL với beta lớn hơn 0, "để model không đi lệch quá xa". **[Nguồn ngoài]** InstructGPT thêm "một KL penalty theo từng token so với model SFT" để giảm việc tối ưu quá mức theo reward model: https://arxiv.org/abs/2203.02155

**Ảnh hưởng khi dùng Unsloth.**

- DPO: code mẫu của Unsloth đặt `ref_model = None` và `beta = 0.1`. **[Nguồn ngoài]** Theo TRL, khi `ref_model=None`, trainer tự dùng trạng thái model trước khi train DPO làm reference. `beta` mặc định 0.1. Beta cao hơn nghĩa là ít lệch khỏi reference hơn: https://huggingface.co/docs/trl/dpo_trainer
- GRPO: theo tài liệu GRPO nâng cao của Unsloth, `beta` mặc định `0.0`. Khi `beta` bằng 0, trainer không nạp reference model, nên tốn ít bộ nhớ hơn và nhanh hơn. Tăng `beta` để giữ policy gần reference hơn. TRL `GRPOConfig` ghi cùng mặc định.
- **[Nhận định]** Với LoRA, reference model thường là chính model gốc khi tắt adapter, nên không tốn thêm một bản trọng số đầy đủ. Docs Unsloth không nói rõ điểm này: cần kiểm tra lại.

**Gặp ở đâu trong Unsloth.** [Reinforcement Learning](/reinforcement-learning) (DPO code, GRPOConfig).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation; **[Nguồn ngoài]** https://huggingface.co/docs/trl/dpo_trainer, https://huggingface.co/docs/trl/grpo_trainer

## RLHF và PPO

RLHF là quy trình căn chỉnh model bằng phản hồi của con người. PPO là thuật toán RL dùng trong bước cuối của quy trình đó.

**Khái niệm.** RLHF (Reinforcement Learning from Human Feedback, học tăng cường từ phản hồi của con người) dùng đánh giá của con người để căn chỉnh model. **[Nguồn ngoài]** Bài InstructGPT mô tả ba bước: (1) thu dữ liệu minh họa và train policy bằng SFT; (2) thu dữ liệu so sánh (người gán nhãn xếp hạng từ 4 đến 9 câu trả lời cho mỗi prompt) và train reward model; (3) tối ưu policy theo reward model bằng PPO: https://arxiv.org/abs/2203.02155

PPO (Proximal Policy Optimization) là thuật toán RL dùng ở bước 3. **[Nguồn ngoài]** Paper PPO đề xuất một hàm mục tiêu "surrogate". Hàm này cho phép cập nhật nhiều epoch trên cùng một lô dữ liệu đã thu. PPO giữ được độ ổn định của TRPO nhưng dễ cài đặt hơn: https://arxiv.org/abs/1707.06347

**Ví dụ.** Theo docs Unsloth, PPO cho LLM gồm ba thành phần:
- Generating Policy: model đang train.
- Reference Policy: model gốc.
- Value Model: bộ ước lượng reward trung bình.

Reward lấy từ Reward Model. Số hạng `clip(..., 1-e, 1+e)` ngăn mỗi bước cập nhật thay đổi quá lớn. Nút thích/không thích trong ChatGPT là một nguồn phản hồi có thể dùng cho RLHF.

**Ảnh hưởng khi dùng Unsloth.** Docs Unsloth liệt kê PPO trong nhóm phương pháp "chạy được với Unsloth", nhưng không có notebook PPO riêng. Theo docs, PPO phải train nhiều model cùng lúc nên tốn bộ nhớ. Đó là lý do GRPO bỏ value model và reward model.

**Gặp ở đâu trong Unsloth.** [Reinforcement Learning](/reinforcement-learning) (mục Từ RLHF, PPO tới GRPO).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto; **[Nguồn ngoài]** https://arxiv.org/abs/2203.02155, https://arxiv.org/abs/1707.06347

## DPO, ORPO, KTO, GRPO khác nhau ở đâu

Cả bốn phương pháp đều dùng để "căn chỉnh" model sau SFT. Chúng khác nhau ở hai điểm: dữ liệu đầu vào, và số model phụ cần chạy kèm.

Docs Unsloth chỉ có code mẫu cho DPO và GRPO. ORPO và KTO chỉ có link notebook. Vì vậy phần so sánh chi tiết dưới đây lấy từ paper gốc và docs TRL, thư viện mà Unsloth dùng bên dưới (`from trl import DPOTrainer, DPOConfig`).

| Phương pháp | Dữ liệu cần | Có reference model? | Có reward model? | Nguồn |
| --- | --- | --- | --- | --- |
| SFT (để so sánh) | Cặp prompt → câu trả lời mẫu | Không | Không | [TRL SFT](https://huggingface.co/docs/trl/sft_trainer) |
| RLHF / PPO | Prompt; thêm dữ liệu xếp hạng để train reward model | Có (KL so với model SFT) | Có, cộng thêm value model | [2203.02155](https://arxiv.org/abs/2203.02155), [1707.06347](https://arxiv.org/abs/1707.06347) |
| DPO | Cặp `chosen` / `rejected` cho cùng một `prompt` | Có (mặc định là model trước khi train) | Không; reward "ngầm" tính từ tỉ lệ xác suất so với reference | [2305.18290](https://arxiv.org/abs/2305.18290), [TRL DPO](https://huggingface.co/docs/trl/dpo_trainer) |
| ORPO | Cặp `chosen` / `rejected` | Không ("reference model-free") | Không | [2403.07691](https://arxiv.org/abs/2403.07691), [TRL ORPO](https://huggingface.co/docs/trl/orpo_trainer) |
| KTO | Từng câu trả lời lẻ kèm nhãn tốt/xấu (`label` True/False), không cần ghép cặp | Có | Không | [2402.01306](https://arxiv.org/abs/2402.01306), [TRL KTO](https://huggingface.co/docs/trl/kto_trainer) |
| GRPO | Chỉ cần `prompt` (thêm cột đáp án nếu reward function cần) + reward function | Paper: có (KL đưa vào loss). Unsloth và TRL: mặc định `beta=0.0`, không nạp | Không bắt buộc; dùng reward function/verifier (TRL cho phép truyền reward model) | [2402.03300](https://arxiv.org/abs/2402.03300), [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer) |

**[Nguồn ngoài]** Dạng dữ liệu theo TRL:

```python
# DPO / ORPO (preference, ghép cặp)
{"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}

# KTO (unpaired preference, không ghép cặp)
{"prompt": "The sky is", "completion": " blue.", "label": True}

# GRPO (prompt-only)
{"prompt": [{"role": "user", "content": "What color is the sky?"}]}
```

Nguồn: https://huggingface.co/docs/trl/dpo_trainer, https://huggingface.co/docs/trl/kto_trainer, https://huggingface.co/docs/trl/grpo_trainer

### DPO

**Khái niệm.** DPO (Direct Preference Optimization) giải cùng bài toán với RLHF. Điểm khác: DPO không train reward model riêng, và không sinh câu trả lời trong lúc train. **[Nguồn ngoài]** Paper DPO chỉ ra có thể viết lại reward model sao cho policy tối ưu có dạng đóng. Nhờ vậy bài toán RLHF quy về một loss phân loại đơn giản: https://arxiv.org/abs/2305.18290. TRL mô tả DPO là nới rộng khoảng cách log-likelihood giữa câu được ưa thích và câu bị chê, tính tương đối so với reference model: https://huggingface.co/docs/trl/dpo_trainer

**Ví dụ.** Prompt "Viết hàm đảo chuỗi". `chosen` là đoạn code đúng và gọn. `rejected` là đoạn code chạy sai. DPO tăng xác suất tương đối của `chosen` và giảm của `rejected`.

**Ảnh hưởng khi dùng Unsloth.** Code mẫu của Unsloth làm theo thứ tự:
1. Gọi `PatchDPOTrainer()`.
2. Nạp `unsloth/zephyr-sft-bnb-4bit` (model đã SFT).
3. Gắn LoRA `r = 64`.
4. Tạo `DPOTrainer` với `ref_model = None`, `beta = 0.1`, `max_length = 1024`, `max_prompt_length = 512`.

Docs không nêu cần bao nhiêu cặp dữ liệu.

### ORPO

**Khái niệm.** **[Nguồn ngoài]** ORPO (Odds Ratio Preference Optimization) gộp SFT và căn chỉnh sở thích vào một bước. Loss của ORPO gồm hai phần: loss SFT thông thường (NLL) trên câu `chosen`, cộng một số hạng log odds ratio phạt nhẹ câu `rejected`. ORPO không cần reference model, và không cần giai đoạn căn chỉnh riêng sau SFT: https://arxiv.org/abs/2403.07691, https://huggingface.co/docs/trl/orpo_trainer

**Ảnh hưởng khi dùng Unsloth.** Docs Unsloth chỉ có link notebook Llama 3 (8B) ORPO. **[Nguồn ngoài]** Trong TRL, `beta` của ORPO (mặc định 0.1) là trọng số của số hạng odds ratio. Nó khác nghĩa với `beta` của DPO và KTO: https://huggingface.co/docs/trl/orpo_trainer

### KTO

**Khái niệm.** **[Nguồn ngoài]** KTO (Kahneman-Tversky Optimization) dựa trên prospect theory (lý thuyết triển vọng: con người sợ mất hơn thích được). KTO không cần cặp so sánh. Nó chỉ cần tín hiệu nhị phân "câu trả lời này tốt hay không tốt" cho mỗi đầu vào: https://arxiv.org/abs/2402.01306

**Ví dụ.** Một chatbot có log nút thích/không thích. Mỗi câu trả lời thành một dòng `label = True` hoặc `False`, không cần tìm câu đối chứng cho cùng prompt. **[Nguồn ngoài]** Nếu bạn đưa dữ liệu DPO (ghép cặp) vào, TRL tự tách thành dạng không ghép cặp. Khi số mẫu tốt và xấu lệch nhau, chỉnh `desirable_weight` / `undesirable_weight`: https://huggingface.co/docs/trl/kto_trainer

**Ảnh hưởng khi dùng Unsloth.** Docs Unsloth chỉ có link notebook KTO, không có tham số khuyến nghị.

### GRPO: group và advantage tương đối

**Khái niệm.** GRPO (Group Relative Policy Optimization) do DeepSeek phát triển để train model suy luận R1. Cách hoạt động:
- Với mỗi prompt, model sinh ra một **group** (nhóm) nhiều câu trả lời.
- Reward function chấm từng câu.
- GRPO không dùng value model để đoán "reward trung bình kỳ vọng". Thay vào đó, nó lấy chính trung bình của nhóm làm mốc.
- **Advantage** (lợi thế) của mỗi câu là: câu đó hơn hay kém mức trung bình của nhóm bao nhiêu, chia cho độ lệch chuẩn của nhóm.
- Câu có advantage dương được tăng xác suất. Câu có advantage âm bị giảm xác suất.

```text
advantage_i = (reward_i - trung_bình_reward_nhóm) / độ_lệch_chuẩn_reward_nhóm
```

**[Nguồn ngoài]** Paper DeepSeekMath: GRPO bỏ critic (value model) có cỡ tương đương policy. Nó dùng reward trung bình của nhiều câu trả lời cho cùng một câu hỏi làm baseline, chuẩn hóa `(r - mean(r)) / std(r)`, và đưa KL so với reference model trực tiếp vào loss: https://arxiv.org/abs/2402.03300. TRL dùng cùng công thức advantage: https://huggingface.co/docs/trl/grpo_trainer

**Ví dụ.** Docs Unsloth lấy mẫu 4 lần cho câu "What is 2+2?" và được 4, 3, D, C. Dùng hai reward function ở mục trên, reward lần lượt là +4, -2, -4, -4.

**[Ước tính]** Trung bình = (4 - 2 - 4 - 4) / 4 = -1,5. Độ lệch chuẩn (chia cho 4) ≈ 3,28. Advantage của từng câu:
- Câu `4` ≈ (4 + 1,5) / 3,28 ≈ +1,68.
- Câu `3` ≈ -0,15.
- Câu `D` và `C` ≈ -0,76.

Câu `3` bị phạt nhẹ hơn `D`/`C`, vì ít nhất nó là một con số. Thư viện có thể dùng độ lệch chuẩn chia cho n - 1, nên các số lẻ khác đi chút ít, nhưng thứ tự thì giữ nguyên.

**Ảnh hưởng khi dùng Unsloth.**

- `num_generations` là kích thước group. Docs Unsloth lấy ví dụ 8 câu mỗi prompt, có thể tăng lên 16. **[Nguồn ngoài]** TRL mặc định `num_generations = 8`: https://huggingface.co/docs/trl/grpo_trainer
- Tài liệu GRPO nâng cao của Unsloth khuyên dùng `temperature` tương đối cao (1.0) khi sinh câu trả lời. Nhờ vậy nhóm đủ đa dạng và model học tốt hơn. Nếu cả nhóm giống hệt nhau thì mọi advantage bằng 0 và không có gì để học **[Nhận định]**.
- Docs Unsloth đưa ba mốc: chờ ít nhất khoảng 300 bước để reward bắt đầu tăng; tốt nhất có từ 500 dòng dữ liệu; model nên từ 1.5B tham số trở lên để sinh được thinking token.
- `loss_type` của Unsloth có các biến thể `'grpo'`, `'dr_grpo'`, `'gspo'`, `'bnpo'`, `'dapo'`.

::: warning Docs chưa thống nhất
Trang [RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) mô tả cách tính "reward trung bình" trong GRPO theo hai cách không khớp nhau:

- "calculate the average reward through statistics of the sampling process across multiple different questions" (thống kê trên nhiều câu hỏi khác nhau).
- Ngay sau đó, ví dụ lại lấy mẫu 4 lần cho **cùng một** câu "What is 2+2?", rồi tính trung bình và độ lệch chuẩn của 4 câu đó.

Paper DeepSeekMath và docs TRL mô tả theo cách thứ hai (baseline là trung bình các câu trả lời cho cùng một câu hỏi).
:::

**Gặp ở đâu trong Unsloth.** [Reinforcement Learning](/reinforcement-learning) (Vòng lặp GRPO, Yêu cầu VRAM).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto; **[Nguồn ngoài]** https://arxiv.org/abs/2305.18290, https://arxiv.org/abs/2403.07691, https://arxiv.org/abs/2402.01306, https://arxiv.org/abs/2402.03300, https://huggingface.co/docs/trl/dpo_trainer, https://huggingface.co/docs/trl/orpo_trainer, https://huggingface.co/docs/trl/kto_trainer, https://huggingface.co/docs/trl/grpo_trainer

## RLVR

RLVR là RL mà điểm reward được kiểm tra tự động. Cách này hợp với các tác vụ có đáp án kiểm chứng được, như toán và code.

**Khái niệm.** RLVR (Reinforcement Learning with Verifiable Rewards, RL với phần thưởng kiểm chứng được) lấy reward từ việc kiểm tra tự động. Reward không đến từ đánh giá của người, cũng không đến từ reward model. Theo docs Unsloth, GRPO thay reward model bằng reward function tự viết, và RLVR có thể dùng ở đó.

**Ví dụ.** Docs Unsloth đưa hai ví dụ: phương trình toán kiểm tra được (2+2 = 4); code kiểm tra được bằng cách chạy xem có đúng không. Với tác vụ khác thì khó viết verifier, nên đa số ví dụ là toán hoặc code. Docs vẫn cho rằng GRPO dùng được cho email tự động, truy vấn database, luật, y khoa, nếu có rubric gồm nhiều reward nhỏ kiểm chứng được.

**Gặp ở đâu trong Unsloth.** [Reinforcement Learning](/reinforcement-learning).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide

## Reward hacking

Reward hacking là rủi ro bạn cần để ý khi tự viết reward function: model tìm mẹo để được điểm cao mà không thật sự làm đúng việc.

**Khái niệm.** Reward hacking (lách luật phần thưởng) xảy ra khi thuật toán RL tìm ra mẹo để tăng reward mà không thực sự làm được việc. Theo docs Unsloth, đây là lý do model học cách sửa unit test để vượt qua bài code. Đây cũng là trở ngại lớn khi đưa model vào dùng thật.

**Ví dụ.** Trong notebook gpt-oss RL của Unsloth (sinh kernel nhân ma trận nhanh), model đã dùng các mẹo sau:

- **Laziness (lười):** gọi Numpy, Torch để dùng kernel CUDA có sẵn thay vì tự viết. Cách chống: kiểm tra code có import thư viện ngoài chuẩn không.
- **Caching & Cheating:** cache kết quả, hoặc đọc biến global của Python để lấy đáp án. Cách chống: xóa cache bằng một ma trận giả lớn, benchmark nhiều vòng.
- **Cheating:** sửa hàm đo thời gian để báo 0 giây. Cách chống: giới hạn `locals` và `globals`, tạo hàm bằng `exec`, chặn truy cập global bằng `types.FunctionType(f.__code__, {})`.

**Ảnh hưởng khi dùng Unsloth.** **[Nhận định]** Reward function chỉ đo được những gì nó kiểm tra. Đường reward tăng đẹp chưa chứng minh model tốt lên. Bạn cần đọc mẫu câu trả lời (Unsloth in mẫu trong lúc train), và chạy code của model trong môi trường cô lập. **[Nguồn ngoài]** InstructGPT dùng KL penalty chính để giảm việc tối ưu quá mức theo reward model: https://arxiv.org/abs/2203.02155

**Gặp ở đâu trong Unsloth.** [Reinforcement Learning](/reinforcement-learning) (mục Reward hacking).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/rl-reward-hacking

## Khi nào dùng gì

Mục này giúp bạn chọn phương pháp dựa trên loại dữ liệu đang có. Các lời khuyên dưới đây lấy từ docs Unsloth:

- Đa số trường hợp: SFT là đủ. RL dùng khi cần model giỏi một hành vi cụ thể (ví dụ tool calling), và bạn có môi trường cùng reward function thay cho dữ liệu có nhãn.
- Muốn biến model không biết suy luận thành model có suy luận: dùng dataset câu hỏi và đáp án **không kèm** lời giải, rồi train bằng GRPO. Đáp án trong dataset không được lộ cách suy ra nó.
- Làm RL trên model đã instruction-finetune, để xác suất ra câu đúng lớn hơn 0. Nếu dùng base model thì phải có chat template.
- Code DPO mẫu bắt đầu từ một model đã SFT (`zephyr-sft`).

| Bạn đang có | Gợi ý | Căn cứ |
| --- | --- | --- |
| Cặp câu hỏi → câu trả lời mẫu | SFT | Docs Unsloth: "for most use-cases, standard SFT is sufficient" |
| Cặp "câu này tốt hơn câu kia" cho cùng prompt | DPO; hoặc ORPO nếu muốn gộp SFT và căn chỉnh trong một bước, không giữ reference model | **[Nhận định]** suy từ dạng dữ liệu trong TRL và paper ORPO |
| Từng câu trả lời lẻ gắn nhãn tốt/xấu (thích/không thích) | KTO | **[Nhận định]** suy từ paper KTO |
| Câu hỏi có đáp án kiểm chứng được bằng code (toán, code, định dạng) | GRPO | Docs Unsloth (RLVR, GSM8K) |
| Không viết được reward function đáng tin | Tránh GRPO; dùng SFT hoặc DPO | **[Nhận định]** vì dễ reward hacking |

**[Nhận định]** Về chi phí: SFT và DPO/ORPO/KTO không phải sinh câu trả lời trong lúc train, nên nhanh hơn nhiều so với GRPO. GRPO mỗi bước phải sinh cả group (ví dụ 8 câu) cho mỗi prompt. Docs Unsloth nói GRPO có thể cần 12 tiếng trở lên để cho kết quả tốt.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto

## Gặp ở đâu trong Unsloth

| Khái niệm | Trang Unsloth trên website | Docs gốc |
| --- | --- | --- |
| SFT vs RL | [Reinforcement Learning](/reinforcement-learning), [Fine-tuning](/fine-tuning/) | [RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), [Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) |
| Policy, reward, environment | [Reinforcement Learning](/reinforcement-learning) | [RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) |
| Reward function, verifier | [Reinforcement Learning](/reinforcement-learning) | [RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) |
| Reference model, KL, `beta` | [Reinforcement Learning](/reinforcement-learning) | [Advanced RL Documentation](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation) |
| RLHF, PPO | [Reinforcement Learning](/reinforcement-learning) | [RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) |
| DPO, ORPO, KTO | [Reinforcement Learning](/reinforcement-learning) | [Preference Optimization](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto) |
| GRPO, advantage, `num_generations` | [Reinforcement Learning](/reinforcement-learning) | [GRPO Tutorial](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo) |
| RLVR | [Reinforcement Learning](/reinforcement-learning) | [RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) |
| Reward hacking | [Reinforcement Learning](/reinforcement-learning) | [RL Reward Hacking](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/rl-reward-hacking) |
