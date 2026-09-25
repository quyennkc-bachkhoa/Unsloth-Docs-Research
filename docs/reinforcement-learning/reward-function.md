---
title: Reward function và verifier
description: "Phân biệt verifier và reward function, các ví dụ reward từ docs và bộ reward GSM8K dùng trong notebook mẫu."
---

# Reward function và verifier

Verifier trả lời câu hỏi "đúng hay sai". Reward function biến kết quả đó thành điểm số. Docs phân biệt hai khái niệm này, dù thực tế người ta thường dùng chung:

| | Verifier (bộ kiểm chứng) | Reward Function (hàm thưởng) |
| --- | --- | --- |
| Làm gì | Xác định câu trả lời đúng hay sai | Đổi kết quả kiểm chứng (hoặc tiêu chí khác) thành điểm số |
| Có cho điểm? | Không, chỉ đúng/sai | Có, ví dụ sai thì -1, -2; đúng thì +1, +2 |
| Ví dụ | Model trả lời "5" cho "2+2" thì gắn nhãn sai; có thể chạy code Python để kiểm tra | Có thể phạt cả tiêu chí ngoài tính đúng, như quá dài hay khó đọc |

Reward function có thể *dùng* verifier bên trong. Docs cũng cảnh báo: reward thiết kế kém có thể làm model tệ đi.

## Ví dụ reward function từ docs

Các ví dụ trong docs được mô tả bằng quy tắc, không phải bằng code Python.

**Ví dụ 1: phép cộng đơn giản** (Question `"2 + 2"`, Answer `"4"`)

| Reward function | Điều kiện | Điểm |
| --- | --- | --- |
| Reward Function 1 | Có số trong câu trả lời | +1 |
| | Không có số | -1 |
| Reward Function 2 | Số khớp đáp án đúng | +3 |
| | Sai | -3 |
| Tổng reward | Cộng tất cả reward function | |

**Ví dụ 2: tự động trả lời email** (Question: email đến, Answer: email trả lời)

| Điều kiện | Điểm |
| --- | --- |
| Có từ khóa bắt buộc | +1 |
| Khớp chính xác câu trả lời lý tưởng | +1 |
| Câu trả lời quá dài | -1 |
| Có tên người nhận | +1 |
| Có khối chữ ký (điện thoại, email, địa chỉ) | +1 |

**Bộ reward GSM8K** do @willccbb viết, được dùng trong các notebook mẫu. Bộ này có tổng cộng 5 hàm:

| Hàm | Kiểm tra gì |
| --- | --- |
| `correctness_reward_func` | Thưởng khi khớp chính xác nhãn đáp án |
| `int_reward_func` | Khuyến khích đáp án chỉ là số nguyên |
| `soft_format_reward_func` | Kiểm tra cấu trúc nhưng chấp nhận lệch dấu xuống dòng |
| `strict_format_reward_func` | Cấu trúc phải khớp prompt, kể cả xuống dòng |
| `xmlcount_reward_func` | Mỗi thẻ XML xuất hiện đúng một lần |

Các hàm format ở trên kiểm tra định dạng được quy định trong system prompt của tutorial:

```
# Define the system prompt that instructs the model to use a specific format
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
```

Notebook Advanced GRPO của Unsloth còn có một **proximity-based reward function** (hàm thưởng theo độ gần). Đáp án càng gần đúng thì càng nhiều điểm: đoán 9 khi đáp án là 10 thì tốt hơn đoán 3. Giá trị ngoại lai bị phạt.

::: info Tự thiết kế reward
Docs gợi ý bạn đưa các câu trả lời của model cho một LLM khác (vd ChatGPT 4o hoặc Llama 3.1 8B) và nhờ nó thiết kế reward function hoặc verifier. Quy tắc có thể có dạng "nếu câu trả lời nghe quá máy móc, trừ 3 điểm".
:::

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo
