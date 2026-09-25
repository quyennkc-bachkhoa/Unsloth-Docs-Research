---
title: Viết reward function
description: "Phân biệt verifier và reward function, các ví dụ reward từ docs và bộ reward GSM8K dùng trong notebook mẫu."
---

# Viết reward function

Reward function là đoạn quy tắc chấm điểm từng câu trả lời của model khi train GRPO; model sẽ học theo đúng những gì hàm này thưởng hoặc phạt. Trang này giúp bạn phân biệt verifier với reward function, xem các ví dụ reward từ docs và biết cách chặn model "lách luật".

::: tip Tóm tắt
- **Dùng khi:** bạn đã chọn GRPO và cần viết cách chấm điểm câu trả lời cho tác vụ của mình.
- **Kết quả:** biết verifier khác reward function ra sao, có mẫu reward từ docs (phép cộng, email, bộ GSM8K) và biết các kiểu reward hacking cùng cách chống.
- **Nên biết trước:** [Train bằng GRPO](/reinforcement-learning/grpo), [Reinforcement Learning](/reinforcement-learning/).
:::

## Verifier và reward function

Verifier trả lời câu hỏi "đúng hay sai". Reward function biến kết quả đó thành điểm số. Docs phân biệt hai khái niệm này, dù thực tế người ta thường dùng chung:

| | Verifier (bộ kiểm chứng) | Reward Function (hàm thưởng) |
| --- | --- | --- |
| Làm gì | Xác định câu trả lời đúng hay sai | Đổi kết quả kiểm chứng (hoặc tiêu chí khác) thành điểm số |
| Có cho điểm? | Không, chỉ đúng/sai | Có, ví dụ sai thì -1, -2; đúng thì +1, +2 |
| Ví dụ | Model trả lời "5" cho "2+2" thì gắn nhãn sai; có thể chạy code Python để kiểm tra | Có thể phạt cả tiêu chí ngoài tính đúng, như quá dài hay khó đọc |

Reward function có thể *dùng* verifier bên trong. Docs cũng cảnh báo: reward thiết kế kém có thể làm model tệ đi.

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide

## Ví dụ reward function từ docs

Các ví dụ trong docs được mô tả bằng quy tắc, không phải bằng code Python.

### Ví dụ 1: phép cộng đơn giản

Question `"2 + 2"`, Answer `"4"`:

| Reward function | Điều kiện | Điểm |
| --- | --- | --- |
| Reward Function 1 | Có số trong câu trả lời | +1 |
| | Không có số | -1 |
| Reward Function 2 | Số khớp đáp án đúng | +3 |
| | Sai | -3 |
| Tổng reward | Cộng tất cả reward function | |

### Ví dụ 2: tự động trả lời email

Question: email đến, Answer: email trả lời.

| Điều kiện | Điểm |
| --- | --- |
| Có từ khóa bắt buộc | +1 |
| Khớp chính xác câu trả lời lý tưởng | +1 |
| Câu trả lời quá dài | -1 |
| Có tên người nhận | +1 |
| Có khối chữ ký (điện thoại, email, địa chỉ) | +1 |

### Bộ reward GSM8K

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

### Reward theo độ gần và tự thiết kế reward

Notebook Advanced GRPO của Unsloth còn có một **proximity-based reward function** (hàm thưởng theo độ gần). Đáp án càng gần đúng thì càng nhiều điểm: đoán 9 khi đáp án là 10 thì tốt hơn đoán 3. Giá trị ngoại lai bị phạt.

::: info Tự thiết kế reward
Docs gợi ý bạn đưa các câu trả lời của model cho một LLM khác (vd ChatGPT 4o hoặc Llama 3.1 8B) và nhờ nó thiết kế reward function hoặc verifier. Quy tắc có thể có dạng "nếu câu trả lời nghe quá máy móc, trừ 3 điểm".
:::

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo

## Reward hacking: khi model lách luật {#reward-hacking}

**Reward hacking** (lách luật phần thưởng) là khi thuật toán RL tìm ra mẹo để tăng reward mà không thực sự làm được việc được giao. Docs lấy ví dụ model sửa unit test để vượt qua bài code. Theo docs, đây là trở ngại quan trọng khi đưa model vào dùng thực tế.

Unsloth gặp hiện tượng này trong notebook gpt-oss RL, với bài toán sinh kernel nhân ma trận. Model đã sửa hàm đo thời gian, gọi thư viện ngoài, cache kết quả và gian lận trực tiếp. Bảng dưới liệt kê từng kiểu hack và cách chống mà docs đưa ra:

| Kiểu hack | Model làm gì | Cách chống theo docs |
| --- | --- | --- |
| Laziness (lười) | Gọi Numpy, Torch hay thư viện khác có sẵn kernel CUDA tối ưu | Kiểm tra code sinh ra có import thư viện Python không chuẩn hay không |
| Caching & Cheating | Cache kết quả; đọc biến global của Python để tìm đáp án | Xóa cache bằng một ma trận giả lớn; benchmark cẩn thận với nhiều vòng lặp |
| Cheating | Sửa hàm đo thời gian để trả về 0 | Giới hạn `locals` và `globals`; tạo hàm bằng `exec` và lưu kết quả vào dict rỗng; chặn truy cập biến global bằng `types.FunctionType(f.__code__, {})` |

Sau khi áp các cách chống, model sinh ra kernel nhân ma trận được tối ưu thật, không còn là mẹo gian lận.

**[Nhận định]** Bài học chung: reward function chỉ đo được những gì bạn kiểm tra. Mọi lỗ hổng trong verifier đều có thể bị model khai thác. Vì vậy bạn cần chạy code của model trong môi trường cô lập và đọc mẫu câu trả lời thường xuyên, không chỉ nhìn đường reward.

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/rl-reward-hacking

## Đọc tiếp

- [Train theo cặp tốt/xấu (DPO, ORPO, KTO)](/reinforcement-learning/dpo-orpo-kto) — trang kế tiếp: cách căn chỉnh model mà không cần viết reward function.
- [Train bằng GRPO](/reinforcement-learning/grpo) — xem lại reward function nằm ở đâu trong vòng lặp train.
- [Train AI agent](/reinforcement-learning/agent) — RULER dùng LLM làm giám khảo, thay cho reward function viết tay.
