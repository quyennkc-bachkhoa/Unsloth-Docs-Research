---
title: Suy luận & sampling
description: Model sinh token thế nào (prefill, decode, KV cache), các tham số sampling temperature/top-p/top-k/min-p/penalty và giá trị Unsloth khuyến nghị, chat template, thinking mode và cơ chế tool calling.
---

# Suy luận & sampling

Trang này giải thích điều gì xảy ra khi bạn chạy một model (inference, suy luận). Bạn sẽ thấy ba chuyện:

- Model sinh từng token ra sao.
- Các tham số `temperature`, `top_p`, `top_k`, `min_p`... điều khiển việc chọn token thế nào.
- Vì sao sai chat template làm model trả lời hỏng.

Nên đọc trang này trước khi chỉnh tham số trong Unsloth Studio, `llama-server` hay trang [Inference & API](/inference/).

## Sinh token: autoregressive, prefill và decode

Model không viết cả câu trả lời một lần. Nó đoán từng token một, và mỗi lần trả lời gồm hai pha có tốc độ rất khác nhau.

**Khái niệm.** LLM sinh văn bản kiểu autoregressive (tự hồi quy). Mỗi lần, model chỉ đoán **một** token tiếp theo. Token đó được nối vào chuỗi, rồi model đoán tiếp dựa trên toàn bộ chuỗi mới. Quá trình dừng khi model sinh token kết thúc (EOS, end-of-sequence) hoặc khi chạm giới hạn số token. Một lần trả lời chia thành hai pha:

- **Prefill** (nạp prompt): model xử lý toàn bộ prompt cùng lúc, song song. Mục đích là tính trạng thái trung gian (key và value) cho mọi token trong prompt.
- **Decode** (giải mã): model sinh từng token một. Mỗi bước chỉ xử lý một token mới, nên GPU không dùng hết sức tính. Tốc độ lúc này bị giới hạn bởi tốc độ đọc trọng số và KV cache từ bộ nhớ.

**[Nguồn ngoài]** Theo Transformers, LLM được train để sinh token tiếp theo dựa trên prompt và các token nó đã sinh. Việc sinh kéo dài tới độ dài định trước hoặc tới token EOS: https://huggingface.co/docs/transformers/llm_tutorial. NVIDIA mô tả prefill là phép nhân ma trận-ma trận song song, "bão hòa" GPU. Decode là phép ma trận-vector, bị giới hạn bởi băng thông bộ nhớ: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

**Ví dụ.** Bạn hỏi "Thủ đô của Pháp là gì?" (khoảng chục token). Prefill xử lý cả chục token trong một lượt. Sau đó decode chạy từng bước: "Thủ" → "đô" → "của" → ... → "Paris" → "." → EOS. Prompt dài làm bạn chờ lâu hơn trước khi thấy token đầu tiên. Câu trả lời dài làm tăng số bước decode.

**Ảnh hưởng khi dùng Unsloth.** Trang Qwen3.8 ghi tốc độ "generation" khoảng 20 tokens/s trên B200 nếu model vừa bộ nhớ. Nếu RAM + VRAM nhỏ hơn kích thước quant, model vẫn chạy nhưng chậm hơn nhiều do offload xuống ổ đĩa. **[Nhận định]** Con số "generation" đó là tốc độ pha decode. Decode bị giới hạn bởi băng thông bộ nhớ, nên offload làm chậm mạnh. Xem [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho).

**Gặp ở đâu trong Unsloth.** [Inference & API](/inference/), [Model catalog](/model-catalog).

**Nguồn:** https://unsloth.ai/docs/models/qwen3.8; **[Nguồn ngoài]** https://huggingface.co/docs/transformers/llm_tutorial, https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

## KV cache

KV cache là chỗ model ghi nhớ phần đã xử lý để khỏi tính lại ở mỗi bước. Nó giúp sinh nhanh hơn, nhưng chiếm thêm bộ nhớ và càng lớn khi context càng dài.

**Khái niệm.** Trong lớp attention, mỗi token tạo ra một vector key (K) và một vector value (V). Muốn đoán token thứ 1001, model cần K, V của cả 1000 token trước. Không có cache thì mỗi bước phải tính lại K, V cho toàn bộ chuỗi. KV cache lưu K, V của các token đã xử lý, cho từng layer. Nhờ vậy bước sau chỉ cần tính cho token mới rồi nối vào. Đổi lại, cache chiếm bộ nhớ tăng tuyến tính theo độ dài context.

**[Nguồn ngoài]** Transformers giải thích: KV cache lưu cặp key-value của token đã xử lý để dùng lại, tránh tính lại. Không có cache, chi phí attention mỗi bước tăng theo bình phương độ dài chuỗi. Có cache thì chi phí tăng tuyến tính, và bộ nhớ cũng tăng tuyến tính: https://huggingface.co/docs/transformers/cache_explanation. Công thức NVIDIA:

```text
KV cache (byte) = batch_size × số_token × 2 (K và V) × số_layer × hidden_size × số_byte_mỗi_số
```

Nguồn: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

**Ví dụ.** Docs Unsloth tính cho Llama 3.1 8B: 32 layer, K và V mỗi cái kích thước 1024, lưu 16-bit. Kết quả là `2 × 2 byte × 32 layer × 20K context × 1024 = 2.5GB` cho mỗi batch. Nếu batch của vLLM là 8 thì cần khoảng 20GB. Cùng model, context gấp đôi thì KV cache cũng gấp đôi.

**Ảnh hưởng khi dùng Unsloth.**

- **Chừa chỗ cho context.** Kích thước file GGUF không tính KV cache. Docs Unsloth nhắc nhiều lần phải chừa thêm bộ nhớ cho context. Ví dụ trang DeepSeek-V4: quant `UD-IQ3_XXS` nặng 103GB thì nên có ít nhất 110GB RAM.
- **Lượng tử hóa KV cache.** llama.cpp cho chọn kiểu lưu cache qua `--cache-type-k` và `--cache-type-v`. Các giá trị hỗ trợ: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`. Mặc định là `f16`. Theo docs Unsloth, `q4_1` (khoảng 5 bit) cho context dài hơn khoảng 3.2 lần. Với Qwen3.5, docs gợi ý thử `--cache-type-k bf16 --cache-type-v bf16` nếu gặp output vô nghĩa.
- **Giữ phần đầu prompt ổn định.** Cache chỉ dùng lại được khi phần đầu prompt giống hệt lần trước. Docs Unsloth ghi Claude Code chèn vào đầu system prompt một header thay đổi theo mỗi request. Header này làm KV cache mất hiệu lực, và inference với model local chậm đi khoảng 90%.

**Gặp ở đâu trong Unsloth.** [Inference & API](/inference/), [Memory-efficient RL](/reinforcement-learning/memory-efficient) (bảng bộ nhớ GRPO), [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho), [Token & context](/kien-thuc-nen/token-va-context).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/models/glm-5.3, https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/deepseek-v4, https://unsloth.ai/docs/basics/claude-code; **[Nguồn ngoài]** https://huggingface.co/docs/transformers/cache_explanation, https://huggingface.co/docs/transformers/kv_cache, https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

## Từ logits đến token: softmax và sampling

Ở mỗi bước, model không đưa ra một token duy nhất mà cho điểm mọi token có thể. Các tham số sampling quyết định cách biến bảng điểm đó thành lựa chọn cuối cùng.

**Khái niệm.** Ở mỗi bước decode, model trả về một **logit** cho mỗi token trong vocabulary (bộ từ vựng, thường hàng chục nghìn token). Logit là điểm thô, có thể âm. **Softmax** đổi các logit thành xác suất, cộng lại bằng 1. Sau đó một **sampler** (bộ chọn) quyết định lấy token nào. Các tham số `temperature`, `top_k`, `top_p`, `min_p` và penalty đều là cách chỉnh hoặc cắt bớt phân phối này trước khi chọn.

```text
xác_suất_i = exp(logit_i / T) / tổng_j exp(logit_j / T)      (T = temperature; T = 1 là softmax thường)
```

**[Nguồn ngoài]** Công thức temperature chia logit cho `t` trước softmax: https://arxiv.org/abs/1904.09751

**Ví dụ.** **[Ước tính]** Giả sử sau "Thủ đô của Pháp là" chỉ có 4 token ứng viên, với logit: Paris 5,0; Lyon 3,0; Nice 2,0; "một" 1,0. Softmax (T = 1) cho xác suất khoảng: Paris 0,831; Lyon 0,113; Nice 0,041; "một" 0,015. Các mục dưới dùng lại ví dụ này.

**Gặp ở đâu trong Unsloth.** [Inference & API](/inference/studio-chat) (mục Tham số sampling).

**Nguồn:** **[Nguồn ngoài]** https://arxiv.org/abs/1904.09751, https://huggingface.co/docs/transformers/generation_strategies

### Greedy và sampling

**Khái niệm.** Có hai cách chọn token cơ bản:

- **Greedy** (tham lam): luôn chọn token có xác suất cao nhất. Cùng prompt thì luôn ra cùng câu.
- **Sampling** (lấy mẫu): bốc ngẫu nhiên theo xác suất. Token nào có xác suất khác 0 cũng có cơ hội được chọn.

**[Nguồn ngoài]** Theo Transformers, greedy là chiến lược mặc định của `generate()`. Nó hợp với câu trả lời ngắn, không cần sáng tạo, nhưng "bắt đầu lặp lại" khi sinh chuỗi dài. Sampling giảm lặp và đa dạng hơn, bật bằng `do_sample=True`: https://huggingface.co/docs/transformers/generation_strategies. Paper nucleus sampling cho thấy với greedy hoặc beam search, xác suất lặp lại một cụm tăng dần sau mỗi lần lặp, tạo thành vòng lặp tự củng cố: https://arxiv.org/abs/1904.09751

**Ví dụ.** Với ví dụ trên, greedy luôn ra "Paris". Sampling ra "Paris" khoảng 83% số lần, "Lyon" khoảng 11%.

**Ảnh hưởng khi dùng Unsloth.** Các trang model của Unsloth đã xem đều khuyến nghị sampling với temperature từ 0.6 đến 1.0. Riêng hướng dẫn tool calling dùng 0.15 cho Devstral 2. Không trang nào khuyến nghị greedy. **[Nhận định]** Greedy (hoặc temperature rất thấp) với model nhỏ dễ gây lặp vô hạn. Nếu thấy lặp, hãy kiểm tra chat template trước (xem mục Chat template), rồi mới tới tham số sampling.

### Temperature

**Khái niệm.** Temperature chia logit trước softmax để chỉnh độ "mạnh tay" khi chọn:

- T nhỏ hơn 1 làm phân phối "nhọn" hơn: token mạnh càng mạnh.
- T lớn hơn 1 làm phân phối "phẳng" hơn: token yếu có thêm cơ hội.
- T tiến về 0 thì gần như greedy.

**Ví dụ.** **[Ước tính]** Cùng 4 logit:

| Temperature | Paris | Lyon | Nice | "một" |
| --- | --- | --- | --- | --- |
| 0.5 | 0,979 | 0,018 | 0,002 | 0,000 |
| 1.0 | 0,831 | 0,113 | 0,041 | 0,015 |
| 2.0 | 0,579 | 0,213 | 0,129 | 0,078 |

**Ảnh hưởng khi dùng Unsloth.**

- Mặc định của `llama-server` là `--temp 0.80`.
- Docs Unsloth khuyến nghị giá trị riêng cho từng model (bảng ở mục dưới). Ví dụ Qwen3.5 thinking mode dùng 1.0 cho tác vụ chung và 0.6 cho code chính xác.
- Trang API của Unsloth ghi: temperature thấp thường cho output ổn định hơn.
- Với GRPO, tài liệu nâng cao của Unsloth khuyên temperature khá cao (1.0) khi sinh câu trả lời, để các câu trong nhóm đa dạng (xem [RL & preference](/kien-thuc-nen/rl-va-preference)).

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation; **[Nguồn ngoài]** https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md, https://huggingface.co/docs/transformers/llm_tutorial

### Top-k

**Khái niệm.** Top-k chỉ giữ `k` token có xác suất cao nhất và bỏ phần còn lại. Xác suất của các token được giữ được chia lại, rồi mới bốc. `k` cố định, bất kể phân phối nhọn hay phẳng.

**Ví dụ.** **[Ước tính]** `top_k = 2` giữ Paris và Lyon. Sau khi chia lại: Paris ≈ 0,881, Lyon ≈ 0,119. **[Nguồn ngoài]** Paper nucleus sampling chỉ ra nhược điểm của `k` cố định. `k` nhỏ dễ cho văn nhạt ở ngữ cảnh "phẳng". `k` lớn lại để lọt token không phù hợp ở ngữ cảnh "nhọn": https://arxiv.org/abs/1904.09751

**Ảnh hưởng khi dùng Unsloth.** Mặc định của `llama-server` là `--top-k 40`; giá trị `0` là tắt. Docs Unsloth dùng 20 cho Qwen, 64 cho Gemma 4, và 0 (tắt) cho gpt-oss. Ví dụ tool calling của Unsloth truyền `top_k = -1`.

### Top-p (nucleus sampling)

**Khái niệm.** Top-p sắp token theo xác suất giảm dần rồi cộng dồn cho tới khi tổng đạt `p`. Chỉ nhóm token đó được giữ lại (gọi là "nucleus", hạt nhân). Khác top-k, số token giữ lại tự co giãn: phân phối nhọn thì giữ ít, phẳng thì giữ nhiều.

**[Nguồn ngoài]** Paper gốc định nghĩa top-p là tập nhỏ nhất có tổng xác suất ≥ p. Sau đó xác suất được chuẩn hóa lại và lấy mẫu trong tập đó: https://arxiv.org/abs/1904.09751

**Ví dụ.** **[Ước tính]**

- `top_p = 0.9`: Paris 0,831 chưa đủ. Cộng Lyon được 0,944 ≥ 0,9 → giữ 2 token.
- `top_p = 0.95`: 0,944 vẫn chưa đủ. Cộng Nice được 0,985 → giữ 3 token.
- `top_p = 1.0` là không cắt.

**Ảnh hưởng khi dùng Unsloth.** Mặc định của `llama-server` là `--top-p 0.95`; `1.0` là tắt. Trong docs Unsloth, đa số model dùng 0.95. Qwen non-thinking dùng 0.8. gpt-oss dùng 1.0.

### Min-p

**Khái niệm.** Min-p bỏ mọi token có xác suất nhỏ hơn `min_p × xác_suất_của_token_mạnh_nhất`. Ngưỡng này tỉ lệ theo token mạnh nhất. Khi model rất chắc chắn, ngưỡng cao nên cắt mạnh. Khi model phân vân, ngưỡng thấp nên giữ nhiều lựa chọn hơn.

**[Nguồn ngoài]** README `llama-server` định nghĩa min_p là "xác suất tối thiểu để một token được xem xét, tương đối so với xác suất của token có khả năng nhất". Mặc định là `0.05`; `0.0` là tắt: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md. Mã nguồn tính ngưỡng đúng theo `p_i >= p × p_max`: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp

**Ví dụ.** **[Ước tính]** `min_p = 0.1`: ngưỡng = 0,1 × 0,831 ≈ 0,083 → chỉ Paris (0,831) và Lyon (0,113) qua được.

**Ảnh hưởng khi dùng Unsloth.** Docs Unsloth đặt `min_p = 0.0` (tắt) cho Qwen3.5, Qwen3.8 và trong lệnh gpt-oss 120B. Trang Gemma 4 không ghi `min_p`. **[Nhận định]** Vì `llama-server` mặc định `0.05`, nếu bạn không truyền `--min-p 0.0` thì bạn đang chạy khác khuyến nghị của Unsloth.

### Repetition penalty và presence penalty

**Khái niệm.** Đây là hai cách phạt token đã xuất hiện, để model bớt lặp:

- **Repetition penalty** (`repeat_penalty` hoặc `repetition_penalty`): áp cho token đã xuất hiện trong `N` token gần nhất. Logit dương bị **chia** cho penalty, logit âm bị **nhân** với penalty. Giá trị 1.0 là tắt, lớn hơn 1 là phạt.
- **Presence penalty**: **trừ** một lượng cố định vào logit của mọi token đã xuất hiện ít nhất một lần, bất kể xuất hiện bao nhiêu lần. Giá trị 0.0 là tắt. Một biến thể gần giống là frequency penalty: nó trừ theo số lần xuất hiện.

**[Nguồn ngoài]** Cách tính lấy từ mã nguồn llama.cpp (`logit /= penalty_repeat` hoặc `*=` nếu logit âm; `logit -= count × penalty_freq + (count > 0) × penalty_present`): https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp. Transformers khuyên đặt `repetition_penalty` lớn hơn 1.0 nếu model hay lặp: https://huggingface.co/docs/transformers/llm_tutorial

**Ví dụ.** **[Ước tính]** "Lyon" đã xuất hiện trước đó, logit 3,0.

- `repeat_penalty = 1.1` → 3,0 / 1,1 ≈ 2,73.
- `presence_penalty = 1.5` → 3,0 - 1,5 = 1,5. Mức giảm mạnh hơn nhiều.

**Ảnh hưởng khi dùng Unsloth.**

- Mặc định của `llama-server` (CLI): `--repeat-penalty 1.00`, `--repeat-last-n 64`, `--presence-penalty 0.00`, `--frequency-penalty 0.00`.
- Docs Unsloth đặt `repetition_penalty = 1.0` (tắt) cho Qwen3.5 và Qwen3.8, và `presence_penalty = 1.5` cho một số chế độ Qwen. Trang Qwen3.5 cảnh báo giá trị `presence_penalty` cao có thể làm giảm nhẹ chất lượng.
- Trang API của Unsloth có ví dụ `--min-p 0.05 --repeat-penalty 1.1` cho Qwen3-1.7B. Đây là ví dụ minh họa cờ, không phải khuyến nghị riêng cho model.

::: warning README llama-server chưa thống nhất
Đây không phải docs Unsloth. Nhưng nó ảnh hưởng trực tiếp khi bạn chạy `llama-server` mà không truyền cờ, vì hai chỗ trong README ghi mặc định khác nhau:

- Bảng tham số dòng lệnh ghi `--repeat-penalty N` mặc định `1.00` (tắt).
- Mục tham số của API `/completion` ghi `repeat_penalty` mặc định `1.1`.

Nguồn: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md. Cách an toàn: luôn truyền rõ giá trị.
:::

### Thứ tự các sampler

**[Nguồn ngoài]** `llama-server` áp dụng các sampler theo thứ tự mặc định `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`. Bạn đổi thứ tự bằng `--samplers`: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md. **[Nhận định]** Nghĩa là trong llama.cpp mặc định:

- Penalty được áp trước tiên.
- Các bộ lọc top-k, top-p, min-p cắt trên phân phối chưa qua temperature.
- Temperature áp cuối cùng.

Engine khác (Transformers, vLLM) có thể theo thứ tự khác, nên cùng một bộ số chưa chắc cho kết quả y hệt.

### Khuyến nghị của Unsloth theo model

| Model / chế độ | temperature | top_p | top_k | min_p | Penalty | Nguồn |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5, thinking, tác vụ chung | 1.0 | 0.95 | 20 | 0.0 | presence 1.5; repeat tắt hoặc 1.0 | [qwen3.5](https://unsloth.ai/docs/models/qwen3.5) |
| Qwen3.5, thinking, code chính xác | 0.6 | 0.95 | 20 | 0.0 | presence 0.0; repeat tắt hoặc 1.0 | [qwen3.5](https://unsloth.ai/docs/models/qwen3.5) |
| Qwen3.5, non-thinking, tác vụ chung | 0.7 | 0.8 | 20 | 0.0 | presence 1.5; repeat tắt hoặc 1.0 | [qwen3.5](https://unsloth.ai/docs/models/qwen3.5) |
| Qwen3.5, non-thinking, tác vụ suy luận | 1.0 | 0.95 | 20 | 0.0 | presence 1.5; repeat tắt hoặc 1.0 | [qwen3.5](https://unsloth.ai/docs/models/qwen3.5) |
| Qwen3.8-27B, thinking | 1.0 | 0.95 | 20 | 0.0 | presence 0.0; repetition 1.0 | [qwen3.8](https://unsloth.ai/docs/models/qwen3.8) |
| Qwen3.8-27B, non-thinking | 0.7 | 0.80 | 20 | 0.0 | presence 1.5; repetition 1.0 | [qwen3.8](https://unsloth.ai/docs/models/qwen3.8) |
| Qwen3.8-2.4T (chỉ thinking) | 1.0 | 0.95 | 20 | 0.0 | presence 0.0; repetition 1.0 | [qwen3.8](https://unsloth.ai/docs/models/qwen3.8) |
| Gemma 4 | 1.0 | 0.95 | 64 | Docs không ghi | Docs không ghi | [gemma-4](https://unsloth.ai/docs/models/gemma-4) |
| gpt-oss (20B, 120B) | 1.0 | 1.0 | 0 (hoặc thử 100) | Mục khuyến nghị không ghi; lệnh 120B dùng 0.0 | Docs không ghi | [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) |

Với model khác, docs Unsloth nhắc "khi đổi model, nhớ dùng đúng tham số sampling" và trỏ tới trang hướng dẫn của từng model. Unsloth Studio tự đặt sẵn temperature, top-p, top-k cho model mới như Qwen3.5.

::: warning Docs chưa thống nhất
Cùng một trang [qwen3.5](https://unsloth.ai/docs/models/qwen3.5), ba chỗ nói khác nhau về `presence_penalty` của Qwen3.5:

- Gợi ý chung ghi `presence_penalty = 0.0 to 2.0`, "default this is off". Chỉ bật khi muốn giảm lặp, và giá trị cao có thể giảm nhẹ chất lượng.
- Bảng "Thinking mode, General tasks" và cả hai bảng non-thinking ghi `presence_penalty = 1.5`.
- Các lệnh `llama-cli` và `llama-server` mẫu cho đúng các chế độ này (ví dụ `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00`) không truyền presence penalty. Tức là chúng chạy với mặc định 0.0 của llama-server.
:::

**Gặp ở đâu trong Unsloth.** [Inference & API](/inference/), [Model catalog](/model-catalog).

**Nguồn:** https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/qwen3.8, https://unsloth.ai/docs/models/gemma-4, https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms, https://unsloth.ai/docs/new/studio/chat; **[Nguồn ngoài]** https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md, https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp, https://arxiv.org/abs/1904.09751

## Chat template

Chat template quyết định hội thoại của bạn được viết thành chuỗi văn bản thế nào trước khi đưa vào model. Dùng sai template là nguyên nhân phổ biến nhất khiến model trả lời hỏng.

**Khái niệm.** Về bản chất, chat model vẫn chỉ nối tiếp một chuỗi token. Danh sách tin nhắn `{"role": ..., "content": ...}` phải được đổi thành một chuỗi văn bản. Chuỗi này có token điều khiển để đánh dấu ai đang nói, tin nhắn bắt đầu ở đâu và kết thúc ở đâu. Chat template (mẫu hội thoại, thường viết bằng Jinja) là quy tắc đổi đó. Mỗi họ model được train với một template riêng, và model chỉ "hiểu" đúng template nó đã học.

**Ví dụ.** Cùng một câu "Hi", hai định dạng khác nhau. **[Nhận định]** Hai chuỗi dưới đây là minh họa người viết dựng lại từ các token điều khiển của ChatML và Harmony. Mẫu đầy đủ nằm ở [Chat Templates](https://unsloth.ai/docs/basics/chat-templates) và [trang gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune):

```text
ChatML (Qwen...):   <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n
gpt-oss (Harmony):  <|start|>user<|message|>Hi<|end|><|start|>assistant
```

**[Nguồn ngoài]** Tài liệu Transformers nêu ba điểm:

- Mistral-7B-Instruct dùng `[INST]`/`[/INST]`, còn Zephyr dùng `<|user|>`/`<|assistant|>`, dù cùng gốc Mistral-7B. Dùng sai token điều khiển thì model "kém đi đáng kể".
- `add_generation_prompt=True` thêm phần mở đầu lượt của assistant ở cuối chuỗi. Thiếu nó, model có thể viết tiếp tin nhắn của user thay vì trả lời.
- Một số tokenizer tự thêm `bos`/`eos`. Nếu template đã có sẵn mà bạn tokenize lại với special token thì token bị lặp, làm model kém đi: https://huggingface.co/docs/transformers/chat_templating

### Vì sao dùng sai template làm model trả lời lỗi

Trang Troubleshooting Inference của Unsloth bàn về một tình huống cụ thể. Model chạy tốt trong Unsloth, nhưng sau khi export sang Ollama, vLLM, llama.cpp lại ra chữ vô nghĩa, sinh **không dừng**, hoặc **lặp lại**. Trang này nêu các nguyên nhân và cách sửa:

- Nguyên nhân phổ biến nhất là **sai chat template**. Khi chạy ở framework khác, bạn phải dùng đúng template đã dùng lúc train trong Unsloth.
- Phải dùng đúng **EOS token**. Sai thì có thể ra chữ vô nghĩa khi sinh dài. **[Nhận định]** Model được dạy kết thúc lượt bằng một token cụ thể (ví dụ `<|im_end|>`). Nếu engine chờ một token khác thì không bao giờ thấy tín hiệu dừng, nên sinh mãi.
- Engine có thể thêm thừa, hoặc thiếu, token "start of sequence" (BOS). Cần kiểm tra cả hai khả năng.
- Cách sửa docs khuyên: dùng notebook conversational của Unsloth để "ép" chat template. Cách này sửa được đa số lỗi.

Docs Unsloth còn ghi hai trường hợp khác:

- **Lộ token lạ:** trang Studio Chat đo trên Qwen3.5-4B. Tool calling thông thường để lọt XML vào câu trả lời 10/10 lần, bản của Unsloth 0/10. Gemma 4 khi tắt thinking vẫn có thể in ra một khối thought rỗng `<|channel>thought<channel|>` trước câu trả lời.
- **Template của engine sai lệch:** với gpt-oss, Unsloth so template Jinja phổ biến với thư viện Harmony của OpenAI và thấy nhiều khác biệt. Tool call bị escape thừa dấu `\`, thừa dòng trống, và dùng sai kênh `final` thay vì `analysis`. Unsloth đã sửa template trong các bản upload của mình.

**Ảnh hưởng khi dùng Unsloth.**

- **Khi train:** dùng `get_chat_template(tokenizer, chat_template = "...")` để gắn template. Khi format dữ liệu, dùng `apply_chat_template(..., add_generation_prompt = False)`. `map_eos_token = True` ánh xạ `<|im_end|>` thành EOS mà không cần train thêm. Nếu dùng base model cho GRPO, docs yêu cầu phải có chat template.
- **Khi chạy `llama-server`:** `--jinja` bật engine Jinja cho chat template (mặc định bật theo README hiện tại). Tham số riêng của template truyền qua `--chat-template-kwargs '{"enable_thinking":false}'`.
- **[Nguồn ngoài]** TRL nhắc: với base model đã có template sẵn (ví dụ Qwen), phải căn chỉnh EOS token với chat template để câu trả lời kết thúc đúng: https://huggingface.co/docs/trl/sft_trainer

**Gặp ở đâu trong Unsloth.** [Dữ liệu](/du-lieu/), [Fine-tuning](/fine-tuning/), [Export & deploy](/export-deploy/), [Inference & API](/inference/).

**Nguồn:** https://unsloth.ai/docs/basics/chat-templates, https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference, https://unsloth.ai/docs/new/studio/chat, https://unsloth.ai/docs/models/gemma-4, https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide; **[Nguồn ngoài]** https://huggingface.co/docs/transformers/chat_templating, https://huggingface.co/docs/transformers/llm_tutorial, https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## Thinking / reasoning mode

Model thinking "nghĩ nháp" trước khi trả lời. Mỗi họ model bật, tắt và xử lý phần nháp này theo cách riêng, nên bạn cần làm theo trang của đúng model.

**Khái niệm.** Model "thinking" (hay reasoning) sinh ra một đoạn suy nghĩ nội bộ trước câu trả lời cuối. Đoạn này được bọc trong token riêng của từng họ model, để app có thể tách ra hoặc ẩn đi. Model "hybrid" bật hoặc tắt được thinking. Việc bật tắt thường đi qua chat template, nên đây cũng là chuyện template.

**Ví dụ.** Cách mỗi model xử lý thinking, theo docs Unsloth:

- **Qwen3.5:** hybrid. Tắt bằng `--chat-template-kwargs '{"enable_thinking":false}'`. Các bản nhỏ 0.8B, 2B, 4B, 9B tắt thinking mặc định; bật bằng `{"enable_thinking":true}`. Docs khuyên để độ dài output khoảng `32,768` token cho đa số câu hỏi.
- **Qwen3.8-27B:** có `reasoning_effort` với các mức `xhigh` (mặc định), `medium`, `low`, none. Đổi mức bằng `--chat-template-kwargs '{"reasoning_effort":"medium"}'`. Có thêm "Preserve Thinking" để giữ lại đoạn suy nghĩ của lượt trước. Tùy chọn này tốn thêm token nhưng có thể chính xác hơn trong hội thoại dài.
- **Gemma 4:** bật thinking bằng cách đặt token `<|think|>` ở đầu system prompt. Với hội thoại nhiều lượt, **chỉ giữ câu trả lời cuối** trong lịch sử, không đưa khối thought cũ vào lượt sau.
- **gpt-oss:** `reasoning_effort` có các mức low, medium, high. Mức cao chính xác hơn nhưng chậm hơn vì tốn nhiều token suy nghĩ.

**Ảnh hưởng khi dùng Unsloth.** Thinking và non-thinking thường có bộ tham số sampling khác nhau (bảng ở trên). Đoạn thinking cũng chiếm context và KV cache như token thường. **[Nhận định]** Qwen3.8 khuyến khích giữ thinking cũ, còn Gemma 4 yêu cầu bỏ; hãy làm theo trang của đúng model. **[Nguồn ngoài]** `llama-server` có `--reasoning-format` để tách phần suy nghĩ vào trường riêng (ví dụ `reasoning_content`), và `--reasoning-budget` để giới hạn số token suy nghĩ: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

**Gặp ở đâu trong Unsloth.** [Inference & API](/inference/), [Model catalog](/model-catalog).

**Nguồn:** https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/qwen3.8, https://unsloth.ai/docs/models/gemma-4, https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune; **[Nguồn ngoài]** https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## Tool calling

Tool calling cho model yêu cầu app của bạn chạy một hàm, thay vì tự đoán câu trả lời. Model chỉ viết ra yêu cầu; việc chạy hàm do app của bạn làm.

**Khái niệm.** Khi được phép gọi công cụ, LLM sinh ra một yêu cầu có cấu trúc: tên hàm cộng tham số JSON. Model **không tự chạy** hàm. Vòng xử lý diễn ra thế này:

1. App của bạn đọc yêu cầu và chạy hàm.
2. App đưa kết quả trở lại hội thoại dưới vai `tool`.
3. Model đọc kết quả rồi viết câu trả lời cuối, hoặc gọi thêm tool.

Danh sách tool được mô tả bằng JSON schema và được chat template chèn vào prompt.

<div class="dg">
<div class="dg-flow">
<div class="dg-node"><div><span class="dg-n">1</span>App gửi messages + tools</div><small>danh sách tools dạng JSON schema</small></div>
<div class="dg-node"><div><span class="dg-n">2</span>Chat template</div><small>chèn mô tả tools vào prompt</small></div>
<div class="dg-node is-main"><div><span class="dg-n">3</span>Model sinh token</div></div>
<div class="dg-node is-q"><div><span class="dg-n">4</span>Model trả về <code>tool_calls</code>?</div></div>
<div class="dg-node is-end" data-e="Không">Câu trả lời cuối cho người dùng</div>
</div>
<div class="dg-group" style="margin-top: 30px">
<span class="dg-glabel">Nếu bước 4 là “Có”</span>
<div class="dg-flow">
<div class="dg-node">App đọc tên hàm + tham số JSON</div>
<div class="dg-node">App tự chạy hàm<small>vd <code>add_number</code>, terminal, python</small></div>
<div class="dg-node">App thêm 2 message<small>role <code>assistant</code> (tool_calls) và role <code>tool</code> (kết quả)</small></div>
<div class="dg-node is-ghost">↺ Quay lại bước 2</div>
</div>
</div>
</div>

**[Nguồn ngoài]** Transformers nhấn mạnh model "không thể tự gọi tool", nó chỉ yêu cầu gọi. Việc của bạn là xử lý lời gọi, rồi thêm lời gọi và kết quả vào lịch sử chat. Kết quả nằm trong message role `tool` và luôn là chuỗi: https://huggingface.co/docs/transformers/chat_extras

**Ví dụ.** Hướng dẫn tool calling của Unsloth làm như sau:

- App định nghĩa các hàm `add_number`, `multiply_number`, `terminal`, `python`... kèm JSON schema.
- App gửi tới `llama-server` qua OpenAI SDK với `tools = tools` và `tool_choice = "auto"`.
- Khi response có `tool_calls`, app lấy `tool_call.function.name` và `json.loads(tool_call.function.arguments)`, rồi gọi hàm tương ứng trong `MAP_FN`.
- App thêm `{"role": "tool", "tool_call_id": ..., "name": ..., "content": str(out)}` vào `messages`.

Hàm `terminal` trong ví dụ tự chặn các lệnh chứa `rm`, `sudo`, `dd`, `chmod`.

**Ảnh hưởng khi dùng Unsloth.**

- **Phụ thuộc chat template:** template phải biết cách hiển thị tools và `tool_calls`. **[Nguồn ngoài]** README `llama-server` ghi function calling kiểu OpenAI được hỗ trợ với cờ `--jinja`, và có thể cần `--chat-template-file` để có template hỗ trợ tool: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md. Lệnh mẫu trong docs Unsloth đều có `--jinja`.
- **Sampling vẫn quan trọng:** docs Unsloth nhắc đổi model thì đổi tham số sampling theo trang của model đó. Ví dụ Devstral 2 dùng temperature 0.15; GLM-4.7 dùng 0.7 và top_p 1.0.
- **Studio Chat:** bật tool calling sẵn, có "self-healing" (tự sửa tool call hỏng), dừng vòng gọi tool ổn định hơn và chặn XML lọt ra output.
- **[Nhận định]** Không cho model gọi hàm nguy hiểm mà không kiểm soát. Ví dụ `terminal` và `python` trong docs chạy code thật trên máy bạn.

**Gặp ở đâu trong Unsloth.** [Inference & API](/inference/tool-calling) (mục Tool calling), [Ứng dụng & RAG](/ung-dung-rag).

**Nguồn:** https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms, https://unsloth.ai/docs/new/studio/chat; **[Nguồn ngoài]** https://huggingface.co/docs/transformers/chat_extras, https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## Gặp ở đâu trong Unsloth

| Khái niệm | Trang Unsloth trên website | Docs gốc |
| --- | --- | --- |
| Autoregressive, prefill, decode | [Inference & API](/inference/) | [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8) |
| KV cache, `--cache-type-k/v` | [Inference & API](/inference/), [Reinforcement Learning](/reinforcement-learning/) | [RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), [GLM-5.3](https://unsloth.ai/docs/models/glm-5.3), [Claude Code](https://unsloth.ai/docs/basics/claude-code) |
| Greedy vs sampling, logits, softmax | [Inference & API](/inference/) | [API](https://unsloth.ai/docs/basics/api) |
| temperature, top_p, top_k, min_p | [Inference & API](/inference/), [Model catalog](/model-catalog) | [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5), [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8), [Gemma 4](https://unsloth.ai/docs/models/gemma-4), [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) |
| repetition / presence penalty | [Inference & API](/inference/) | [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5), [API](https://unsloth.ai/docs/basics/api) |
| Chat template, EOS, BOS | [Dữ liệu — Chat template](/du-lieu/chat-template), [Export & deploy](/export-deploy/) | [Chat Templates](https://unsloth.ai/docs/basics/chat-templates), [Troubleshooting Inference](https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference) |
| Thinking / reasoning mode | [Inference & API](/inference/), [Model catalog](/model-catalog) | [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5), [Gemma 4](https://unsloth.ai/docs/models/gemma-4), [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) |
| Tool calling | [Inference & API](/inference/) | [Tool Calling Guide](https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms), [Studio Chat](https://unsloth.ai/docs/new/studio/chat) |
