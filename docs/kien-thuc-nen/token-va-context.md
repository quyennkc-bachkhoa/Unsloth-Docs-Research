---
title: Token & context
description: Token, tokenizer, vocabulary, special token (BOS/EOS), context window và cơ chế dự đoán token tiếp theo — những khái niệm nằm sau max_seq_length, chat template và max_new_tokens trong Unsloth.
---

# Token & context

Trang này giải thích hai thứ: token là đơn vị mà LLM thực sự "đọc" và "viết", còn context là giới hạn độ dài model xử lý được. Bạn nên đọc trang này trước khi chỉnh `max_seq_length`, chọn chat template, hoặc gỡ lỗi khi model sinh chữ vô nghĩa sau khi export.

```mermaid
flowchart LR
  A["Văn bản đầu vào"] --> B["Tokenizer tách thành token"]
  B --> C["Token ID (số nguyên)"]
  C --> D["Model dự đoán token tiếp theo"]
  D --> E{"Là EOS hoặc đủ max_new_tokens?"}
  E -- "Chưa" --> F["Nối token mới vào chuỗi"]
  F --> D
  E -- "Rồi" --> G["Tokenizer giải mã ID thành văn bản"]
```

## Token

**Khái niệm.** Token là mảnh văn bản nhỏ nhất mà model xử lý. Một token có thể là một từ, một phần của từ (subword) hoặc một dấu câu. Từ phổ biến thường là một token; từ hiếm bị tách thành nhiều mảnh. Model không nhìn thấy chữ cái. Nó chỉ thấy một dãy token đã được đổi thành số. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/glossary

**Ví dụ.** Tokenizer của BERT tách câu `A Titan RTX has 24GB of VRAM` thành `['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']`. Từ "VRAM" không có trong từ vựng nên bị cắt thành `V`, `##RA`, `##M`. Dấu `##` báo rằng mảnh này nối tiếp mảnh trước, cùng thuộc một từ. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/glossary

Để hình dung quy mô, docs Unsloth dùng một quy ước thô: "cứ coi một token như một từ tiếng Anh". Docs dùng quy ước này khi nói Llama-3 được huấn luyện trên 15 nghìn tỷ token. Đây chỉ là cách minh họa, không phải tỷ lệ chính xác.

**Ảnh hưởng khi dùng Unsloth.** Mọi giới hạn độ dài trong Unsloth đều đếm bằng **token**, không phải ký tự hay từ. Điều này áp dụng cho `max_seq_length`, `max_new_tokens`, `max_prompt_length` và "Context Length" trong Studio. Dataset cũng phải ở dạng tokenizer đọc được thì mới train được; docs Datasets Guide nói rõ điều này.

**Tiếng Việt tốn bao nhiêu token?** Các nguồn đã duyệt không đưa con số cho tiếng Việt. Nguồn HF chỉ nói: tokenizer huấn luyện trên tiếng Anh sẽ làm việc kém với ngôn ngữ có cách dùng khoảng trắng và dấu câu khác, ví dụ tiếng Nhật **[Nguồn ngoài]** https://huggingface.co/learn/llm-course/chapter6/1. **[Nhận định]** Tiếng Việt có dấu thanh và nhiều âm tiết, nên có thể tốn nhiều token hơn tiếng Anh cho cùng nội dung, tùy tokenizer. Đừng giả định; hãy tự đo bằng tokenizer của chính model bạn dùng. Cách đo: đối tượng `tokenizer` trả về từ `FastLanguageModel.from_pretrained(...)` có phương thức `tokenize()`, trả về danh sách token của một chuỗi, như mô tả trong Transformers glossary và LLM Course chương 2 **[Nguồn ngoài]** https://huggingface.co/docs/transformers/glossary , https://huggingface.co/learn/llm-course/chapter2/4 . Đưa vào cùng một đoạn văn bằng tiếng Việt và tiếng Anh, rồi so độ dài hai danh sách.

**Gặp ở đâu trong Unsloth.** [Dữ liệu](/du-lieu), [Fine-tuning](/fine-tuning), [Inference](/inference).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://huggingface.co/docs/transformers/glossary, https://huggingface.co/learn/llm-course/chapter6/1

## Tokenizer và các thuật toán tách token

**Khái niệm.** Tokenizer (bộ tách token) là thành phần đổi văn bản thành số và đổi số trở lại thành văn bản. Chiều đi gồm hai bước:

1. Tách văn bản thành token.
2. Tra từ vựng để đổi mỗi token thành một **input ID** (mã số nguyên).

Chiều ngược lại, từ ID về văn bản, gọi là decode (giải mã). **[Nguồn ngoài]** https://huggingface.co/learn/llm-course/chapter2/4

Có ba cách tách chính **[Nguồn ngoài]** https://huggingface.co/docs/transformers/tokenizer_summary:

| Cách tách | Ưu điểm | Nhược điểm |
|---|---|---|
| Theo từ (word-level) | Mỗi token mang nhiều nghĩa | Từ vựng khổng lồ; từ lạ thành `<unk>` (token "không biết") |
| Theo ký tự (character-level) | Từ vựng nhỏ, không có `<unk>` | Chuỗi rất dài, mỗi token ít nghĩa |
| Subword (mảnh từ) | Từ phổ biến giữ nguyên, từ hiếm tách mảnh; từ vựng gọn, gần như không có `<unk>` | Số token phụ thuộc dữ liệu tokenizer đã học |

Các LLM hiện nay dùng subword. Ba thuật toán chính, ở mức khái niệm **[Nguồn ngoài]** https://huggingface.co/docs/transformers/tokenizer_summary:

- **BPE** (Byte Pair Encoding): bắt đầu từ các ký tự. Thuật toán lặp lại việc gộp cặp token đứng cạnh nhau hay gặp nhất, cho tới khi từ vựng đủ lớn. Docs Transformers ghi Llama, Gemma, Qwen2 dùng BPE. **Byte-level BPE** lấy 256 giá trị byte làm từ vựng gốc, nên chuỗi nào cũng tách được mà không cần `<unk>`.
- **WordPiece**: giống BPE, nhưng chọn cặp để gộp theo mức "đi cùng nhau nhiều hơn ngẫu nhiên". Họ BERT dùng cách này.
- **Unigram**: đi theo chiều ngược lại. Bắt đầu từ một tập ứng viên lớn, rồi loại dần các token ít đóng góp. T5 dùng cách này.
- **SentencePiece** là thư viện chạy BPE hoặc Unigram trực tiếp trên văn bản thô. Nó coi khoảng trắng là ký hiệu `▁`, nên hợp với ngôn ngữ không tách từ bằng khoảng trắng.

**Ví dụ.** Số token của cùng một văn bản phụ thuộc vào tokenizer. Với cùng một hàm Python ngắn, tokenizer GPT-2 gốc tách thành 36 token. Tokenizer được huấn luyện lại trên code Python chỉ cần 27 token, vì nó đã học token riêng cho thụt lề và cho `"""`. **[Nguồn ngoài]** https://huggingface.co/learn/llm-course/chapter6/2

Docs Unsloth nêu thêm một chi tiết: tokenizer của Llama coi `"A"` và `" A"` (có khoảng trắng phía trước) là **hai token ID khác nhau**. Khi Unsloth tự chấm MMLU và tính cả hai dạng, điểm Llama 3.1 (8B) Instruct tăng từ 67,8% lên 68,2%.

**Ảnh hưởng khi dùng Unsloth.** Model và tokenizer luôn đi thành cặp. `FastLanguageModel.from_pretrained(...)` trả về cả `model, tokenizer`, và các hàm lưu (`save_pretrained_merged`, `save_pretrained_gguf`) đều nhận `tokenizer`. Docs Unsloth ghi các bản upload của họ đôi khi có sửa lỗi chat template hoặc tokenizer so với bản gốc. Vì vậy docs khuyên dùng bản của Unsloth khi có.

**Gặp ở đâu trong Unsloth.** [Model catalog](/model-catalog), [Export & deploy](/export-deploy).

**Nguồn:** https://huggingface.co/learn/llm-course/chapter2/4, https://huggingface.co/docs/transformers/tokenizer_summary, https://huggingface.co/learn/llm-course/chapter6/2, https://unsloth.ai/docs/basics/dynamic-3.0-ggufs, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

## Vocabulary

**Khái niệm.** Vocabulary (từ vựng) là danh sách cố định mọi token mà tokenizer biết. Mỗi token trong danh sách có một ID. Kích thước từ vựng được chọn lúc huấn luyện tokenizer. Từ vựng gắn với hai lớp trong model:

- ma trận embedding (lớp `embed_tokens`), có một hàng cho mỗi token;
- lớp đầu ra (`lm_head`), chấm điểm cho mọi token trong từ vựng.

**[Nguồn ngoài]** https://huggingface.co/docs/transformers/tokenizer_summary

**Ví dụ.**
- GPT-2 có từ vựng 50.257 token = 256 token byte + 50.000 lần gộp + 1 token đặc biệt kết thúc văn bản. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/tokenizer_summary
- Docs GRPO của Unsloth dùng từ vựng 128.256 token (của Llama) khi tính bộ nhớ logits: `2 × 2 byte × 8 (số lần sinh) × 20K (context) × 128256 (từ vựng) = 78,3 GB`. Từ vựng càng lớn thì bước tính xác suất cho từng token càng tốn bộ nhớ.

**Ảnh hưởng khi dùng Unsloth.**
- **Thêm token mới:** dùng `add_new_tokens(model, tokenizer, new_tokens = [...])`. Docs yêu cầu **phải gọi hàm này trước** `FastLanguageModel.get_peft_model`.
- **Dạy model ngôn ngữ mới bằng continued pretraining:** docs khuyên thêm `"lm_head", "embed_tokens"` vào `target_modules`, và đặt `embedding_learning_rate` nhỏ hơn `learning_rate` 2–10 lần. Nếu Colab hết bộ nhớ với Llama-3 8B thì chỉ thêm `lm_head`.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "lm_head", "embed_tokens",],
    lora_alpha = 16,
)
```

**Gặp ở đâu trong Unsloth.** [Dữ liệu — Thêm token mới](/du-lieu), [Fine-tuning — Continued pretraining](/fine-tuning), [Reinforcement learning](/reinforcement-learning). Chi tiết về embedding: [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer).

**Nguồn:** https://huggingface.co/docs/transformers/tokenizer_summary, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/basics/chat-templates, https://unsloth.ai/docs/basics/continued-pretraining

## Special token: BOS, EOS và token điều khiển hội thoại

**Khái niệm.** Special token (token đặc biệt) không đại diện cho chữ bình thường. Nó mang tín hiệu điều khiển cho model. Hai loại quan trọng nhất:
- **BOS** (beginning of sequence, đầu chuỗi), ví dụ `<s>`, `<bos>`.
- **EOS** (end of sequence, cuối chuỗi), ví dụ `</s>`. Khi model sinh ra EOS, quá trình sinh dừng lại.

Model chat còn có token đánh dấu vai trò người nói, ví dụ `<|im_start|>`, `<|im_end|>` (ChatML), `[INST]`/`[/INST]` (Mistral), `<start_of_turn>` (Gemma). Tokenizer tự chèn special token nếu model cần. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/glossary

**Liên hệ chat template.** Chat template là quy tắc biến danh sách tin nhắn `role`/`content` thành một chuỗi token, có chèn đúng các token điều khiển của model đó. Thực chất, model chat nào cũng chỉ "viết tiếp một chuỗi token". Token điều khiển cho nó biết đâu là lượt của người dùng, đâu là lượt của trợ lý. Hai model fine-tune từ cùng một base vẫn có thể dùng template khác nhau. Dùng sai token điều khiển làm chất lượng giảm mạnh. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/chat_templating

**Ví dụ.** Với cùng một hội thoại, Mistral-7B-Instruct cho ra `<s>[INST] Hello, how are you? [/INST]...</s>`, còn Zephyr-7B cho ra `<|user|>\nHello, how are you?</s>\n<|assistant|>...`. Khi đặt `add_generation_prompt=True`, template ChatML thêm `<|im_start|>assistant` vào cuối để model biết tới lượt nó trả lời. Thiếu phần này, model có thể viết tiếp câu của người dùng. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/chat_templating

**Ảnh hưởng khi dùng Unsloth.**
- **Gắn template:** `get_chat_template(tokenizer, chat_template = "...")` gắn template vào tokenizer. Tham số `map_eos_token = True` ánh xạ `<|im_end|>` thành EOS `</s>` mà không cần train. Nếu bạn tự viết template, phải truyền cặp `(template, eos_token)`, và EOS phải xuất hiện trong template.
- **Gỡ lỗi sau khi export:** chạy model đã export trên Ollama, llama.cpp hoặc vLLM mà ra chữ vô nghĩa, sinh mãi không dừng hoặc lặp lại? Docs Unsloth nêu các nguyên nhân sau:
  - **sai chat template** (nguyên nhân phổ biến nhất);
  - **sai EOS token**;
  - engine thêm thừa hoặc thiếu token "start of sequence" (BOS).

  Cách tránh: dùng cùng một template lúc train và lúc chạy.
- **[Nguồn ngoài]** Nếu đã format bằng `apply_chat_template(tokenize=False)` rồi mới tokenize, cần `add_special_tokens=False` để không chèn BOS/EOS hai lần. https://huggingface.co/docs/transformers/chat_templating

**Gặp ở đâu trong Unsloth.** [Dữ liệu — Chat template](/du-lieu), [Export & deploy — Troubleshooting](/export-deploy), [Inference](/inference).

**Nguồn:** https://unsloth.ai/docs/basics/chat-templates, https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference, https://huggingface.co/docs/transformers/chat_templating, https://huggingface.co/docs/transformers/glossary

## Context window và `max_seq_length`

**Khái niệm.** Context window (cửa sổ ngữ cảnh) là số token tối đa model xử lý trong một lần. Con số này tính gộp prompt, lịch sử hội thoại, tài liệu đính kèm **và** phần model đang sinh ra. Bạn cần tách hai con số:

1. **Context của model**: giới hạn do hãng công bố trong model card. Ví dụ Qwen3-8B hỗ trợ gốc 32.768 token, và mở rộng tới 131.072 token bằng YaRN (một kỹ thuật kéo giãn vị trí). **[Nguồn ngoài]** https://huggingface.co/Qwen/Qwen3-8B
2. **Context bạn đặt**: con số bạn chọn khi train hoặc chạy. Trong code Unsloth là `max_seq_length`, trong Studio là "Context Length", trong llama.cpp là `--ctx-size`. Con số này thường nhỏ hơn context của model để tiết kiệm bộ nhớ.

**Ví dụ (context model công bố trong docs Unsloth).**

| Model | Context theo docs Unsloth |
|---|---|
| Llama-3 | 8.192 |
| gpt-oss | 131.072 |
| IBM Granite 4.1 | 131.072 |
| Qwen3.5, Qwen3.8 | 262.144 (mở rộng tới 1M bằng YaRN) |
| Gemma 4 E2B / E4B | 128K |
| Gemma 4 12B / 26B-A4B / 31B | 256K |
| DeepSeek-V4, GLM-5.3, Kimi K3 | 1.048.576 |

Với model đa phương thức, ảnh cũng chiếm chỗ trong context. Model card Gemma 3 ghi: mỗi ảnh được chuẩn hóa về 896 × 896 và mã hóa thành 256 token; bản 4B có context đầu vào 128K, đầu ra tối đa 8.192 token. **[Nguồn ngoài]** https://huggingface.co/google/gemma-3-4b-it

**Ảnh hưởng khi dùng Unsloth.**
- **Khi train:** `max_seq_length` quyết định độ dài tối đa của mỗi mẫu. Docs khuyên đặt `2048` để thử nghiệm (Llama-3 hỗ trợ 8192). Studio mặc định Context Length `2048` và cho chọn từ 512 → 32768. Notebook ghi Unsloth hỗ trợ RoPE Scaling bên trong nên "chọn số nào cũng được". **[Nhận định]** Đặt vượt context gốc của model thì vẫn chạy được. Nhưng điều đó không có nghĩa model dùng tốt đoạn dài như vậy, nếu dữ liệu train không có mẫu dài tương tự.
- **Đặt quá cao** làm tốn VRAM và dễ OOM (hết bộ nhớ GPU). Trang Granite 4.1 khuyên giảm `max_seq_length` nếu hết bộ nhớ. Benchmark Unsloth cho thấy độ dài context tối đa phụ thuộc VRAM. Với Llama 3.1 (8B) QLoRA 4-bit, rank 32, batch 1, context đạt khoảng 40.724 token trên GPU 16 GB và 78.475 token trên GPU 24 GB. Liên hệ bộ nhớ: [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho).
- **Đặt quá thấp** thì mẫu dài bị cắt. Khi chạy, docs Qwen3.5 ghi: model ra chữ vô nghĩa có thể do context đặt quá thấp.
- **Khi chạy:**
  - `max_new_tokens` giới hạn số token được sinh ra. Docs gợi ý tăng từ 128 lên 256 hoặc 1024 nếu muốn câu trả lời dài hơn, đổi lại phải chờ lâu hơn.
  - Với GRPO, `max_completion_length = max_seq_length - max_prompt_length`. Nói cách khác, prompt và câu trả lời chia nhau cùng một ngân sách token.
  - Trong Studio Chat, docs ghi bạn không cần chỉnh context, vì llama.cpp tự dùng lượng context cần thiết.

::: tip Một phép tính nhanh
**[Ước tính]** Cài `max_seq_length = 2048` và prompt RAG đã chiếm 1.800 token thì model chỉ còn tối đa `2048 − 1800 = 248` token để trả lời. Phép tính này dựa trên quy tắc prompt + phần sinh không vượt quá context (https://unsloth.ai/docs/get-started/install/intel dùng đúng phép trừ này cho GRPO).
:::

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning), [Inference](/inference), [Model catalog](/model-catalog), [Reinforcement learning](/reinforcement-learning).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/get-started/install/google-colab, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/new/studio/chat, https://unsloth.ai/docs/basics/unsloth-benchmarks, https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/qwen3.8, https://unsloth.ai/docs/models/gemma-4, https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune, https://unsloth.ai/docs/models/ibm-granite-4.1, https://unsloth.ai/docs/models/deepseek-v4, https://unsloth.ai/docs/models/glm-5.3, https://unsloth.ai/docs/models/kimi-k3, https://unsloth.ai/docs/get-started/install/intel, https://huggingface.co/Qwen/Qwen3-8B, https://huggingface.co/google/gemma-3-4b-it

## Dự đoán token tiếp theo (next-token prediction)

**Khái niệm.** LLM sinh văn bản làm đúng một việc: nhìn các token đã có rồi đoán token kế tiếp.

- **Khi pre-train:** các LLM sinh văn bản (decoder model) học bằng causal language modeling. Model đọc văn bản theo thứ tự và đoán token kế tiếp. Một cơ chế che (mask) ngăn model nhìn thấy các token phía sau.
- **Khi chạy:** model lặp lại ba bước: đoán một token, nối token đó vào chuỗi, rồi đoán token kế tiếp. Cách sinh này gọi là autoregressive (tự hồi quy). Vòng lặp dừng khi model sinh ra EOS hoặc chạm giới hạn độ dài.

**[Nguồn ngoài]** https://huggingface.co/docs/transformers/glossary, https://huggingface.co/learn/llm-course/chapter1/4

**Ví dụ.** Một prompt ChatML kết thúc bằng `<|im_start|>assistant`. Model đoán token đầu tiên của câu trả lời. Sau đó nó đoán token thứ hai dựa trên toàn bộ chuỗi phía trước, và cứ thế tiếp tục. Khi token được đoán là `<|im_end|>` (đã được ánh xạ làm EOS), việc sinh dừng lại. Nếu engine không nhận ra EOS này, model sẽ viết tiếp mãi. Đây chính là lỗi "sinh vô hạn" mà docs troubleshooting của Unsloth nhắc tới.

**Ảnh hưởng khi dùng Unsloth.**
- **Fine-tuning (SFT)** vẫn dùng mục tiêu đoán token tiếp theo; chỉ có dữ liệu là thay đổi. Training loss trong log cho biết model đoán sai token trên dataset của bạn nhiều hay ít. Docs coi loss khoảng 0,5–1,0 là dấu hiệu tốt, tùy tác vụ. Loss về 0 có thể là overfitting (học thuộc).
- **`train_on_responses_only`** (trong Studio là "Train on Completions") chỉ tính loss trên token của câu trả lời, bỏ qua phần câu hỏi. Bạn phải khai báo đúng chuỗi token điều khiển của template. Ví dụ với Llama 3:

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

- Cách chọn token ở mỗi bước (temperature, top-p, top-k) xem ở [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling). Quá trình train xem ở [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen).

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning), [Dữ liệu — Chỉ train trên câu trả lời](/du-lieu), [Inference](/inference).

**Nguồn:** https://huggingface.co/docs/transformers/glossary, https://huggingface.co/learn/llm-course/chapter1/4, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference, https://unsloth.ai/docs/new/studio/start

## Gặp ở đâu trong Unsloth

Bảng này gom lại các khái niệm trên trang, kèm trang Unsloth trên website và docs gốc nơi khái niệm đó xuất hiện.

| Khái niệm | Trang Unsloth trên website | Docs gốc |
|---|---|---|
| Token | [Dữ liệu](/du-lieu), [Fine-tuning](/fine-tuning) | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide |
| Tokenizer | [Model catalog](/model-catalog), [Export & deploy](/export-deploy) | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide |
| Vocabulary, `add_new_tokens`, `embed_tokens`/`lm_head` | [Dữ liệu](/du-lieu), [Fine-tuning](/fine-tuning) | https://unsloth.ai/docs/basics/chat-templates, https://unsloth.ai/docs/basics/continued-pretraining |
| Special token BOS/EOS, chat template | [Dữ liệu](/du-lieu), [Export & deploy](/export-deploy) | https://unsloth.ai/docs/basics/chat-templates, https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference |
| Context window, `max_seq_length`, `max_new_tokens` | [Fine-tuning](/fine-tuning), [Inference](/inference), [Model catalog](/model-catalog) | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/basics/unsloth-benchmarks |
| Next-token prediction, `train_on_responses_only` | [Fine-tuning](/fine-tuning), [Dữ liệu](/du-lieu) | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide |
