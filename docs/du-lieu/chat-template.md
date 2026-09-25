---
title: Chat template
description: "Áp chat template với Unsloth, ánh xạ khóa ShareGPT, thêm token mới và chỉ train trên câu trả lời."
---

# Chat template

Chat template là khuôn quy định cách một hội thoại (các lượt user và assistant, system prompt, token kết thúc) được ghép thành chuỗi văn bản trước khi đưa vào tokenizer. Mỗi model có template riêng, và dữ liệu phải được định dạng đúng template của model thì fine-tune mới hiệu quả.

::: tip Tóm tắt
- **Dùng khi:** dataset đã ở dạng hội thoại (ShareGPT, ChatML...) và bạn cần ghép nó đúng khuôn của model trước khi train.
- **Kết quả:** áp được chat template của Unsloth trong 4 bước, ánh xạ khóa ShareGPT, thêm token mới và chỉ tính loss trên câu trả lời.
- **Nên biết trước:** [Suy luận và sampling](/kien-thuc-nen/suy-luan-va-sampling) (chat template, system prompt, tool calling là gì), [Định dạng dữ liệu](/du-lieu/dinh-dang).
:::

## Vì sao dùng template của Unsloth

Template của Unsloth được khuyên dùng thay vì `apply_chat_template` gốc của tokenizer, và docs đưa ra hai lý do:

- Thuộc tính `chat_template` do nhà phát hành model upload đôi khi có lỗi, và lâu mới được sửa. Unsloth kiểm tra và sửa lỗi template cho mọi model khi upload bản lượng tử hóa lên repo của mình.
- `get_chat_template` có thêm tính năng xử lý dữ liệu.

Nếu template bạn cần chưa được hỗ trợ, hãy gửi feature request trên GitHub. Trong lúc chờ, tạm dùng `apply_chat_template` của tokenizer. Danh sách mọi template Unsloth dùng nằm trong [chat_templates.py](https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py).

**Nguồn:** https://unsloth.ai/docs/basics/chat-templates

## Áp chat template với Unsloth (4 bước)

Quy trình gồm bốn bước: xem template có sẵn, gắn template vào tokenizer, viết hàm định dạng, rồi áp hàm đó lên dataset.

1. Xem các template được hỗ trợ:

```
from unsloth.chat_templates import CHAT_TEMPLATES
print(list(CHAT_TEMPLATES.keys()))
```

Ví dụ kết quả:

```
['unsloth', 'zephyr', 'chatml', 'mistral', 'llama', 'vicuna', 'vicuna_old', 'vicuna old', 'alpaca', 'gemma', 'gemma_chatml', 'gemma2', 'gemma2_chatml', 'llama-3', 'llama3', 'phi-3', 'phi-35', 'phi-3.5', 'llama-3.1', 'llama-31', 'llama-3.2', 'llama-3.3', 'llama-32', 'llama-33', 'qwen-2.5', 'qwen-25', 'qwen25', 'qwen2.5', 'phi-4', 'gemma-3', 'gemma3']
```

2. Gắn template phù hợp vào tokenizer bằng `get_chat_template`:

```
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3", # change this to the right chat_template name
)
```

3. Định nghĩa hàm định dạng. Hàm này áp template cho từng mẫu:

```
def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
   return { "text" : texts, }
```

4. Nạp dataset và áp hàm định dạng:

```
# Import and load dataset
from datasets import load_dataset
dataset = load_dataset("repo_name/dataset_name", split = "train")

# Apply the formatting function to your dataset using the map method
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

Nếu dataset ở dạng ShareGPT (`"from"`/`"value"`) mà model cần ChatML (`"role"`/`"content"`), bạn chuyển đổi trước bằng `standardize_sharegpt`:

```
# Import dataset
from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

# Convert your dataset to the "role"/"content" format if necessary
from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)

# Apply the formatting function to your dataset using the map method
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

Lưu ý hai chỗ trong docs nói khác nhau. Mục Q&A của Datasets Guide nói chỉ dùng `standardize_sharegpt` khi dataset là ShareGPT còn model cần ChatML. Tutorial lại dặn "Always call this!". Xem hộp cảnh báo ở mục [Dữ liệu dạng bảng nhiều cột](/du-lieu/dinh-dang#bang-nhieu-cot).

**Nguồn:** https://unsloth.ai/docs/basics/chat-templates, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama

## Ánh xạ khóa ShareGPT trực tiếp (`mapping`)

Thay vì chuyển đổi dataset, bạn có thể dùng tham số `mapping` của `get_chat_template` để ánh xạ thẳng các khóa `from`/`value`/`human`/`gpt`. Tham số `map_eos_token` ánh xạ `<|im_end|>` thành EOS (token kết thúc chuỗi) mà không cần train:

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("philschmid/guanaco-sharegpt-style", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

::: warning Docs chưa thống nhất: template nào dùng được với `mapping`
Danh sách template trong comment ngắn hơn nhiều so với danh sách in ra:
- Comment trong code trên: `chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth` — [Chat Templates](https://unsloth.ai/docs/basics/chat-templates)
- Danh sách `CHAT_TEMPLATES` in ra dài hơn nhiều (thêm `gemma`, `llama-3.1`, `qwen-2.5`, `phi-4`, `gemma-3`...) — [Chat Templates](https://unsloth.ai/docs/basics/chat-templates), [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)

Docs không nói `mapping` có dùng được với các template ngoài 8 tên trong comment hay không: cần kiểm tra lại.
:::

Bạn cũng có thể tự viết template riêng bằng cách truyền tuple `(custom_template, eos_token)`. Khi đó `eos_token` phải được dùng bên trong template. Với template tùy biến kiểu notebook Ollama, bắt buộc có trường `{INPUT}` cho chỉ dẫn và `{OUTPUT}` cho đầu ra. Trường `{SYSTEM}` là tùy chọn.

**Nguồn:** https://unsloth.ai/docs/basics/chat-templates, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama

## Thêm token mới {#them-token-moi}

Hàm `add_new_tokens` thêm token đặc biệt vào tokenizer, ví dụ `<CHARACTER_1>`, `<THINKING>`, `<SCRATCH_PAD>`. Hàm này **phải được gọi trước** `FastLanguageModel.get_peft_model`:

```python
model, tokenizer = FastLanguageModel.from_pretrained(...)
from unsloth import add_new_tokens
add_new_tokens(model, tokenizer, new_tokens = ["<CHARACTER_1>", "<THINKING>", "<SCRATCH_PAD>"])
model = FastLanguageModel.get_peft_model(...)
```

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/basics/chat-templates

## Chỉ train trên câu trả lời {#chi-train-tren-cau-tra-loi}

Sau khi đã áp template, bạn có thể che phần user và chỉ tính loss trên phần assistant bằng `train_on_responses_only`. Hàm này cần chuỗi đánh dấu đầu lượt user và đầu lượt assistant **đúng theo template** của model. Ví dụ Llama 3 dùng `<|start_header_id|>user<|end_header_id|>`, còn Gemma dùng `<start_of_turn>user`. Code đầy đủ ở trang [Quy trình](/fine-tuning/quy-trinh).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

## Đọc tiếp

- [Làm dữ liệu trong Studio](/du-lieu/studio) — trang kế tiếp: chọn format và mapping cột trong giao diện Studio.
- [Quy trình từng bước](/fine-tuning/quy-trinh) — code đầy đủ, gồm cả `train_on_responses_only`.
- [Định dạng dữ liệu](/du-lieu/dinh-dang) — xem lại khuôn ShareGPT và ChatML trước khi áp template.
