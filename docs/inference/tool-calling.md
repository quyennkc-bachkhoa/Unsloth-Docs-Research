---
title: "Tool calling"
description: "Tool calling trong Studio, qua API (client-side và server-side tools) và vòng lặp tự viết với llama-server."
---

# Tool calling

Tool calling (gọi công cụ) cho phép LLM kích hoạt một hàm cụ thể, như "tìm file", "chạy máy tính", "gọi API". Thay vì đoán câu trả lời bằng văn bản, model xuất ra một request có cấu trúc.

Điểm quan trọng: model chỉ **đề xuất** lời gọi, gồm tên hàm và tham số JSON. Chương trình của bạn mới là bên chạy hàm. Sau đó chương trình gửi kết quả lại cho model ở lượt sau.

::: tip Kiến thức nền
Cơ chế tool calling và vai trò của chat template: xem [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling).
:::

## Unsloth hỗ trợ thế nào

Có ba cách dùng tool calling với Unsloth, khác nhau ở chỗ ai chạy hàm:

- **Trong Studio:** tool calling được cài sẵn. Bạn chỉ cần chọn model và bật hoặc tắt. Studio có thêm:
  - self-healing (tự sửa tool call hỏng);
  - khử trùng lặp, chống XML lọt vào output;
  - cho phép hơn 25 tool call, và dừng vòng gọi ổn định hơn.

  Docs Unsloth ghi tool call "chính xác hơn 30% đến 80%" trên mọi model.
- **Qua API, client-side tools:** client của bạn tự chạy hàm và trả kết quả. Gửi `tools` + `tool_choice` kiểu OpenAI tới `/v1/chat/completions`. Hoặc gửi `tools` (có `input_schema`) kiểu Anthropic tới `/v1/messages`. Giá trị `tool_choice` của hai bên ánh xạ như sau: Anthropic `auto` → OpenAI `auto`, `any` → `required`, `none` → `none`.
- **Qua API, server-side tools:** Unsloth tự chạy Python, bash, web search, rồi stream kết quả về dưới dạng sự kiện `tool_result`. Bật bằng các trường thêm:

| Trường | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `enable_thinking` | `boolean` | `false` để tắt thinking; mặc định `true` |
| `enable_tools` | `boolean` | `true` để bật chạy tool phía server |
| `enabled_tools` | `array<string>` | Tool model được gọi: `python`, `bash`, `web_search` |
| `session_id` | `string` | Tùy chọn; giữ trạng thái tool (vd Python kernel) giữa các lần gọi |

::: warning Docs chưa thống nhất
Các trang ghi tên tool hợp lệ trong `enabled_tools` không giống nhau:

| Thông số | [basics/api](https://unsloth.ai/docs/basics/api) | [integrations/connect-python-sdk-to-unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth), [integrations/connect-curl-and-http-to-unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth) |
| --- | --- | --- |
| Ví dụ JSON | `["python", "web_search","terminal"]` | `["python", "web_search"]`, `["python"]`, `["web_search", "python"]` |
| Danh sách hỗ trợ | Văn bản: "execute Python, web search, and bash"; Troubleshooting ví dụ `["python", "web_search"]` | "supports `python`, `bash`, and `web_search`" |
:::

Ví dụ function calling phía client với OpenAI SDK. Model trả về tên hàm và tham số, code của bạn in chúng ra:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
                },
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Perth right now?"}],
    tools=tools,
    tool_choice="auto",
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, tool_call.function.arguments)
```

Ví dụ server-side tools qua OpenAI SDK. Các trường riêng của Unsloth được đặt trong `extra_body` để đi thẳng tới server:

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 123 * 456? Use Python to compute it."}],
    stream=True,
    extra_body={
        "enable_tools": True,
        "enabled_tools": ["python", "web_search"],
        "session_id": "my-session",
    },
)
for chunk in stream:
    if chunk.choices:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
```

**Nguồn:** https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/new/studio/chat

## Vòng lặp tool calling tự viết với llama-server

Tool Calling Guide của Unsloth hướng dẫn tự viết vòng gọi tool bằng Python. Cách chuẩn bị:

- Chạy model bằng `llama-server` ở port `8001`.
- Định nghĩa sẵn các hàm Python: `add_number`, `multiply_number`, `subtract_number`, `write_a_story`, `terminal`, `python`.
- Tạo bảng `MAP_FN` ánh xạ tên sang hàm, và danh sách `tools` theo schema OpenAI. Code đầy đủ xem ở trang nguồn.

Hàm dưới đây gọi endpoint OpenAI-compatible và đọc `tool_calls`. Sau đó nó chạy hàm tương ứng và nối kết quả vào `messages` với `role: "tool"`:

```python
from openai import OpenAI
def unsloth_inference(
    messages,
    temperature = 0.7,
    top_p = 0.95,
    top_k = 40,
    min_p = 0.01,
    repetition_penalty = 1.0,
):
    messages = messages.copy()
    openai_client = OpenAI(
        base_url = "http://127.0.0.1:8001/v1",
        api_key = "sk-no-key-required",
    )
    model_name = next(iter(openai_client.models.list())).id
    print(f"Using model = {model_name}")
    has_tool_calls = True
    original_messages_len = len(messages)
    while has_tool_calls:
        print(f"Current messages = {messages}")
        response = openai_client.chat.completions.create(
            model = model_name,
            messages = messages,
            temperature = temperature,
            top_p = top_p,
            tools = tools if tools else None,
            tool_choice = "auto" if tools else None,
            extra_body = {"top_k": top_k, "min_p": min_p, "repetition_penalty" :repetition_penalty,}
        )
        tool_calls = response.choices[0].message.tool_calls or []
        content = response.choices[0].message.content or ""
        tool_calls_dict = [tc.to_dict() for tc in tool_calls] if tool_calls else tool_calls
        messages.append({"role": "assistant", "tool_calls": tool_calls_dict, "content": content,})
        for tool_call in tool_calls:
            fx, args, _id = tool_call.function.name, tool_call.function.arguments, tool_call.id
            out = MAP_FN[fx](**json.loads(args))
            messages.append({"role": "tool", "tool_call_id": _id, "name": fx, "content": str(out),})
        else:
            has_tool_calls = False
    return messages
```

Gọi thử:

```python
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "What is today's date plus 3 days?"}],
}]
unsloth_inference(messages, temperature = 0.15, top_p = 1.0, top_k = -1, min_p = 0.00)
```

Ví dụ này dùng Devstral 2. Nếu đổi model, bạn phải dùng đúng sampling parameters khuyến nghị cho model đó. Server llama.cpp cần cờ `--jinja` để tool calling dùng đúng chat template.

**[Nhận định]** Đoạn code trên có một điểm dễ bỏ sót. Trong Python, `for ... else` chạy nhánh `else` khi vòng `for` kết thúc mà không gặp `break`. Vì vậy code luôn đặt `has_tool_calls = False` sau lượt đầu. Nó chạy tool đúng một vòng rồi dừng, không tự gửi kết quả tool lại cho model. Nếu cần vòng lặp nhiều bước, bạn phải tự sửa điều kiện dừng.

**Nguồn:** https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/new/studio/chat
