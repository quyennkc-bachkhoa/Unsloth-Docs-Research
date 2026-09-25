---
title: "MCP"
description: "Bật MCP server trong Unsloth Studio và ví dụ ghép llama-server với mcp-cli."
---

# MCP

MCP (Model Context Protocol) là giao thức chuẩn để model gọi công cụ và dịch vụ bên ngoài. Nhờ MCP, model local như Qwen hay Gemma dùng được file, app, cơ sở dữ liệu, dịch vụ của bạn, thay vì chỉ trả lời từ trí nhớ.

MCP chạy được với cả GGUF local lẫn model của nhà cung cấp cloud đã kết nối. Trong cùng một thread, MCP dùng song song được với code execution và web search.

## Bật MCP trong Unsloth Studio

<figure>

![Menu MCP trên thanh công cụ chat](/images/inference/mcp-menu.webp)

<figcaption>Nút <b>MCP</b> trên thanh chat mở danh sách server có sẵn: Context7, Exa, Hugging Face và <b>Add custom MCP</b>. Ảnh: <a href="https://unsloth.ai/docs/basics/mcp">docs Unsloth</a>.</figcaption>
</figure>

1. Bấm **MCP** trên thanh công cụ chat. Studio có sẵn MCP server của **Context7, Exa và Hugging Face**. Lưu ý: bật Exa sẽ tắt công cụ web search mặc định của Unsloth.
2. Muốn thêm server riêng, bấm **Add custom MCP**. Điền **Display name**, ví dụ `Vercel`. Điền **URL** là endpoint gốc của server, ví dụ `https://mcp.vercel.com`.
3. Chọn cách xác thực:
   - Server đăng nhập qua trình duyệt (GitHub, Linear, Vercel…): bật **Use OAuth sign-in**.
   - Server dùng token: để OAuth tắt, bấm **Add header** và thêm:

```
Authorization: Bearer <your-token>
```

4. Bấm **Test connection**. Thành công thì bấm **Add server**. Unsloth tự lấy danh sách tool, ví dụ *Refreshed "Vercel" (18 tools)*.
5. Bật công tắc của từng server, và cả công tắc tổng **Use MCP Servers**. Sau đó chọn model và chat. Model tự gọi tool khi cần.

<figure>

![Hộp thoại thêm MCP server](/images/inference/mcp-add-server.webp)

<figcaption>Thêm server Vercel: điền tên, URL, bật OAuth hoặc thêm header, rồi <b>Test connection</b>. Ảnh: <a href="https://unsloth.ai/docs/basics/mcp">docs Unsloth</a>.</figcaption>
</figure>

<figure>

![Server đã được thêm](/images/inference/mcp-server-added.webp)

<figcaption>Server đã thêm xong, có công tắc bật tắt, nút làm mới danh sách tool, sửa và xóa. Ảnh: <a href="https://unsloth.ai/docs/basics/mcp">docs Unsloth</a>.</figcaption>
</figure>

Tài liệu có một ví dụ dùng cùng lúc 3 server mặc định:

- Hỏi về fine-tune Qwen: Exa trả lời.
- Tìm trong docs Unsloth: Context7.
- Tìm model `unsloth/Qwen` trên Hugging Face: MCP của Hugging Face.

<figure>

![Model gemma-4 dùng MCP của Vercel trong chat](/images/inference/mcp-dung-trong-chat.webp)

<figcaption>Model local <code>gemma-4-E2B-it-GGUF</code> gọi MCP của Vercel để liệt kê các tool nó có thể dùng. Ảnh: <a href="https://unsloth.ai/docs/basics/mcp">docs Unsloth</a>.</figcaption>
</figure>

**Nguồn:** https://unsloth.ai/docs/basics/mcp

## Ví dụ cấu hình với llama.cpp

Nếu không dùng Studio, bạn có thể ghép `llama-server` với `mcp-cli` như tài liệu hướng dẫn. `mcp-cli` là MCP client dòng lệnh của IBM. Đầu tiên, tạo file `server_config.json` khai báo MCP server filesystem:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/ABSOLUTE/PATH/TO/mcp-workspace"
      ],
      "env": {}
    }
  }
}
```

```bash
# terminal 1
llama-server -hf unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL \
  --alias local --host 127.0.0.1 --port 8080 --no-ui \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --chat-template-kwargs '{"enable_thinking":false}'

# terminal 2, in the folder with server_config.json
uvx mcp-cli \
  --provider llamacpp \
  --api-base http://127.0.0.1:8080/v1 \
  --api-key none \
  --model local \
  --server filesystem \
  --config-file server_config.json
```

Trước khi chạy, `mcp-cli` cần thêm file cấu hình `~/.chuk_llm/config.yaml` (xem trang nguồn). Mặc định nó hỏi xác nhận trước mỗi lần chạy tool.

::: danger Bảo mật MCP
Chỉ kết nối MCP server bạn tin cậy. Giữ bước xác nhận của người dùng cho các hành động như đọc dữ liệu riêng tư, đổi deployment, mua domain, sửa project. Cẩn thận khi kết hợp MCP với web search. Một trang web có thể bị prompt injection, tức là bị chèn lệnh độc hại vào nội dung. Khi model đọc trang đó, lệnh độc hại có thể kích hoạt tool call ngoài ý muốn.
:::

**Nguồn:** https://unsloth.ai/docs/basics/mcp
