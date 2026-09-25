---
title: Mở cho máy khác (LAN, Cloudflare)
description: "Mở server Unsloth cho máy khác trong mạng nội bộ, hoặc ra Internet qua Cloudflare tunnel, và mỗi kiểu mở ra tới đâu."
---

# Mở cho máy khác (LAN, Cloudflare)

Unsloth Studio có sẵn hai chế độ cho thiết bị khác dùng model đang chạy trên máy bạn: chế độ LAN dành cho thiết bị cùng mạng nội bộ, còn chế độ remote qua Cloudflare dành cho truy cập từ bất kỳ đâu qua internet.

::: tip Tóm tắt
- **Dùng khi:** bạn muốn điện thoại, laptop khác hoặc người ở xa dùng model đang chạy trong Unsloth Studio trên máy mình.
- **Kết quả:** bật được chế độ LAN hoặc link Cloudflare, biết mỗi kiểu launch mở máy bạn ra tới đâu và các điểm bảo mật cần giữ.
- **Nên biết trước:** [Cài đặt và phần cứng](/cai-dat), [Chạy model và gọi API](/inference/).
:::

## LAN (mạng nội bộ)

Thiết bị dùng cùng Wi-Fi hoặc cùng mạng dây truy cập qua địa chỉ dạng `http://192.168.1.42:8888`. Chế độ này không cần internet, và dữ liệu không rời mạng nội bộ.

- Bật lúc khởi động: thêm `-H 0.0.0.0`.

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

- Bật khi Studio đang chạy: mở trang cài đặt Remote & LAN (đường dẫn menu xem hộp bên dưới), vào thẻ **LAN access**, bấm **Start**. Trạng thái **Online** nghĩa là địa chỉ đã phản hồi. Toggle **Start automatically** giúp tự bật LAN mỗi lần khởi động.

**Nguồn:** https://unsloth.ai/docs/basics/lan

## Remote qua Cloudflare tunnel

Unsloth tạo một link HTTPS kiểu `https://<random>.trycloudflare.com`. Bạn không cần tài khoản Cloudflare, domain hay mở port trên router.

```bash
unsloth studio --secure -p 8888
```

Cờ `--secure` giữ Unsloth bind ở `127.0.0.1` và chỉ publish ra ngoài qua tunnel. Nếu tunnel lỗi, Unsloth **thoát** chứ không fallback sang port thô. Khi Studio đang chạy, bạn tìm thẻ **Remote access**, bấm **Start**, rồi copy **Remote URL** hoặc quét mã QR.

::: warning Docs chưa thống nhất
Đường dẫn menu tới thẻ LAN access và thẻ Remote access được ghi khác nhau ở các chỗ:

| Chỗ trong docs | Đường dẫn menu |
| --- | --- |
| Trang LAN, đoạn mở đầu và Quickstart ([lan](https://unsloth.ai/docs/basics/lan)) | Settings → API → **Remote & LAN** (thẻ LAN access) |
| Trang Cloudflare, đoạn mở đầu ([remote access](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth)) | Settings → API → **Remote access** |
| Trang Cloudflare, mục Quickstart ([remote access](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth)) | Settings → **Remote & LAN** → Remote access |
| Trang Cloudflare, mục "UI: start a link…" ([remote access](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth)) | Settings → API → tìm thẻ **Remote access** |

Nếu không thấy menu theo một đường dẫn, hãy thử các đường dẫn còn lại.
:::

Đặt mật khẩu khi launch headless (không có terminal để nhập):

```bash
UNSLOTH_STUDIO_PASSWORD='your-strong-password' unsloth studio --secure
```

**Nguồn:** https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth, https://unsloth.ai/docs/basics/lan

## Mỗi kiểu launch mở ra tới đâu

Cờ bạn truyền khi launch quyết định port thô mở cho ai và có URL Cloudflare công khai hay không:

| Lệnh | Port thô | URL Cloudflare công khai |
| --- | --- | --- |
| `unsloth studio` | chỉ máy này | không |
| `unsloth studio -H 0.0.0.0` | cả mạng nội bộ | không |
| `unsloth studio -H 0.0.0.0 --cloudflare` | cả mạng nội bộ | **có** (kém riêng tư nhất) |
| `unsloth studio --secure` | chỉ máy này | **có**, là lối vào duy nhất |

**Nguồn:** https://unsloth.ai/docs/basics/lan, https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth

## Bảo mật khi mở ra mạng

Mở Unsloth ra mạng nghĩa là người khác có thể tới được server và tool trên máy bạn, nên cần giữ vài nguyên tắc sau.

::: warning Lỗi thường gặp
- **LAN là HTTP thường, không mã hóa.** Ai cùng phân đoạn mạng đều đọc được dữ liệu. Trên mạng không do bạn kiểm soát, hãy dùng `--secure` (HTTPS).
- **Tool phía server chạy dưới quyền user của bạn.** Ai có API key và truy cập được server là chạy được code trên máy bạn. Docs khuyên thêm `--disable-tools` khi mở ra mạng và giữ kín API key. Tool bật hay tắt mặc định: xem hộp "Docs chưa thống nhất" bên dưới.
- Ai có **URL + mật khẩu** là đăng nhập được (user `unsloth`). URL tunnel đổi mỗi lần start. Hãy coi URL là bí mật.
- `-H 0.0.0.0` vẫn để port thô mở. Chỉ `--secure` mới đóng port này.
- Lần đầu publish công khai, nếu admin vẫn dùng mật khẩu tự sinh, Unsloth bắt đổi mật khẩu trước. Tránh truyền `--password VALUE` trên dòng lệnh, vì mật khẩu sẽ lộ trong `ps` và shell history.
- Khi tunnel bật, các MCP server stdio cục bộ bị tắt (trừ khi bạn đặt `UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1`).
:::

::: warning Docs chưa thống nhất
Các trang ghi khác nhau về việc tool phía server (web search, Python, terminal) **bật hay tắt mặc định** khi mở Unsloth ra mạng:

| Thông số | Trang LAN ([lan](https://unsloth.ai/docs/basics/lan)) | Trang Cloudflare ([remote access](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth)) | Trang API ([api](https://unsloth.ai/docs/basics/api)) |
| --- | --- | --- | --- |
| Lệnh được nói tới | Trang hướng dẫn `unsloth studio -H 0.0.0.0`; mục Security không ghi rõ lệnh | Trang hướng dẫn `unsloth studio --secure` / `--cloudflare`; mục Security không ghi rõ lệnh | `unsloth run` |
| Tool mặc định | "on by default" | "on by default" | Bind `127.0.0.1`: bật; bind `0.0.0.0` hoặc địa chỉ không phải loopback: **tắt** |
| Cách ép | `--disable-tools` | `--disable-tools` | `--enable-tools` / `--disable-tools` (trên `0.0.0.0`, `--enable-tools` hỏi y/N) |

Khi mở ra mạng, hãy luôn truyền `--disable-tools` một cách tường minh thay vì dựa vào mặc định.
:::

**Nguồn:** https://unsloth.ai/docs/basics/lan, https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth, https://unsloth.ai/docs/basics/api

## Đọc tiếp

- [Lỗi thường gặp](/export-deploy/loi-thuong-gap) — trang kế tiếp: có cả lỗi LAN, `--secure` và `cloudflared`.
- [Gọi qua API (OpenAI, Anthropic)](/inference/api) — gọi model từ máy khác sau khi đã mở server.
- [Chạy model đã xuất](/export-deploy/chay-model) — các engine khác ngoài Studio nếu bạn tự host.
