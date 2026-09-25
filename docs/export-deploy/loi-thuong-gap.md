---
title: Lỗi thường gặp
description: "Các lỗi hay gặp sau khi export model và cách xử lý theo docs."
---

# Lỗi thường gặp

Trang này gom các lỗi hay gặp khi export và deploy, kèm nguyên nhân và cách xử lý mà docs đưa ra, chia thành lỗi của chính model sau khi export và lỗi khi mở Studio cho máy khác.

::: tip Tóm tắt
- **Dùng khi:** model chạy tốt trong Unsloth nhưng ra rác, lặp hoặc không chạy sau khi export, hoặc máy khác không vào được Studio.
- **Kết quả:** nhận ra triệu chứng của mình trong bảng và biết cách xử lý theo docs.
- **Nên biết trước:** [Export và deploy](/export-deploy/), [Chạy model đã xuất](/export-deploy/chay-model).
:::

## Model chạy sai sau khi export

Nguyên nhân phổ biến nhất là chat template không khớp với lúc train; các lỗi còn lại chủ yếu do hết bộ nhớ hoặc đặt file sai chỗ.

| Triệu chứng | Nguyên nhân / cách xử lý theo docs |
| --- | --- |
| Chạy trong Unsloth tốt, sang Ollama, vLLM hay llama.cpp thì ra rác hoặc lặp vô hạn | Nguyên nhân phổ biến nhất là **chat template sai**: phải dùng đúng template lúc train. Kiểm tra `eos token`. Kiểm tra engine có thêm thừa hay thiếu token "start of sequence". Dùng conversational notebook của Unsloth để ép template |
| LM Studio ra rác hoặc lặp | Template không khớp. Vào **My Models** → bánh răng → **Prompt Template**, đặt đúng template đã train |
| Save GGUF hoặc vLLM 16-bit bị crash (OOM, hết bộ nhớ) | Giảm `maximum_memory_usage` (mặc định `0.75`), ví dụ xuống `0.5` |
| LM Studio không thấy model | Dùng `lms import`, hoặc đặt file đúng cấu trúc `~/.lmstudio/models/publisher/model/model-file.gguf` |
| LM Studio OOM hoặc chạy chậm | Dùng quant nhỏ hơn (`Q4_K_M`), giảm context, chỉnh GPU offload |

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf, https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio

## Lỗi khi mở Studio cho máy khác

Các lỗi này liên quan tới việc chạy Studio và mở nó qua LAN hoặc Cloudflare tunnel.

| Triệu chứng | Nguyên nhân / cách xử lý theo docs |
| --- | --- |
| `Unsloth Studio is already running on port 8888` | Chạy `unsloth studio stop` hoặc đổi `--port` |
| Thiết bị khác không vào được địa chỉ LAN | Hai thiết bị phải cùng mạng. Kiểm tra Guest Wi-Fi, AP isolation, VPN và firewall của máy |
| `--secure` thoát ngay | Tunnel bị lỗi. Sửa kết nối mạng trước (đừng vội dùng `--no-secure` vì nó mở port thô) |
| `cloudflared is unavailable` | Kiểm tra máy có truy cập ra `github.com` không, hoặc tự cài `cloudflared` vào `PATH` |

**Nguồn:** https://unsloth.ai/docs/basics/lan, https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth

## Đọc tiếp

- [Ứng dụng RAG](/ung-dung-rag) — trang kế tiếp: khi nào dùng RAG, khi nào fine-tune, và phác một chatbot RAG nội bộ chạy bằng Unsloth.
- [Chat template](/du-lieu/chat-template) — sửa tận gốc lỗi phổ biến nhất: template lúc chạy không khớp lúc train.
- [Mở cho máy khác (LAN, Cloudflare)](/export-deploy/lan-remote) — xem lại các chế độ LAN, `--secure` và cách mỗi kiểu mở máy bạn ra ngoài.
