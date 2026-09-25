---
title: Nạp dữ liệu trong Studio
description: "Bước Dataset trong Unsloth Studio: tab HuggingFace Hub và Local, chọn format và các tùy chọn split, slice, mapping cột."
---

# Nạp dữ liệu trong Unsloth Studio

Nếu train bằng Studio (giao diện web chạy local), bạn nạp dataset ở bước Dataset. Bước này có hai tab:

- **HuggingFace Hub**: tìm dataset trực tiếp trên Hub, kèm ngày cập nhật gần nhất.
- **Local**: kéo-thả hoặc upload file có cấu trúc hoặc không cấu trúc: `PDF`, `DOCX`, `JSONL`, `JSON`, `CSV`, `Parquet`.

Sau đó bạn chọn cách Studio hiểu dữ liệu:

| Format | Khi nào dùng |
| --- | --- |
| `auto` | Để Unsloth tự nhận diện |
| `alpaca` | Có cột `instruction` / `input` / `output` |
| `chatml` | Mảng `messages` kiểu OpenAI |
| `sharegpt` | Hội thoại kiểu ShareGPT |

Các tùy chọn khác:

- **Subset**: tự lấy từ dataset card.
- **Train split / Eval split**: nếu chọn eval split, bạn có thêm biểu đồ Eval Loss.
- **Dataset slice**: giới hạn khoảng dòng để thử nhanh.

Nếu Studio không tự ánh xạ được cột, hộp **Dataset Preview** sẽ mở ra để bạn gán từng cột vào `instruction`, `input`, `output`, `image`...

**Nguồn:** https://unsloth.ai/docs/new/studio/start
