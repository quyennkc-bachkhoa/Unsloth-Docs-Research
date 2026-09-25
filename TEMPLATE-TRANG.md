# Template trang — route Unsloth

File này là chuẩn trình bày cho mọi trang trong route Unsloth (`docs/`, trừ `docs/kien-thuc-nen/`). File nằm ngoài `docs/` nên VitePress không render.

## Khung một trang

```md
---
title: <Tên theo việc làm được>
description: "<1 câu mô tả trang>"
---

# <Tên theo việc làm được>

<1–2 câu lời thường: trang này giúp bạn làm gì, đọc xong có được gì.>

::: tip Tóm tắt
- **Dùng khi:** <tình huống cần tới trang này>
- **Kết quả:** <đọc/làm xong bạn có gì>
- **Nên biết trước:** <link Kiến thức nền hoặc trang trước; không cần thì ghi "Không cần kiến thức trước.">
:::

## <Mục 1>

<Câu đầu tiên nói ý chính của mục. Sau đó mới tới giải thích, bảng, code, sơ đồ.>

**Nguồn:** <URL docs dùng trong mục này>

## <Mục 2>

...

**Nguồn:** ...

## Đọc tiếp

- [<Trang kế tiếp theo thứ tự sidebar>](/...) — <1 câu: vì sao nên đọc tiếp>
- [<Trang liên quan>](/...) — <1 câu>
```

## Quy tắc

**Tên trang**
- Tên mô tả việc người đọc làm được, không để thuật ngữ trần. Thuật ngữ nếu cần thì để sau hoặc trong ngoặc: "Train bằng GRPO", "Định dạng trọng số NVFP4 và FP8".
- `title` trong frontmatter, H1 và text trên sidebar giống hệt nhau.
- Ngắn, khoảng 30 ký tự trở xuống. Dùng "và", không dùng "&".
- Title có dấu `:` thì đặt trong ngoặc kép.

**Mở đầu**
- Luôn có 1–2 câu lời thường ngay dưới H1, rồi tới hộp `::: tip Tóm tắt` đúng 3 dòng: Dùng khi / Kết quả / Nên biết trước.
- Hộp "Kiến thức nền" cũ gộp vào dòng "Nên biết trước". Không đặt hộp Kiến thức nền rải rác nữa.

**Thân trang**
- Ít nhất 2 mục H2, để cột "Trên trang này" luôn có nội dung. Chia nhỏ hơn thì dùng H3.
- Câu đầu mỗi H2 là ý chính của mục.
- Mỗi H2 kết thúc bằng một dòng `**Nguồn:**` chỉ gồm URL dùng trong mục đó. Riêng "Đọc tiếp" và bảng "Các trang trong phần này" thì không cần.
- Hộp cảnh báo lỗi/cạm bẫy luôn tên `::: warning Lỗi thường gặp`.
- Hộp mâu thuẫn giữ nguyên `::: warning Docs chưa thống nhất: <chủ đề>`.
- Ý kiến người viết gắn `**[Nhận định]**` (in đậm). Nguồn ngoài docs Unsloth gắn `**[Nguồn ngoài]**`.
- Sơ đồ vẽ bằng HTML/CSS (`<div class="dg">`, xem `docs/.vitepress/theme/diagrams.css`), không dùng mermaid.

**Kết trang**
- Luôn kết bằng `## Đọc tiếp` với 2–4 link: trang kế tiếp theo thứ tự sidebar, rồi các trang liên quan.

## Trang index của một phần

```md
# <Tên phần>

<1–2 câu: phần này nói về gì.>

::: tip Tóm tắt
- **Dùng khi:** ...
- **Kết quả:** ...
- **Nên biết trước:** ...
:::

## Các trang trong phần này

| Trang | Giúp bạn làm gì | Đọc khi nào |
| --- | --- | --- |
| [<Tên trang>](/...) | ... | ... |

## <Khái niệm chung của phần>

...

**Nguồn:** ...

## Đọc tiếp

- ...
```
