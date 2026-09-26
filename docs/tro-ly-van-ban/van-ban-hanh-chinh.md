---
title: Tìm và chuẩn bị dữ liệu
description: "Cách tìm dataset tiếng Việt trên Hugging Face, các mục cần xem trên trang dataset, và các bộ về pháp luật hiện có."
---

# Tìm và chuẩn bị dữ liệu

Trang này hướng dẫn cách tìm dataset tiếng Việt trên Hugging Face và các mục cần xem trên trang dataset trước khi dùng.

::: tip Tóm tắt
- **Tìm:** huggingface.co/datasets, lọc ngôn ngữ Vietnamese, gõ từ khóa không dấu hoặc tiếng Anh (`legal`, `phap luat`).
- **Xem trên trang dataset:** nguồn dữ liệu, license, người tổng hợp, ngày cập nhật, cách tạo dữ liệu, cột dữ liệu.
- **Kết quả tìm ngày 26/09/2026:** `legal` + tiếng Việt ra 50 dataset. `công văn` và `cong van` ra 0 dataset.
:::

::: warning Dữ liệu có hạn dùng
Tra cứu ngày **2026-09-26**. Kết quả tìm kiếm thay đổi khi có dataset mới.
:::

Thể thức văn bản hành chính quy định tại **Nghị định 30/2020/NĐ-CP**, toàn văn trên [vanban.chinhphu.vn](https://vanban.chinhphu.vn/default.aspx?pageid=27160&docid=199378).

## Cách tìm trên Hugging Face

### Trên web

1. Vào [huggingface.co/datasets](https://huggingface.co/datasets?language=language:vi&sort=downloads). Link này đã lọc ngôn ngữ Vietnamese và xếp theo lượt tải (Most downloads).
2. Gõ từ khóa vào ô tìm kiếm.
3. Muốn xem bộ mới, đổi cách sắp xếp sang **Recently updated** hoặc **Recently created**.
4. Bấm vào dataset để xem **dataset card** (mô tả) và **Dataset Viewer**. Dataset Viewer hiện nội dung dataset theo trang 100 dòng, có biểu đồ phân bố từng cột, ô tìm từ và SQL Console.

Số kết quả khi tìm bằng API ngày 26/09/2026:

| Từ khóa | Lọc tiếng Việt | Số dataset |
| --- | --- | --- |
| `legal` | Có | 50 |
| `law` | Có | 4 |
| `phap luat` | Không | 13 |
| `pháp luật` (có dấu) | Không | 0 |
| `công văn` / `cong van` | Không | 0 |

Theo docs `huggingface_hub`, tham số `search` trả về các dataset "chứa" chuỗi tìm kiếm. Trong lần thử trên, gõ có dấu không ra kết quả, gõ không dấu thì có.

### Bằng code

```python
from huggingface_hub import HfApi

api = HfApi()
for ds in api.list_datasets(search="legal", filter="language:vi", sort="downloads", limit=20):
    print(ds.id, ds.downloads)
```

Đã chạy thử với `huggingface_hub` 2.0.0. `sort` nhận `"created_at"`, `"downloads"`, `"last_modified"`, `"likes"`, `"trending_score"`. Dùng dòng lệnh: `hf datasets ls --search legal`.

**Nguồn:** https://huggingface.co/docs/hub/datasets-overview, https://huggingface.co/docs/hub/datasets-viewer, https://huggingface.co/docs/huggingface_hub/guides/search

## Các mục cần xem trên trang dataset

Ví dụ lấy từ kết quả tìm `legal` + tiếng Việt ngày 26/09/2026. Mọi thông tin trong cột "Ví dụ" trích từ card của chính dataset đó.

| Mục | Xem ở đâu | Ví dụ |
| --- | --- | --- |
| **Nguồn dữ liệu** | Phần mô tả, mục Source | `th1nhng0/vietnamese-legal-documents`: nguồn vbpl.vn (Bộ Tư pháp). `vohuutridung/vietnamese-legal-documents`: nguồn thuvienphapluat.vn |
| **Người tổng hợp** | Mục Curated by, Source | `anhquan12/vietnamese-legal-documents` ghi người tổng hợp là Thịnh Ngô (tác giả `th1nhng0`), 153.420 văn bản. Bản `th1nhng0` hiện có 171.556 văn bản |
| **Card có nội dung không** | Toàn bộ card | `DuongTrongChi/vi-legal-docs`: card chỉ có khung mẫu, không có mô tả, không có license |
| **License** | Tag `license:`, thanh bên phải | `tmquan/pbgdpl-vn-legal-qna`: license `other`, tên `vietnamese-government-public-q-and-a` |
| **Thời gian của dữ liệu** | Phần mô tả | `tmquan/pbgdpl-vn-legal-qna`: câu hỏi từ 13/03/2007 đến 20/10/2021 |
| **Căn cứ pháp lý dùng khi tạo** | Phần mô tả | `duyet/vietnamese-legal-instruct` mã hóa thứ bậc văn bản "per Luật ban hành VBQPPL 2015". Luật 64/2025/QH15 "có hiệu lực thi hành từ ngày 01/4/2025 thay thế Luật Ban hành văn bản quy phạm pháp luật số 80/2015/QH13" |
| **Cách tạo dữ liệu** | Tag `annotations_creators`, `synthetic`, phần mô tả | `ThanhVu101/Vietnamese-Legal-QA`: tag `machine-generated`, `synthetic`, có cột nhãn của LLM judge. `duyet/vietnamese-legal-instruct`: tạo bằng 14 bộ sinh câu hỏi chạy local, không gọi API LLM |
| **Cột dữ liệu** | Dataset Viewer, phần Format | `duyet/vietnamese-legal-instruct`: cột `conversations` gồm `role`/`content`, card ghi "Unsloth-compatible conversation format" |

Tải về xem vài mẫu:

```python
from datasets import load_dataset

ds = load_dataset("duyet/vietnamese-legal-instruct", split="train")
print(ds)       # số dòng, tên các cột
print(ds[0])    # một mẫu
```

Cách đổi sang định dạng Unsloth nhận: [Định dạng dữ liệu](/du-lieu/dinh-dang).

**Nguồn:** https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents, https://huggingface.co/datasets/vohuutridung/vietnamese-legal-documents, https://huggingface.co/datasets/anhquan12/vietnamese-legal-documents, https://huggingface.co/datasets/DuongTrongChi/vi-legal-docs, https://huggingface.co/datasets/tmquan/pbgdpl-vn-legal-qna, https://huggingface.co/datasets/duyet/vietnamese-legal-instruct, https://huggingface.co/datasets/ThanhVu101/Vietnamese-Legal-QA, https://daibieunhandan.dienbien.gov.vn/blog/van-ban-hanh-chinh-13/luat-ban-hanh-van-ban-quy-pham-phap-luat-so-64-2025-qh15-co-hieu-luc-thi-hanh-tu-ngay-01-4-2025-549

## Các bộ về pháp luật tiếng Việt

**[Nguồn ngoài]** Số liệu lấy từ card và Dataset Viewer ngày 26/09/2026:

| Dataset | Nội dung (theo card) | Số dòng | License |
| --- | --- | --- | --- |
| [`th1nhng0/vietnamese-legal-documents`](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) | Văn bản pháp luật từ vbpl.vn, có cột `tinh_trang_hieu_luc` (tình trạng hiệu lực) và bảng quan hệ giữa các văn bản | 171.556 văn bản | CC-BY-4.0 |
| [`duyet/vietnamese-legal-instruct`](https://huggingface.co/datasets/duyet/vietnamese-legal-instruct) | Cặp hỏi đáp dạng `conversations`, tạo từ `th1nhng0/vietnamese-legal-documents` | 467.732 | CC-BY-4.0 |
| [`GreenNode/zalo-ai-legal-text-retrieval-vn`](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) | Truy hồi văn bản pháp luật, dữ liệu Zalo AI, nằm trong bộ đánh giá MTEB | corpus 61,4 nghìn, 818 câu hỏi | MIT |
| [`hirine/dataset-thu-tuc-hanh-chinh-5733-samples`](https://huggingface.co/datasets/hirine/dataset-thu-tuc-hanh-chinh-5733-samples) | Thủ tục hành chính | 5.733 | CC-BY-4.0 |
| [`tmquan/pbgdpl-vn-legal-qna`](https://huggingface.co/datasets/tmquan/pbgdpl-vn-legal-qna) | Hỏi đáp pháp luật từ pbgdpl.gov.vn (Bộ Tư pháp) | 4.593 | other |
| [`VLSP2025-LegalSML/Public-Test`](https://huggingface.co/datasets/VLSP2025-LegalSML/Public-Test) | Đề công khai VLSP 2025: trắc nghiệm, NLI, suy luận tam đoạn luận | 440 | Không ghi |

**Nguồn:** https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents, https://huggingface.co/datasets/duyet/vietnamese-legal-instruct, https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn, https://huggingface.co/datasets/hirine/dataset-thu-tuc-hanh-chinh-5733-samples, https://huggingface.co/datasets/tmquan/pbgdpl-vn-legal-qna, https://huggingface.co/datasets/VLSP2025-LegalSML/Public-Test

## Quy định liên quan

**[Nguồn ngoài]**

- **Quyền tác giả:** theo Điều 15 Luật Sở hữu trí tuệ, văn bản của cơ quan nhà nước, tổ chức chính trị, tổ chức chính trị - xã hội… không thuộc phạm vi bảo hộ quyền tác giả.
- **Bí mật nhà nước:** Luật 117/2025/QH15, hiệu lực 01/3/2026. Cho phép soạn thảo, lưu giữ bí mật nhà nước trên mạng LAN độc lập, tức "mạng máy tính cục bộ được thiết lập, giới hạn trong một trụ sở cơ quan, tổ chức, không kết nối với mạng Internet". Cấm "sử dụng hệ thống trí tuệ nhân tạo hoặc công nghệ mới để xâm phạm bí mật nhà nước".
- **Dữ liệu cá nhân:** Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15, ban hành 26/6/2025, hiệu lực 01/01/2026.
- **Chatbot AI:** Công văn 557/BKHCN-CĐSQG ngày 31/3/2025 hướng dẫn không đưa lên chatbot "bất kỳ dữ liệu, tài liệu nào thuộc danh mục bí mật Nhà nước, cũng như các thông tin nhạy cảm như thông tin cá nhân, tài liệu nội bộ".

**Nguồn:** https://lsvn.vn/doi-tuong-nao-khong-thuoc-pham-vi-bao-ho-quyen-tac-gia-1683458018-a130123.html, https://xaydungchinhsach.chinhphu.vn/nhung-diem-moi-quan-trong-cua-luat-bao-ve-bi-mat-nha-nuoc-co-hieu-luc-tu-1-3-2026-119260301170433415.htm, https://chinhphu.vn/?pageid=27160&docid=214590&classid=1&typegroupid=3, https://xaydungchinhsach.chinhphu.vn/huong-dan-su-dung-chatbot-ai-ho-tro-can-bo-cong-chuc-nguoi-lao-dong-trong-cong-viec-119250405175138265.htm

## Đọc tiếp

- [Định dạng dữ liệu](/du-lieu/dinh-dang) — cấu trúc file dataset mà Unsloth nhận.
- [Ứng dụng RAG](/ung-dung-rag) — dựng kho tra cứu văn bản.
- [Host model trên máy chủ](/tro-ly-van-ban/hosting) — trang kế tiếp.
