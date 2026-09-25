---
title: Dữ liệu văn bản hành chính
description: "Văn bản hành chính theo Nghị định 30/2020, nguồn dữ liệu công khai, quy định về dữ liệu cá nhân, bí mật nhà nước và AI, và cách thiết kế dataset fine-tune."
---

# Dữ liệu văn bản hành chính

Trang này gom những gì cần biết trước khi đưa văn bản hành chính vào model: văn bản hành chính là gì theo quy định, lấy dữ liệu ở đâu, luật nào cần tuân thủ, và nên thiết kế dataset ra sao. Phần lớn nội dung đến từ nguồn ngoài docs Unsloth (văn bản pháp luật, Hugging Face). Phần thiết kế dataset là đề xuất của người viết.

::: tip Tóm tắt
- **Dùng khi:** chuẩn bị dataset để fine-tune, hoặc thiết kế hệ thống AI hỗ trợ soạn thảo và tra cứu văn bản cho cơ quan nhà nước.
- **Kết quả:** biết 29 loại văn bản hành chính và các thành phần thể thức, danh sách nguồn dữ liệu, 4 luật mới cần lưu ý, và mẫu 6 tác vụ cho dataset.
- **Nên biết trước:** [Định dạng dữ liệu](/du-lieu/dinh-dang), [Cần bao nhiêu dữ liệu](/du-lieu/#so-luong), [Fine-tune bằng Unsloth](/bai-toan/fine-tune).
:::

::: warning Dữ liệu có hạn dùng
Tra cứu ngày **2026-09-25**. Văn bản pháp luật thay đổi thường xuyên. Trang này không phải tư vấn pháp lý: trước khi triển khai, đối chiếu lại toàn văn trên vanban.chinhphu.vn và hỏi bộ phận pháp chế.
:::

## Văn bản hành chính theo Nghị định 30

Căn cứ chính là **Nghị định 30/2020/NĐ-CP về công tác văn thư**, ban hành ngày 05/3/2020. Tới 09/2026 chưa thấy nghị định nào thay thế: Công văn 221/CVT&LTNN-NV ngày 27/02/2026 của Cục Văn thư và Lưu trữ nhà nước vẫn lấy nghị định này làm căn cứ. Có thông tin "sẽ sửa Nghị định 30" nhưng chưa xác minh được.

**[Nguồn ngoài]** Nghị định định nghĩa văn bản hành chính là "văn bản hình thành trong quá trình chỉ đạo, điều hành, giải quyết công việc của các cơ quan, tổ chức".

### 29 loại văn bản hành chính (Điều 7)

> Nghị quyết (cá biệt), quyết định (cá biệt), chỉ thị, quy chế, quy định, thông cáo, thông báo, hướng dẫn, chương trình, kế hoạch, phương án, đề án, dự án, báo cáo, biên bản, tờ trình, hợp đồng, công văn, công điện, bản ghi nhớ, bản thỏa thuận, giấy ủy quyền, giấy mời, giấy giới thiệu, giấy nghỉ phép, phiếu gửi, phiếu chuyển, phiếu báo, thư công.

### Thành phần thể thức (Điều 8)

| Thành phần chính | Thành phần bổ sung |
| --- | --- |
| Quốc hiệu và Tiêu ngữ | Phụ lục |
| Tên cơ quan, tổ chức ban hành | Dấu chỉ độ mật, mức độ khẩn, chỉ dẫn phạm vi lưu hành |
| Số, ký hiệu văn bản | Ký hiệu người soạn thảo, số lượng bản phát hành |
| Địa danh và thời gian ban hành | Địa chỉ, thư điện tử, trang web, điện thoại, fax |
| Tên loại và trích yếu nội dung | |
| Nội dung văn bản | |
| Chức vụ, họ tên, chữ ký người có thẩm quyền | |
| Dấu, chữ ký số của cơ quan | |
| Nơi nhận | |

### Kỹ thuật trình bày chính (Phụ lục I)

| Yếu tố | Quy định |
| --- | --- |
| Khổ giấy, lề | A4. Lề trên và dưới 20–25 mm, trái 30–35 mm, phải 15–20 mm |
| Phông | Times New Roman, Unicode theo TCVN 6909:2001, màu đen |
| Quốc hiệu | IN HOA, cỡ 12–13, đứng, đậm |
| Tiêu ngữ | Cỡ 13–14, đậm, có gạch nối, dưới có đường kẻ liền |
| Số, ký hiệu | Số nhỏ hơn 10 thêm số 0 phía trước. Giữa số và ký hiệu có "/", giữa các nhóm chữ viết tắt có "-" |
| Trích yếu công văn | Sau chữ "V/v", cỡ 12–13, đặt dưới số, ký hiệu |
| Nội dung | Cỡ 13–14, canh đều hai lề, lùi đầu dòng 1 cm hoặc 1,27 cm, giãn dòng từ đơn tới 1,5 |

### Khác văn bản quy phạm pháp luật thế nào

- **Văn bản quy phạm pháp luật** (luật, nghị định, thông tư…) chứa quy tắc xử sự chung, bắt buộc chung. Luật Ban hành VBQPPL hiện hành là **Luật 64/2025/QH15** (hiệu lực 01/4/2025), đã sửa bởi **Luật 87/2025/QH15** (hiệu lực 01/7/2025).
- **Văn bản hành chính** là thứ đơn vị soạn hằng ngày để điều hành công việc: công văn, tờ trình, báo cáo…

**[Nhận định]** Hai loại này hợp với hai cách khác nhau. Văn bản quy phạm pháp luật có hiệu lực, bị sửa đổi, bị thay thế, nên đưa vào **RAG**. Văn bản hành chính là "đầu ra" đơn vị cần viết, nên dùng để **fine-tune** văn phong và bố cục.

**Nguồn:** https://vanban.chinhphu.vn/default.aspx?pageid=27160&docid=199378, https://luatvietnam.vn/hanh-chinh/cong-van-221-cvtltnn-nv-2026-tra-loi-phan-anh-kien-nghi-ve-cong-tac-van-thu-quoc-te-427073-d6.html, https://xaydungchinhsach.chinhphu.vn/quy-dinh-ve-cong-tac-van-thu-quan-ly-van-ban-the-thuc-ky-thuat-trinh-bay-van-ban-ban-sao-van-ban-viet-hoa-mau-trinh-bay-van-ban-hanh-chinh-119251023153016706.htm, https://vanban.chinhphu.vn/?pageid=27160&docid=213327, https://vanban.chinhphu.vn/?pageid=27160&docid=214592

## Lấy dữ liệu ở đâu

Nguồn tốt nhất để học văn phong là **văn bản đã ban hành của chính đơn vị** (không mật, đã lọc dữ liệu cá nhân). Dữ liệu công khai chủ yếu là văn bản quy phạm pháp luật, hợp để làm kho RAG hơn là để fine-tune.

**[Nguồn ngoài]** Nguồn chính thống:

| Nguồn | Có gì |
| --- | --- |
| [vbpl.vn](https://vbpl.vn/) | Cơ sở dữ liệu quốc gia về văn bản pháp luật (Bộ Tư pháp) |
| Cổng Pháp luật quốc gia (phapluat.gov.vn) | Ra mắt 07/11/2025, có kho dữ liệu mở và trợ lý AI pháp luật |
| [vanban.chinhphu.vn](https://vanban.chinhphu.vn/), Công báo | Metadata và PDF ký số. Nhiều PDF là **bản scan**, phải OCR mới lấy được chữ |

**[Nguồn ngoài]** Dataset trên Hugging Face (metadata tra qua API ngày 25/09/2026):

| Dataset | Nội dung | Số mẫu | License |
| --- | --- | --- | --- |
| `th1nhng0/vietnamese-legal-documents` | VBQPPL từ vbpl.vn, có **tình trạng hiệu lực** và quan hệ sửa đổi, thay thế | ~171 nghìn văn bản | CC-BY-4.0 |
| `duyet/vietnamese-legal-instruct` | Cặp hỏi đáp dạng `conversations`, dùng thẳng với Unsloth | 467.732 | CC-BY-4.0 |
| `GreenNode/zalo-ai-legal-text-retrieval-vn` | Zalo AI 2021, truy hồi điều luật | corpus 61.425 | MIT |
| `hirine/dataset-thu-tuc-hanh-chinh-5733-samples` | Thủ tục hành chính | 5.733 | CC-BY-4.0 |
| `VLSP2025-LegalSML/Public-Test` | Đề thi công khai VLSP 2025 | 440 | Không khai |

- **Không có dataset công khai nào về công văn soạn theo thể thức Nghị định 30.** Phần này đơn vị phải tự làm.
- Card của `duyet/vietnamese-legal-instruct` mô tả thứ bậc văn bản theo **Luật ban hành VBQPPL 2015, đã hết hiệu lực**. Dùng bộ này mà không lọc thì model học kiến thức cũ.

### Bản quyền

**[Nguồn ngoài]**
- Điều 15 Luật Sở hữu trí tuệ: văn bản quy phạm pháp luật và văn bản hành chính **không thuộc phạm vi bảo hộ quyền tác giả**.
- Luật 131/2025/QH15 sửa Luật Sở hữu trí tuệ, hiệu lực 01/4/2026: cho phép dùng dữ liệu đã công bố hợp pháp để huấn luyện AI, với điều kiện không ảnh hưởng bất hợp lý tới quyền của chủ thể.

**[Nhận định]** Văn bản gốc không có quyền tác giả. Nhưng phần do trang thương mại tự làm thêm (bài hỏi đáp, tóm tắt, bản dịch) thì có thể bị ràng buộc bởi điều khoản dịch vụ. Dataset crawl từ trang thương mại mà tự gắn CC-BY cũng không xóa được rủi ro này. Nên lấy từ nguồn nhà nước.

**Nguồn:** https://vbpl.vn/, https://nhandan.vn/cong-phap-luat-quoc-gia-nen-tang-phap-ly-so-hien-dai-va-toan-dien-post921622.html, https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents, https://huggingface.co/datasets/duyet/vietnamese-legal-instruct, https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn, https://huggingface.co/datasets/hirine/dataset-thu-tuc-hanh-chinh-5733-samples, https://lsvn.vn/doi-tuong-nao-khong-thuoc-pham-vi-bao-ho-quyen-tac-gia-1683458018-a130123.html, https://lsvn.vn/mot-so-diem-dang-chu-y-tai-luat-so-huu-tri-tue-sua-doi-nam-2025-a167372.html

## Luật cần tuân thủ

Chạy model local trên máy của đơn vị giúp dữ liệu không ra ngoài, nhưng **không miễn** các nghĩa vụ dưới đây.

**[Nguồn ngoài]**

| Văn bản | Hiệu lực | Điểm liên quan |
| --- | --- | --- |
| Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15, Nghị định 356/2025/NĐ-CP | 01/01/2026 | Nghị định 13/2023 hết hiệu lực từ ngày này (theo nguồn thứ cấp). Dữ liệu có họ tên, số định danh, địa chỉ phải có cơ sở pháp lý để xử lý |
| Luật Bảo vệ bí mật nhà nước 117/2025/QH15 | 01/3/2026 | Soạn thảo, lưu giữ bí mật nhà nước trên **mạng LAN độc lập**, không nối Internet. Cấm dùng AI để xâm phạm bí mật nhà nước |
| Luật Trí tuệ nhân tạo 134/2025/QH15 | 01/3/2026 | Quản lý theo 3 mức rủi ro. AI không thay thế thẩm quyền và trách nhiệm quyết định của con người. Số điều mới đối chiếu qua bản dịch |
| Công văn 557/BKHCN-CĐSQG (31/3/2025) | | Không đưa bí mật nhà nước, thông tin cá nhân nhạy cảm, tài liệu nội bộ lên chatbot. Phải kiểm tra kết quả trước khi dùng |

Một số chủ trương khuyến khích hướng đi này:
- Nghị quyết 57-NQ/TW về khoa học công nghệ và chuyển đổi số.
- Thông tư 31/2025/TT-BKHCN đưa "mô hình ngôn ngữ lớn tiếng Việt" và "trợ lý ảo" vào danh mục sản phẩm công nghệ số trọng điểm.
- Quyết định 350/QĐ-TTg (26/02/2026) về chuyển đổi số ngành Nội vụ, có trợ lý ảo AI.

**[Nhận định]** Hệ quả cho dự án:
- Máy 2×L40S xử lý tài liệu mật thì phải nằm trong mạng độc lập.
- Lọc bỏ mọi văn bản có dấu chỉ độ mật khỏi dataset. Dấu chỉ độ mật là một thành phần thể thức nên phát hiện tự động được.
- Ẩn danh dữ liệu cá nhân trước khi train.
- Giữ người duyệt và ký, lưu log, ghi rõ văn bản có AI hỗ trợ soạn.

**Nguồn:** https://chinhphu.vn/?pageid=27160&docid=214590&classid=1&typegroupid=3, https://vanban.chinhphu.vn/?pageid=27160&docid=216387, https://xaydungchinhsach.chinhphu.vn/nhung-diem-moi-quan-trong-cua-luat-bao-ve-bi-mat-nha-nuoc-co-hieu-luc-tu-1-3-2026-119260301170433415.htm, https://vanban.chinhphu.vn/?pageid=27160&docid=216334&classid=1&typegroupid=3, https://english.luatvietnam.vn/law-no-134-2025-qh15-dated-december-10-2025-of-the-national-assembly-on-artificial-intelligence-422299-doc1.html, https://xaydungchinhsach.chinhphu.vn/huong-dan-su-dung-chatbot-ai-ho-tro-can-bo-cong-chuc-nguoi-lao-dong-trong-cong-viec-119250405175138265.htm, https://luatvietnam.vn/tin-van-ban-moi/mo-hinh-ngon-ngu-lon-tieng-viet-va-tro-ly-ao-thuoc-danh-muc-san-pham-cong-nghe-so-trong-diem-186-112245-article.html, https://luatvietnam.vn/khoa-hoc/quyet-dinh-350-qd-ttg-2026-phe-duyet-de-an-chuyen-doi-so-linh-vuc-noi-vu-den-2030-427085-d1.html

## Nơi khác đã làm gì

Soạn văn bản hành chính đúng thể thức bằng AI hiện chủ yếu nằm trong sản phẩm đóng. Nghiên cứu mở tập trung vào hỏi đáp và tra cứu pháp luật.

**[Nguồn ngoài]**

| Dự án | Làm gì | Model |
| --- | --- | --- |
| Trợ lý ảo cán bộ tỉnh Hưng Yên (Viettel AI), thí điểm 2025 | Tìm kiếm văn bản, **tự sinh dự thảo đúng thể thức** | Không công bố |
| Cổng Pháp luật quốc gia (Bộ Tư pháp, FPT) | Hỏi đáp, tóm tắt văn bản pháp luật | Không công bố |
| TP.HCM cấp tài khoản cho cán bộ (319 tài khoản tới 26/8/2026) | Soạn thảo, tóm tắt báo cáo | Dịch vụ thương mại (Gemini, ChatGPT…) |
| `NaverHustQA/LawVinaLlama` | Hỏi đáp, tóm tắt pháp luật, **train bằng Unsloth** | vinallama-7b |
| VLSP 2025 LegalSLM | Hỏi đáp trắc nghiệm, suy luận pháp luật | Chủ yếu Qwen3 1.7B–4B. Continued pretraining giúp bản 4B, không giúp bản 1.7B |

**[Nhận định]** Chưa có công bố mở nào về soạn văn bản theo Nghị định 30 bằng LLM. Dataset tự xây của đơn vị chính là thứ lấp khoảng trống này.

**Nguồn:** https://nhandan.vn/thi-diem-su-dung-tro-ly-ao-ho-tro-can-bo-cong-chuc-tinh-hung-yen-post919132.html, https://fpt.com/vi/tin-tuc/tin-fpt/fpt-song-hanh-cung-bo-tu-phap-ra-mat-cong-phap-luat-quoc-gia, https://dantri.com.vn/noi-vu/cong-chuc-giam-tai-nho-tro-ly-ao-ai-20260829063246268.htm, https://huggingface.co/NaverHustQA/LawVinaLlama, https://aclanthology.org/2025.vlsp-1.21.pdf

## Thiết kế dataset

**[Nhận định]** Toàn bộ mục này là đề xuất của người viết.

Nguyên tắc: **model sinh nội dung, code lo định dạng**. Model không kiểm soát được font và lề, và quy tắc trình bày có thể đổi khi Nghị định 30 được sửa. Vì vậy model chỉ trả về các trường thể thức (dạng JSON). Một đoạn code riêng dựng file `.docx` theo Phụ lục I.

<div class="dg">
  <div class="dg-title">Luồng soạn một công văn</div>
  <div class="dg-flow">
    <div class="dg-node is-ghost">Cán bộ nhập ý chính</div>
    <div class="dg-node"><div>RAG: văn bản còn hiệu lực</div><small>tra căn cứ</small></div>
    <div class="dg-node is-main">Model fine-tune: sinh JSON theo trường thể thức</div>
    <div class="dg-node">Code: kiểm lỗi, dựng .docx</div>
    <div class="dg-node is-end">Người có thẩm quyền duyệt, ký</div>
  </div>
</div>

### Sáu tác vụ nên có

| Tác vụ | Đầu vào | Đầu ra |
| --- | --- | --- |
| 1. Soạn từ ý chính | Loại văn bản, cơ quan, nơi nhận, ý chính, căn cứ (từ RAG) | JSON: `so_ky_hieu`, `trich_yeu`, `noi_dung`, `noi_nhan`… |
| 2. Trích xuất thể thức | Văn bản (PDF, OCR) | JSON các trường thể thức |
| 3. Kiểm lỗi thể thức | Văn bản bị làm hỏng có chủ đích: thiếu nơi nhận, số không có số 0, thiếu "V/v"… | Danh sách lỗi kèm điều khoản Nghị định 30 |
| 4. Tóm tắt văn bản đến | Công văn, báo cáo | Cơ quan gửi, yêu cầu chính, thời hạn, đơn vị xử lý |
| 5. Hỏi đáp quy định | Câu hỏi và đoạn luật từ RAG | Câu trả lời có trích dẫn. Từ chối khi không có căn cứ |
| 6. Chuyển văn phong | Ghi chú, email | Đoạn văn phong hành chính, hoặc trích yếu |

Mỗi tác vụ là một mẫu hội thoại (người dùng hỏi, model trả lời), đúng [định dạng ChatML hoặc ShareGPT](/du-lieu/dinh-dang) mà Unsloth nhận.

### Quy trình làm dữ liệu

1. **Gom văn bản thật** của đơn vị, đã ban hành, cân đối giữa các loại phổ biến: công văn, tờ trình, báo cáo, kế hoạch, thông báo, quyết định cá biệt, giấy mời, biên bản.
2. **Lọc bắt buộc:**
   - bỏ văn bản có dấu chỉ độ mật;
   - ẩn danh dữ liệu cá nhân;
   - bỏ hoặc gắn nhãn văn bản hết hiệu lực;
   - sửa các mẫu nhắc luật cũ như Luật 2015, Nghị định 13/2023.
3. **Sinh thêm biến thể** bằng model lớn chạy **local**, ví dụ Qwen3.5-122B-A10B bản 4-bit, không dùng chatbot bên ngoài. Cán bộ văn thư duyệt từng mẫu. Xem [Sinh dữ liệu tổng hợp](/du-lieu/synthetic).
4. **Tách tập đánh giá** theo loại văn bản và theo thời gian:
   - chấm thể thức bằng code (regex số ký hiệu, trường bắt buộc);
   - cán bộ chấm nội dung;
   - thêm đề VLSP 2025 (440 mẫu) để chắc model không quên kiến thức pháp luật chung.
5. Rà lại bằng [Checklist chuẩn bị dữ liệu](/du-lieu/checklist) trước khi train.

::: warning Lỗi thường gặp
- Dùng dataset công khai nhắc luật đã hết hiệu lực mà không lọc.
- Để model tự "nhớ" số hiệu, ngày ban hành của văn bản pháp luật. Model sẽ bịa số hiệu nghe rất thật. Luôn lấy từ RAG.
- Đưa văn bản nội bộ lên chatbot bên ngoài để sinh dữ liệu, trái Công văn 557/BKHCN-CĐSQG.
:::

**Nguồn:** https://xaydungchinhsach.chinhphu.vn/huong-dan-su-dung-chatbot-ai-ho-tro-can-bo-cong-chuc-nguoi-lao-dong-trong-cong-viec-119250405175138265.htm, https://huggingface.co/datasets/VLSP2025-LegalSML/Public-Test, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Đọc tiếp

- [Host model trên máy chủ](/bai-toan/hosting) — trang kế tiếp: đem model đã train đi phục vụ người dùng.
- [Định dạng dữ liệu](/du-lieu/dinh-dang) — cấu trúc file dataset mà Unsloth nhận.
- [Ứng dụng RAG](/ung-dung-rag) — dựng kho tra cứu văn bản pháp luật còn hiệu lực.
- [Đánh giá và overfitting](/fine-tuning/danh-gia) — đo model sau khi train.
