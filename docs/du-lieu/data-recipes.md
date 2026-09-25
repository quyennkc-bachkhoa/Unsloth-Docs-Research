---
title: Data Recipes
description: "Biến tài liệu thành dataset bằng workflow dạng node trong Studio: các bước, các loại block và cách tham chiếu giữa block."
---

# Data Recipes trong Studio

Data Recipes biến tài liệu của bạn (PDF, CSV...) thành dataset dùng được, hoặc thành dataset tổng hợp. Bạn dựng quy trình này trên một workflow dạng đồ thị node (graph-node) và chỉnh sửa trực quan. Tính năng chạy trên nền NVIDIA NeMo [Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner). Recipe được lưu cục bộ trong trình duyệt, và có thể export hoặc import để chia sẻ.

::: warning Docs chưa thống nhất: Data Recipes nhận loại file nào
Các trang mô tả loại file đầu vào khác nhau:
- "upload documents like PDFs or CSVs files" — [Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe), [Studio](https://unsloth.ai/docs/new/studio/start), [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- Block Seed: dữ liệu từ Hugging Face, "local structured files" hoặc "unstructured documents that get chunked into rows" (không liệt kê định dạng cụ thể) — [Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe)
- "upload any unstructured or structured data into Unsloth Studio's Data Recipes" — [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- Tab Local ở bước Dataset khi train (không phải Data Recipes) nhận `PDF`, `DOCX`, `JSONL`, `JSON`, `CSV`, `Parquet` — [Studio](https://unsloth.ai/docs/new/studio/start)

Data Recipes có nhận DOCX, JSONL, Parquet hay không: cần kiểm tra lại. Trang Data Recipes cũng chưa hoàn chỉnh: còn dòng nháp "need to add more here", và mục "Run the full dataset build" để trống.
:::

## Các bước

<div class="dg">
<div class="dg-flow">
<div class="dg-node">Mở trang Recipes</div>
<div class="dg-node">Tạo mới / mở recipe</div>
<div class="dg-node">Thêm block<small>Seed, LLM, Expression, Validator…</small></div>
<div class="dg-node">Validate</div>
<div class="dg-node is-main">Preview<small>vài dòng mẫu</small></div>
<div class="dg-node" data-e="Ổn">Chạy full dataset</div>
<div class="dg-node is-end">Chọn dataset trong Unsloth<small>để fine-tune</small></div>
<div class="dg-back" style="grid-column: 3 / 6"><span>Chưa ổn: sửa prompt, seed, validator</span></div>
</div>
</div>

1. Mở trang recipes.
2. Tạo recipe mới hoặc mở recipe có sẵn. Có ba lựa chọn:
   - **Start Empty**: tự dựng nhanh.
   - **Start from Learning Recipe**: học từ ví dụ mẫu, nhanh nhất cho người mới.
   - **mở recipe đã lưu**.
3. Thêm block để định nghĩa workflow: chọn block từ block sheet, cấu hình trong dialog, rồi nối các block trên canvas.
4. Bấm **Validate** để bắt lỗi cấu hình sớm.
5. Chạy **preview** để xem nhanh các dòng mẫu và phân tích.
6. Khi recipe đã ổn, chạy **full dataset build**.
7. Theo dõi tiến độ và kết quả trên graph hoặc ở view **Executions**.
8. Chọn dataset kết quả trong Unsloth và fine-tune.

Preview dùng để thử và sửa nhanh. Full run tạo ra một dataset lưu cục bộ, dataset này xuất hiện trong bộ chọn dataset local của Studio. Bạn cũng có thể publish nó lên một repo Hugging Face.

## Các loại block

Mỗi recipe được ghép từ các block. Mỗi loại block đảm nhận một vai trò:

| Block | Vai trò |
| --- | --- |
| **Seed** | Dữ liệu đầu vào: từ Hugging Face, file có cấu trúc local, hoặc tài liệu không cấu trúc được chia (chunk) thành các dòng |
| **LLM + Models** | Provider, cấu hình model, các block sinh bằng LLM, tool profile dùng chung |
| **Expression** | Biến đổi dựa trên Jinja2, không cần gọi LLM |
| **Validators** | Lọc code sinh ra bị lỗi bằng linter có sẵn cho Python, SQL, JavaScript/TypeScript |
| **Samplers** | Cột xác định (deterministic) như category, subcategory |
| **Tool Profiles** | Cấp quyền dùng tool qua MCP cho một hoặc nhiều block LLM (ví dụ tra tài liệu code qua `Context7`) |

Cấu hình model chia làm hai lớp:

- **Model provider**: endpoint và thông tin xác thực.
- **Model Config**: tên model và tham số inference.

Data Recipes hoạt động với provider hosted, endpoint tự host, `vLLM`, `llama.cpp`, hoặc bất kỳ API tương thích OpenAI nào chạy ngoài Unsloth. Một recipe có thể dùng nhiều model cho các bước khác nhau.

Có bốn loại block LLM:

| Block | Đầu ra | Phù hợp cho |
| --- | --- | --- |
| LLM Text | Văn bản tự do | Chỉ dẫn, giải thích, hội thoại, mô tả |
| LLM Structured | JSON | Đầu ra cần trường cố định, cấu trúc dự đoán được |
| LLM Code | Code | Sinh Python, SQL, TypeScript... |
| LLM Judge | Điểm đánh giá | Chấm đầu ra theo một hoặc nhiều tiêu chí tự định nghĩa |

## Tham chiếu giữa các block

Hầu hết block sinh dữ liệu sẽ trở thành **tham chiếu** cho các block sau. Nghĩa là giá trị được tạo một lần, rồi dùng lại trong prompt, expression, structured output và bước validate. Một số ví dụ:

- Một block category tên `domain`.
- Cột của seed data (cột dataset HF, cột CSV) dùng thẳng trong prompt.
- Trường do LLM Structured sinh ra dùng cho prompt ở bước sau.
- Block expression ghép các giá trị trước đó mà không cần gọi model.

Cú pháp Jinja docs đưa ra:

```
{{ domain }}
{{customer.first_name}}
{{customer.first_name}} {{customer.last_name}}
{% if condition %}...{% endif %}
```

**Nguồn:** https://unsloth.ai/docs/new/studio/data-recipe, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide
