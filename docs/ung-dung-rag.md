---
title: Ứng dụng RAG
description: RAG và fine-tuning theo FAQ của Unsloth, cùng kịch bản chatbot RAG nội bộ cho ngân hàng dùng Unsloth API, LAN access và fine-tune embedding.
---

# Ứng dụng RAG

::: info Lưu ý về nội dung
Phần lớn trang này là nhận định của người viết. Chỗ nào lấy từ docs Unsloth sẽ có dòng Nguồn.
:::

## RAG là gì

[Nhận định] RAG — Retrieval-Augmented Generation (sinh câu trả lời có truy xuất tài liệu) — là cách cho LLM (mô hình ngôn ngữ lớn) trả lời dựa trên tài liệu của bạn mà không cần train lại model.

[Nhận định] RAG ghép một hệ thống tìm kiếm với LLM. Luồng chạy gồm ba bước:

1. Người dùng đặt câu hỏi.
2. Hệ thống tìm các đoạn tài liệu liên quan trong kho. Việc tìm thường dùng vector embedding, tức vector số biểu diễn ý nghĩa của câu.
3. Hệ thống đưa các đoạn đó vào prompt, và LLM trả lời dựa trên chúng.

Khi tài liệu thay đổi, bạn chỉ cần cập nhật kho, không phải train lại model.

Docs Unsloth không định nghĩa RAG chi tiết. FAQ chỉ nhắc điểm mạnh của RAG là **truy cập thông tin mới từ cơ sở dữ liệu bên ngoài**. Trang embedding thì nói fine-tune embedding model giúp cải thiện retrieval (khâu truy xuất) và RAG.

::: tip Kiến thức nền
Embedding model là gì, khác LLM thế nào: xem [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh). Token và context window (giới hạn độ dài đầu vào): xem [Token & context](/kien-thuc-nen/token-va-context).
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/basics/embedding-finetuning

## RAG vs fine-tuning

RAG và fine-tuning giải quyết hai việc khác nhau: RAG đưa thông tin mới vào lúc trả lời, còn fine-tuning thay đổi chính model. Mục này trình bày quan điểm của FAQ Unsloth trước, rồi đến bảng so sánh của người viết.

### FAQ của Unsloth nói gì

FAQ của Unsloth có các ý sau.

**Quan hệ giữa hai cách:**

- Fine-tuning có thể đưa kiến thức và hành vi **trực tiếp vào model**, theo cách RAG không làm được. Trong thực tế, **kết hợp cả hai** cho kết quả tốt nhất: chính xác hơn và ít hallucination (bịa thông tin) hơn.
- "Fine-tuning làm được gần như mọi thứ RAG làm, nhưng không ngược lại." Dù vậy, RAG vẫn có lợi thế ở việc truy cập thông tin cập nhật từ database bên ngoài.

**Những quan niệm FAQ bác bỏ:**

- Fine-tune **có thể** thêm kiến thức mới cho model. Docs bác bỏ quan niệm "fine-tune không thêm kiến thức".
- RAG không phải lúc nào cũng tốt hơn. Các nhận định "RAG luôn hơn" thường đến từ fine-tune cấu hình sai, ví dụ LoRA parameter sai hoặc train chưa đủ.

**Lợi ích của fine-tune mà docs liệt kê:**

- Chuyên sâu vào tác vụ.
- Không phụ thuộc hệ thống retrieval lúc inference.
- Trả lời nhanh hơn vì bỏ bước truy xuất.
- Kiểm soát giọng văn và tuân thủ quy định.
- Làm phương án dự phòng khi retrieval trả sai.

**Lý do nên kết hợp:**

- Fine-tune giỏi các tác vụ và định dạng chuyên biệt.
- RAG giữ thông tin luôn mới mà không cần train lại cho mỗi lần có dữ liệu mới.
- Kết hợp giúp giảm tổng chi phí tính toán.

Các ví dụ ứng dụng fine-tune trong FAQ có **phân tích cảm xúc tin tài chính** và **chatbot chăm sóc khách hàng** theo phong cách, thuật ngữ của công ty.

[Nhận định] FAQ này là tài liệu của một công ty bán công cụ fine-tune, nên lập luận nghiêng về fine-tuning. Bạn nên đọc kèm bảng so sánh bên dưới.

### Bảng so sánh

[Nhận định] Toàn bộ bảng là đánh giá của người viết, không lấy từ docs. Cột "Gợi ý" cho biết cách nào hợp hơn với từng tiêu chí.

| Tiêu chí | RAG | Fine-tuning | Gợi ý |
| --- | --- | --- | --- |
| Dữ liệu thay đổi thường xuyên (lãi suất, biểu phí, quy trình mới) | Cập nhật kho là xong | Phải train lại | RAG |
| Trích dẫn nguồn chính xác (điều khoản, số văn bản) | Trả được đoạn gốc kèm tài liệu | Model nhớ "mờ", khó chỉ ra nguồn | RAG |
| Phong cách, giọng văn, định dạng trả lời cố định | Chỉ điều khiển qua prompt | Học ổn định vào model | Fine-tune |
| Thuật ngữ nội bộ, viết tắt đặc thù | Phụ thuộc chất lượng retrieval | Học được | Kết hợp |
| Chi phí ban đầu | Cần vector DB, pipeline chunking | Cần GPU, dữ liệu train sạch | Tùy đội |
| Chi phí vận hành | Mỗi câu hỏi tốn thêm bước tìm kiếm và prompt dài hơn | Prompt ngắn hơn | — |
| Phân quyền theo người dùng | Lọc tài liệu theo quyền trước khi đưa vào prompt | Kiến thức đã nằm trong trọng số, không phân quyền được | RAG |
| Xóa dữ liệu (yêu cầu gỡ thông tin) | Xóa khỏi kho | Phải train lại | RAG |
| Độ trễ | Cao hơn (thêm retrieval) | Thấp hơn | Fine-tune |

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me

## Kịch bản: chatbot RAG nội bộ cho ngân hàng

[Nhận định] Mục này dựng một ví dụ cụ thể để thấy các phần của Unsloth ghép vào một hệ thống RAG thật như thế nào.

[Nhận định] Bối cảnh giả định: một ngân hàng muốn có chatbot trả lời nhân viên về quy trình, sản phẩm và văn bản nội bộ. Dữ liệu nhạy cảm nên **không gửi ra API đám mây**. Mọi model chạy trên máy chủ trong mạng nội bộ. Backend viết bằng FastAPI.

```mermaid
flowchart LR
  U["Nhân viên<br/>(trình duyệt nội bộ)"] --> BE["FastAPI backend<br/>xác thực + phân quyền"]
  BE --> EMB["Embedding model<br/>đã fine-tune bằng Unsloth"]
  EMB --> VDB["Vector DB<br/>(pgvector / FAISS)"]
  DOC["Tài liệu nội bộ<br/>quy trình, sản phẩm"] --> ING["Pipeline nạp tài liệu<br/>chia đoạn + embed"]
  ING --> VDB
  VDB --> BE
  BE --> LLM["Unsloth API<br/>/v1/chat/completions<br/>(máy GPU trong LAN)"]
  LLM --> BE
  BE --> LOG["Audit log"]
  BE --> U
```

[Nhận định] Một câu hỏi đi qua hệ thống theo thứ tự sau:

1. FastAPI nhận câu hỏi.
2. Kiểm tra quyền của người dùng.
3. Embed câu hỏi.
4. Tìm top-k đoạn trong vector DB, chỉ trong những tài liệu người đó được xem.
5. Ghép các đoạn vào prompt.
6. Gọi Unsloth API.
7. Trả lời kèm danh sách tài liệu nguồn.
8. Ghi audit log.

pgvector và FAISS nằm trong danh sách công cụ mà docs nói embedding model của Unsloth dùng được. Việc chọn cái nào là [Nhận định].

**Nguồn:** https://unsloth.ai/docs/basics/embedding-finetuning, https://unsloth.ai/docs/basics/api

## (a) Chạy LLM local qua API của Unsloth

Phần này biến máy GPU trong mạng nội bộ thành một "máy chủ LLM" mà backend FastAPI gọi vào, giống như gọi API của OpenAI.

Unsloth mở model đã load (kể cả GGUF) thành một **API có xác thực**. Bên dưới, API này chạy bằng `llama-server`. Cùng một port phục vụ hai kiểu API:

| Endpoint | Tương thích |
| --- | --- |
| `POST /v1/chat/completions` (và `/v1/responses`) | OpenAI Chat Completions — dùng OpenAI SDK |
| `POST /v1/messages` | Anthropic Messages API |
| `GET /v1/models` | Liệt kê model đang load |

Mọi request phải có header `Authorization: Bearer sk-unsloth-…`. Về API key:

- Tạo key ở **Settings → API**.
- Key chỉ hiện một lần; Unsloth chỉ lưu hash.
- Key đã bị revoke sẽ nhận về `401 Unauthorized`.

Lệnh dưới load model GGUF và mở ra mạng nội bộ, để máy chạy FastAPI gọi vào được:

```bash
# Allow LAN devices to connect
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  -H 0.0.0.0 \
  -p 8888
```

Muốn tắt hẳn tool phía server (web search, chạy code), dùng lệnh:

```bash
# Explicitly disable tools
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  --disable-tools
```

Trang API ghi: khi `unsloth run` bind `0.0.0.0`, tool **tắt mặc định**. Chính sách này áp ở mức process, nên request không thể bật lại bằng `enable_tools=true`.

::: warning Docs chưa thống nhất
Các trang docs ghi khác nhau về việc tool phía server (web search, Python, terminal) **bật hay tắt mặc định** khi bạn mở Unsloth ra mạng:

| Thông số | Trang API ([api](https://unsloth.ai/docs/basics/api)) | Trang LAN ([lan](https://unsloth.ai/docs/basics/lan)) | Trang Cloudflare ([remote access](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth)) |
| --- | --- | --- | --- |
| Lệnh được nói tới | `unsloth run` | Trang hướng dẫn `unsloth studio -H 0.0.0.0`; mục Security không ghi rõ lệnh | Trang hướng dẫn `unsloth studio --secure` / `--cloudflare`; mục Security không ghi rõ lệnh |
| Tool mặc định | Bind `127.0.0.1`: bật; bind `0.0.0.0` / không phải loopback: **tắt** | "on by default" | "on by default" |

Vì vậy, hãy luôn truyền `--disable-tools` tường minh thay vì dựa vào mặc định.
:::

Kiểm tra kết nối từ máy backend:

```bash
curl http://localhost:8888/v1/models \
  -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx"
```

::: warning Docs chưa thống nhất
Trang [api](https://unsloth.ai/docs/basics/api) ghi port mặc định của Unsloth API khác nhau ở các chỗ. Port này ảnh hưởng tới `base_url` bạn cấu hình:

| Chỗ trong docs | Giá trị |
| --- | --- |
| Quickstart ([api](https://unsloth.ai/docs/basics/api)) | `http://localhost:PORT` (không nêu số) |
| Mục Endpoints ([api](https://unsloth.ai/docs/basics/api)) | "typically `http://localhost:8000` or `http://localhost:8888`" |
| Mục kết nối từ máy khác và Troubleshooting ([api](https://unsloth.ai/docs/basics/api)) | `http://127.0.0.1:8888`, `http://localhost:8888` |

Cách an toàn: đặt `-p` tường minh (như lệnh ở trên) và đọc URL endpoint mà `unsloth run` in ra console.
:::

Phía FastAPI chỉ cần dùng OpenAI SDK, với hai thiết lập:

- `base_url` trỏ tới địa chỉ LAN của máy GPU, dạng `http://<IP máy GPU>:8888/v1`.
- API key `sk-unsloth-…`.

Cách gọi giống ví dụ OpenAI SDK ở trang [Export & deploy](/export-deploy).

[Nhận định] Nên để FastAPI là điểm duy nhất giữ API key. Trình duyệt của nhân viên không gọi thẳng vào Unsloth.

::: warning LAN access không mã hóa
Docs ghi rõ: LAN dùng **HTTP thường**, nên ai cùng phân đoạn mạng đều đọc được traffic. Nếu bạn đặt Unsloth sau reverse proxy của mình, biến `UNSLOTH_STUDIO_TRUST_FORWARDED=1` cho phép Unsloth nhận header `X-Forwarded-For`.

[Nhận định] Với ngân hàng, nên làm các việc sau: đặt máy GPU trong VLAN riêng; chỉ cho IP của FastAPI truy cập port 8888 (bằng firewall); và/hoặc đặt reverse proxy có TLS phía trước. **Không** dùng `--secure` / Cloudflare tunnel cho dữ liệu ngân hàng, vì traffic đi qua hạ tầng bên thứ ba và URL là công khai.
:::

::: info Môi trường không có internet
Khi bind wildcard (`0.0.0.0`), Unsloth tự kiểm tra xem port có lộ ra internet không, bằng cách gọi `ifconfig.me` và `check-host.net`. Đặt `UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK=1` để bỏ bước này. [Nhận định] Trong mạng ngân hàng cách ly, nên bật biến này để máy không cố gọi ra ngoài.
:::

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/basics/lan

## (b) Fine-tune embedding tiếng Việt bằng Unsloth

[Nhận định] Embedding model quyết định RAG tìm đúng đoạn tài liệu hay không. Fine-tune embedding giúp model hiểu "giống nhau" theo đúng nghĩa mà bài toán của bạn cần.

Theo docs, fine-tune embedding model giúp vector "hiểu" đúng kiểu tương đồng mà bài toán cần, nhờ đó cải thiện search và RAG trên dữ liệu riêng. Unsloth hỗ trợ các điểm sau.

**Tốc độ và bộ nhớ.** Unsloth train embedding, classifier, BERT, reranker nhanh hơn, ít hơn 20% bộ nhớ, và context dài gấp 2 so với các bản Flash Attention 2 khác. Con số tốc độ khác nhau tùy chỗ trong docs, đều so với SentenceTransformers + Flash Attention 2:

- "~1.8-3.3x" ở đoạn mở đầu và mục Benchmarks.
- "1.8x to 2.6x" cho QLoRA 4-bit.
- "1.2x to 3.3x" cho LoRA 16-bit.

Với EmbeddingGemma-300M, QLoRA cần 3GB VRAM (VRAM = bộ nhớ card đồ họa), LoRA cần 6GB.

**Cách train.** Unsloth hỗ trợ LoRA, QLoRA hoặc full fine-tuning. Hỗ trợ tốt nhất là cho model `SentenceTransformer` encoder-only có `modules.json`. Model không có `modules.json` chỉ được hỗ trợ hạn chế: Unsloth tự gán pooling mặc định, nên bạn cần kiểm tra lại output embedding.

**Class trung tâm** là `FastSentenceTransformer`.

Model được docs liệt kê (danh sách không đầy đủ):

```
Alibaba-NLP/gte-modernbert-base
BAAI/bge-large-en-v1.5
BAAI/bge-m3
BAAI/bge-reranker-v2-m3
Qwen/Qwen3-Embedding-0.6B
answerdotai/ModernBERT-base
answerdotai/ModernBERT-large
google/embeddinggemma-300m
intfloat/e5-large-v2
intfloat/multilingual-e5-large-instruct
mixedbread-ai/mxbai-embed-large-v1
sentence-transformers/all-MiniLM-L6-v2
sentence-transformers/all-mpnet-base-v2
Snowflake/snowflake-arctic-embed-l-v2.0
```

[Nhận định] Docs **không** nói model nào hỗ trợ tiếng Việt. Nhìn theo tên, các ứng viên đa ngôn ngữ là `BAAI/bge-m3` và `intfloat/multilingual-e5-large-instruct`, cùng Qwen3-Embedding. Khả năng tiếng Việt của chúng cần kiểm tra lại trên model card và bằng đánh giá trên dữ liệu thật.

### Lưu và load

Có bốn hàm để lưu model, khác nhau ở chỗ lưu adapter hay model đã merge, và lưu ra thư mục hay đẩy lên Hugging Face:

| Hàm | Tác dụng |
| --- | --- |
| `save_pretrained()` | Lưu LoRA adapter ra thư mục |
| `save_pretrained_merged()` | Lưu model đã merge ra thư mục |
| `push_to_hub()` | Đẩy LoRA adapter lên Hugging Face |
| `push_to_hub_merged()` | Đẩy model đã merge lên Hugging Face |

Khi load để inference, bạn **bắt buộc** truyền `for_inference=True`:

```python
model = FastSentenceTransformer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    for_inference=True,
)
```

Ví dụ dùng model đã fine-tune, chép nguyên văn từ docs:

```python
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("<your-unsloth-finetuned-model")

query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# 2. Encode via encode_query and encode_document to automatically use the right prompts, if needed
query_embedding = model.encode_query(query)
document_embedding = model.encode_document(documents)
print(query_embedding.shape, document_embedding.shape)

# 3. Compute similarity, e.g. via the built-in similarity helper function
similarity = model.similarity(query_embedding, document_embedding)
print(similarity)
```

::: warning Docs chưa thống nhất
Dòng `SentenceTransformer("<your-unsloth-finetuned-model")` được chép nguyên văn từ [embedding-finetuning](https://unsloth.ai/docs/basics/embedding-finetuning). Placeholder này mở bằng `<` nhưng **thiếu `>`** đóng. Đây là chỗ bạn thay bằng tên repo Hugging Face hoặc đường dẫn thư mục model đã fine-tune của mình. Một ví dụ load khác trong cùng trang dùng tên repo thật (`"sentence-transformers/all-MiniLM-L6-v2"`).
:::

Model sau fine-tune dùng được với nhiều công cụ:

- Thư viện: transformers, sentence-transformers, LangChain.
- Vector DB và tìm kiếm: Weaviate, pgvector, FAISS.
- Serving: Text Embeddings Inference (TEI), vLLM, llama.cpp.
- Framework RAG bất kỳ.

::: info Code train và định dạng dữ liệu không có trong trang docs
Trang embedding **không** có code vòng train và không mô tả định dạng dataset (ví dụ dạng cặp câu hỏi–đoạn văn). Phần này nằm trong các notebook (EmbeddingGemma 300M, Qwen3-Embedding 4B/0.6B, BGE M3, All-MiniLM-L6-v2, ModernBERT) — cần kiểm tra lại trực tiếp trong notebook.

[Nhận định] Với ngân hàng, dữ liệu train hợp lý là các cặp (câu hỏi thực tế của nhân viên, đoạn quy trình trả lời đúng), lấy từ log hỏi đáp nội bộ đã ẩn danh hóa.
:::

**Nguồn:** https://unsloth.ai/docs/basics/embedding-finetuning

## (c) Tùy chọn: fine-tune LLM cho giọng văn và định dạng

Bước này không bắt buộc. [Nhận định] Nó dùng khi bạn muốn chatbot luôn trả lời theo một khuôn cố định mà prompt không giữ được ổn định.

FAQ nói fine-tune giúp kiểm soát giọng văn, bám thương hiệu và **tuân thủ yêu cầu quy định**. FAQ cũng khuyên bắt đầu bằng QLoRA (LoRA trên model lượng tử hóa 4-bit).

[Nhận định] Trong kịch bản này, bạn chỉ nên fine-tune LLM để cố định **cách trả lời**: luôn trích nguồn, giữ cấu trúc câu trả lời, từ chối câu hỏi ngoài phạm vi, xưng hô chuẩn. Còn **kiến thức nghiệp vụ** thay đổi liên tục nên để RAG lo.

Sau khi train, bạn xuất model sang GGUF để chạy trên Unsloth API. Các bước xuất xem [Export & deploy](/export-deploy). Việc `unsloth run` load file GGUF cục bộ (thay vì repo Hugging Face) cần kiểm tra lại, vì trang API chỉ đưa ví dụ tên repo. Quy trình train xem [Fine-tuning](/fine-tuning/).

::: tip Kiến thức nền
LoRA/QLoRA: xem [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora).
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/basics/api

## Rủi ro và lưu ý

Mục này chia làm hai phần: các cảnh báo có sẵn trong docs Unsloth, và các rủi ro riêng khi áp dụng cho ngân hàng.

::: warning Từ docs Unsloth
- **Tool phía server chạy dưới quyền user hệ điều hành.** Ai có API key và truy cập được server thì có thể chạy code trên máy. Khi mở ra mạng, hãy dùng `--disable-tools` và giữ kín key.
- **API key giống như mật khẩu**: ai có key và có đường mạng tới Unsloth đều gửi request vào model được.
- LAN dùng HTTP không mã hóa.
- Qua Cloudflare quick tunnel, server-sent events (streaming) không hoạt động. Bạn phải đặt `stream: false`.
- Câu trả lời bị cắt: kiểm tra **Context used** trong API monitor. Nếu gần 100% hoặc stop reason là `length`, nghĩa là context đã đầy.
:::

::: warning [Nhận định] Riêng cho ngân hàng
- **Rò rỉ qua retrieval**: nếu không lọc theo quyền trước khi đưa đoạn văn vào prompt, người dùng có thể hỏi ra tài liệu họ không được xem. Phân quyền phải làm ở tầng FastAPI hoặc vector DB, không trông vào LLM.
- **Prompt injection** (chèn lệnh độc qua nội dung tài liệu): tài liệu nạp vào kho có thể chứa câu điều khiển model. Hãy giữ tool tắt, và không cho LLM gọi hành động nghiệp vụ trực tiếp.
- **Hallucination**: bắt buộc hiển thị nguồn trích dẫn và cho phép nhân viên mở tài liệu gốc. Câu trả lời chỉ mang tính tham khảo.
- **Dữ liệu train**: dữ liệu dùng để fine-tune embedding hoặc LLM phải được ẩn danh hóa. Thông tin khách hàng một khi đã vào trọng số model thì khó gỡ.
- **Tuân thủ**: cần đối chiếu với quy định nội bộ và pháp luật hiện hành về bảo vệ dữ liệu cá nhân, an toàn thông tin ngân hàng — cần kiểm tra lại với bộ phận pháp chế.
- **Audit log**: log câu hỏi, tài liệu được truy xuất, câu trả lời và người hỏi. Bản thân log cũng là dữ liệu nhạy cảm cần bảo vệ.
- **Đánh giá**: trước mỗi lần đổi model, chạy một bộ câu hỏi kiểm thử tiếng Việt có đáp án chuẩn. Bộ này đo cả retrieval (đã tìm đúng đoạn chưa) lẫn câu trả lời.
:::

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/basics/lan, https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth
