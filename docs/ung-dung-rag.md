---
title: Ứng dụng RAG
description: RAG và fine-tuning theo FAQ của Unsloth, cùng kịch bản chatbot RAG nội bộ cho ngân hàng dùng Unsloth API, LAN access và fine-tune embedding.
---

# Ứng dụng RAG

::: info Lưu ý về nội dung
Trang này chủ yếu là nhận định của người viết; chỗ nào có trong docs Unsloth sẽ có Nguồn.
:::

## RAG là gì

[Nhận định] RAG — Retrieval-Augmented Generation (sinh câu trả lời có truy xuất tài liệu) — là cách ghép một hệ thống tìm kiếm với LLM (mô hình ngôn ngữ lớn). Khi người dùng hỏi, hệ thống tìm các đoạn tài liệu liên quan trong kho (thường bằng vector embedding — vector số biểu diễn ý nghĩa của câu), rồi đưa các đoạn đó vào prompt để LLM trả lời dựa trên chúng. Model không cần train lại khi tài liệu thay đổi; chỉ cần cập nhật kho.

Docs Unsloth không định nghĩa RAG chi tiết. FAQ chỉ nhắc RAG mạnh ở chỗ **truy cập thông tin mới từ cơ sở dữ liệu bên ngoài**, và trang embedding nói fine-tune embedding model giúp cải thiện retrieval (khâu truy xuất) và RAG.

::: tip Kiến thức nền
Embedding model là gì, khác LLM thế nào: xem [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh). Token và context window (giới hạn độ dài đầu vào): xem [Token & context](/kien-thuc-nen/token-va-context).
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/basics/embedding-finetuning

## RAG vs fine-tuning

### FAQ của Unsloth nói gì

- Fine-tuning có thể đưa kiến thức và hành vi **trực tiếp vào model** theo cách RAG không làm được; trong thực tế **kết hợp cả hai** cho kết quả tốt nhất (chính xác hơn, ít hallucination — bịa thông tin — hơn).
- "Fine-tuning làm được gần như mọi thứ RAG làm, nhưng không ngược lại." RAG vẫn có lợi thế ở việc truy cập thông tin cập nhật từ database bên ngoài.
- Fine-tune **có thể** thêm kiến thức mới cho model (docs bác bỏ quan niệm "fine-tune không thêm kiến thức").
- RAG không phải lúc nào cũng tốt hơn; các nhận định "RAG luôn hơn" thường đến từ fine-tune cấu hình sai (LoRA parameter sai, train chưa đủ).
- Lợi ích fine-tune docs liệt kê: chuyên sâu tác vụ, không phụ thuộc hệ thống retrieval lúc inference, trả lời nhanh hơn vì bỏ bước truy xuất, kiểm soát giọng văn/tuân thủ quy định, làm phương án dự phòng khi retrieval trả sai.
- Lý do nên kết hợp: fine-tune giỏi tác vụ/định dạng chuyên biệt; RAG giữ thông tin luôn mới mà không cần train lại cho mỗi dữ liệu mới; giảm tổng chi phí tính toán.
- Ví dụ ứng dụng fine-tune trong FAQ có **phân tích cảm xúc tin tài chính** và **chatbot chăm sóc khách hàng** theo phong cách, thuật ngữ công ty.

[Nhận định] FAQ này là tài liệu của một công ty bán công cụ fine-tune, nên lập luận nghiêng về fine-tuning. Nên đọc kèm bảng so sánh dưới.

### Bảng so sánh

[Nhận định] Toàn bộ bảng là đánh giá của người viết, không lấy từ docs.

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

[Nhận định] Bối cảnh giả định: ngân hàng muốn chatbot trả lời nhân viên về quy trình, sản phẩm, văn bản nội bộ. Dữ liệu nhạy cảm nên **không gửi ra API đám mây**; mọi model chạy trên máy chủ trong mạng nội bộ. Backend viết bằng FastAPI.

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

[Nhận định] Luồng một câu hỏi: FastAPI nhận câu hỏi → kiểm tra quyền người dùng → embed câu hỏi → tìm top-k đoạn trong vector DB (chỉ trong tài liệu người đó được xem) → ghép các đoạn vào prompt → gọi Unsloth API → trả lời kèm danh sách tài liệu nguồn → ghi audit log.

Các tên công cụ pgvector, FAISS nằm trong danh sách nơi docs nói embedding model của Unsloth dùng được; việc chọn cái nào là [Nhận định].

**Nguồn:** https://unsloth.ai/docs/basics/embedding-finetuning, https://unsloth.ai/docs/basics/api

## (a) Chạy LLM local qua API của Unsloth

Unsloth mở model đã load (kể cả GGUF) thành **API có xác thực**, chạy bằng `llama-server` bên dưới. Cùng một port phục vụ hai kiểu:

| Endpoint | Tương thích |
| --- | --- |
| `POST /v1/chat/completions` (và `/v1/responses`) | OpenAI Chat Completions — dùng OpenAI SDK |
| `POST /v1/messages` | Anthropic Messages API |
| `GET /v1/models` | Liệt kê model đang load |

Mọi request phải có header `Authorization: Bearer sk-unsloth-…`. Tạo key ở **Settings → API**; key chỉ hiện một lần, Unsloth chỉ lưu hash; key bị revoke trả về `401 Unauthorized`.

Load model GGUF và mở ra mạng nội bộ để máy chạy FastAPI gọi vào:

```bash
# Allow LAN devices to connect
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  -H 0.0.0.0 \
  -p 8888
```

Tắt hẳn tool phía server (web search, chạy code):

```bash
# Explicitly disable tools
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  --disable-tools
```

Theo docs, `unsloth run` bind `0.0.0.0` thì tool **tắt mặc định**; chính sách này áp ở mức process, request không thể bật lại bằng `enable_tools=true`.

Kiểm tra từ máy backend:

```bash
curl http://localhost:8888/v1/models \
  -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx"
```

Phía FastAPI chỉ cần dùng OpenAI SDK với `base_url` trỏ tới địa chỉ LAN của máy GPU (dạng `http://<IP máy GPU>:8888/v1`) và API key `sk-unsloth-…`. Cách gọi giống ví dụ OpenAI SDK ở trang [Export & deploy](/export-deploy).

[Nhận định] Nên để FastAPI là điểm duy nhất giữ API key; trình duyệt nhân viên không gọi thẳng Unsloth.

::: warning LAN access không mã hóa
Docs ghi rõ: LAN là **HTTP thường**, ai cùng phân đoạn mạng đều đọc được traffic. Unsloth có biến `UNSLOTH_STUDIO_TRUST_FORWARDED=1` để nhận `X-Forwarded-For` khi đặt sau reverse proxy của bạn.

[Nhận định] Với ngân hàng: đặt máy GPU trong VLAN riêng, chỉ cho IP của FastAPI truy cập port 8888 (firewall), và/hoặc đặt reverse proxy có TLS phía trước. **Không** dùng `--secure` / Cloudflare tunnel cho dữ liệu ngân hàng vì traffic đi qua hạ tầng bên thứ ba và URL công khai.
:::

::: info Môi trường không có internet
Khi bind wildcard (`0.0.0.0`), Unsloth tự kiểm tra port có lộ ra internet không bằng cách gọi `ifconfig.me` và `check-host.net`. Đặt `UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK=1` để bỏ bước này. [Nhận định] Trong mạng ngân hàng cách ly, nên bật biến này để máy không cố gọi ra ngoài.
:::

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/basics/lan

## (b) Fine-tune embedding tiếng Việt bằng Unsloth

Docs cho biết fine-tune embedding model giúp vector "hiểu" đúng kiểu tương đồng mà bài toán cần, cải thiện search và RAG trên dữ liệu riêng. Unsloth hỗ trợ:

- Train embedding, classifier, BERT, reranker nhanh hơn ~1.8–3.3 lần, ít hơn 20% bộ nhớ, context dài gấp 2 so với các bản Flash Attention 2 khác. EmbeddingGemma-300M: QLoRA cần 3GB VRAM (VRAM = bộ nhớ card đồ họa), LoRA cần 6GB.
- LoRA/QLoRA hoặc full fine-tuning; hỗ trợ tốt nhất cho model `SentenceTransformer` encoder-only có `modules.json`. Model không có `modules.json` được hỗ trợ hạn chế (tự gán pooling mặc định) — cần kiểm tra lại output embedding.
- Class trung tâm: `FastSentenceTransformer`.

Model được docs liệt kê (không đầy đủ):

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

[Nhận định] Docs **không** nói model nào hỗ trợ tiếng Việt. Các ứng viên đa ngôn ngữ nhìn theo tên là `BAAI/bge-m3` và `intfloat/multilingual-e5-large-instruct` (cùng Qwen3-Embedding) — khả năng tiếng Việt cần kiểm tra lại trên model card và bằng đánh giá trên dữ liệu thật.

### Lưu và load

| Hàm | Tác dụng |
| --- | --- |
| `save_pretrained()` | Lưu LoRA adapter ra thư mục |
| `save_pretrained_merged()` | Lưu model đã merge ra thư mục |
| `push_to_hub()` | Đẩy LoRA adapter lên Hugging Face |
| `push_to_hub_merged()` | Đẩy model đã merge lên Hugging Face |

Load để inference **bắt buộc** có `for_inference=True`:

```python
model = FastSentenceTransformer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    for_inference=True,
)
```

Dùng model đã fine-tune (ví dụ nguyên văn trong docs):

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

Model sau fine-tune dùng được với transformers, sentence-transformers, LangChain, Weaviate, Text Embeddings Inference (TEI), vLLM, llama.cpp, pgvector, FAISS và framework RAG bất kỳ.

::: info Code train và định dạng dữ liệu không có trong trang docs
Trang embedding **không** có code vòng train và không mô tả định dạng dataset (ví dụ dạng cặp câu hỏi–đoạn văn). Phần này nằm trong các notebook (EmbeddingGemma 300M, Qwen3-Embedding 4B/0.6B, BGE M3, All-MiniLM-L6-v2, ModernBERT) — cần kiểm tra lại trực tiếp trong notebook.

[Nhận định] Với ngân hàng, dữ liệu train hợp lý là các cặp (câu hỏi nhân viên thực tế, đoạn quy trình trả lời đúng), lấy từ log hỏi đáp nội bộ đã ẩn danh hóa.
:::

**Nguồn:** https://unsloth.ai/docs/basics/embedding-finetuning

## (c) Tùy chọn: fine-tune LLM cho giọng văn và định dạng

FAQ nói fine-tune giúp kiểm soát giọng văn, bám thương hiệu và **tuân thủ yêu cầu quy định**, đồng thời khuyên bắt đầu bằng QLoRA (LoRA trên model lượng tử hóa 4-bit).

[Nhận định] Trong kịch bản này chỉ nên fine-tune LLM để cố định **cách trả lời** (luôn trích nguồn, cấu trúc câu trả lời, từ chối câu hỏi ngoài phạm vi, xưng hô chuẩn), còn **kiến thức nghiệp vụ** để RAG lo vì thay đổi liên tục. Sau khi train, xuất sang GGUF để chạy trên Unsloth API — các bước xuất xem [Export & deploy](/export-deploy); việc `unsloth run` load file GGUF cục bộ (thay vì repo Hugging Face) cần kiểm tra lại vì trang API chỉ đưa ví dụ tên repo. Quy trình train xem [Fine-tuning](/fine-tuning).

::: tip Kiến thức nền
LoRA/QLoRA: xem [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora).
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/basics/api

## Rủi ro và lưu ý

::: warning Từ docs Unsloth
- **Tool phía server chạy dưới quyền user hệ điều hành.** Ai có API key và truy cập được server có thể chạy code trên máy. Khi mở ra mạng: `--disable-tools` và giữ kín key.
- **API key như mật khẩu**: ai có key và đường mạng tới Unsloth đều gửi request vào model được.
- LAN là HTTP không mã hóa.
- Qua Cloudflare quick tunnel, server-sent events (streaming) không hoạt động — phải đặt `stream: false`.
- Câu trả lời bị cắt: kiểm tra **Context used** trong API monitor; gần 100% hoặc stop reason `length` nghĩa là context đầy.
:::

::: warning [Nhận định] Riêng cho ngân hàng
- **Rò rỉ qua retrieval**: nếu không lọc theo quyền trước khi đưa đoạn văn vào prompt, người dùng có thể hỏi ra tài liệu họ không được xem. Phân quyền phải làm ở tầng FastAPI/vector DB, không trông vào LLM.
- **Prompt injection** (chèn lệnh độc qua nội dung tài liệu): tài liệu nạp vào kho có thể chứa câu điều khiển model. Giữ tool tắt, không cho LLM gọi hành động nghiệp vụ trực tiếp.
- **Hallucination**: bắt buộc hiển thị nguồn trích dẫn và cho phép nhân viên mở tài liệu gốc; câu trả lời chỉ mang tính tham khảo.
- **Dữ liệu train**: dữ liệu dùng fine-tune embedding hoặc LLM phải được ẩn danh hóa; thông tin khách hàng một khi đã vào trọng số model thì khó gỡ.
- **Tuân thủ**: cần đối chiếu với quy định nội bộ và pháp luật về bảo vệ dữ liệu cá nhân, an toàn thông tin ngân hàng hiện hành — cần kiểm tra lại với bộ phận pháp chế.
- **Audit log**: log câu hỏi, tài liệu được truy xuất, câu trả lời và người hỏi; bản thân log cũng là dữ liệu nhạy cảm cần bảo vệ.
- **Đánh giá**: lập bộ câu hỏi kiểm thử tiếng Việt có đáp án chuẩn để đo retrieval (tìm đúng đoạn chưa) và câu trả lời trước mỗi lần đổi model.
:::

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/basics/lan, https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth
