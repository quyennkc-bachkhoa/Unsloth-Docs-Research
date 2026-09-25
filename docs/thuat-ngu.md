---
title: Thuật ngữ
description: Bảng thuật ngữ Anh – Việt dùng trong website, mỗi thuật ngữ trỏ tới trang giải thích.
---

# Thuật ngữ

Trang này gom các thuật ngữ tiếng Anh dùng trên website, mỗi thuật ngữ kèm một câu giải thích ngắn bằng tiếng Việt. Cột "Xem" trỏ tới trang giải thích kỹ nhất. Trang thuộc route [Kiến thức nền LLM](/kien-thuc-nen/) giải thích khái niệm; trang thuộc route Unsloth giải thích cách dùng trong Unsloth. Muốn tra nhanh, dùng ô tìm kiếm ở góc trên.

::: tip Tóm tắt
- **Dùng khi:** Bạn gặp một thuật ngữ tiếng Anh trên website mà chưa rõ nghĩa.
- **Kết quả:** Một câu giải thích ngắn cho mỗi thuật ngữ và link tới trang giải thích kỹ nhất.
- **Nên biết trước:** Không cần kiến thức trước.
:::

## Mô hình và kiến trúc

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| LLM (Large Language Model) | Mô hình ngôn ngữ lớn. Không có mốc số tham số chính thức để gọi là "lớn" | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| SLM (Small Language Model) | Mô hình ngôn ngữ nhỏ. Đây là cách gọi tương đối, không có mốc cố định | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| Base model | Model mới qua pre-train, chỉ biết viết tiếp văn bản chứ chưa biết trả lời theo yêu cầu | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| Instruct / chat model | Base model đã được train thêm (post-train) để làm theo chỉ dẫn và trò chuyện | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| Multimodal, VLM | Model xử lý nhiều loại dữ liệu (chữ, ảnh, âm thanh). VLM là loại nhận ảnh và chữ, rồi sinh ra chữ | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| TTS / STT / ASR | TTS: đọc chữ thành giọng nói. STT: chuyển giọng nói thành chữ. ASR: nhận dạng giọng nói | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| Embedding model | Model trả về một vector (dãy số) thể hiện nghĩa của đoạn văn, không sinh ra chữ | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh), [Ứng dụng RAG](/ung-dung-rag) |
| Encoder / decoder model | Encoder: đọc cả câu rồi tạo biểu diễn số (ví dụ BERT). Decoder: sinh từng token nối tiếp nhau (ví dụ Llama) | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| Token | Mảnh văn bản nhỏ nhất mà model xử lý, có thể là một từ, một phần từ hoặc dấu câu | [Token & context](/kien-thuc-nen/token-va-context) |
| Tokenizer | Bộ phận cắt văn bản thành token, rồi đổi mỗi token thành một mã số (ID) | [Token & context](/kien-thuc-nen/token-va-context) |
| Vocabulary | Danh sách cố định mọi token mà tokenizer biết | [Token & context](/kien-thuc-nen/token-va-context) |
| BPE, WordPiece, SentencePiece | Các thuật toán và thư viện cắt văn bản thành mảnh từ (subword) | [Token & context](/kien-thuc-nen/token-va-context) |
| Special token, BOS, EOS | Special token: token mang tín hiệu điều khiển. BOS: token đánh dấu đầu chuỗi. EOS: token kết thúc, model sinh ra nó thì dừng | [Token & context](/kien-thuc-nen/token-va-context) |
| Context window | Số token tối đa model xử lý trong một lần, tính cả prompt lẫn phần model sinh ra | [Token & context](/kien-thuc-nen/token-va-context) |
| `max_seq_length` | Độ dài chuỗi tối đa (tính bằng token) bạn đặt khi nạp hoặc train model trong Unsloth | [Token & context](/kien-thuc-nen/token-va-context), [Fine-tuning](/fine-tuning/) |
| Next-token prediction | Cách LLM làm việc: nhìn các token đã có rồi đoán token kế tiếp | [Token & context](/kien-thuc-nen/token-va-context) |
| YaRN, RoPE scaling | Kỹ thuật kéo giãn cách model ghi nhận vị trí token, để dùng được context dài hơn | [Token & context](/kien-thuc-nen/token-va-context) |
| Transformer | Kiến trúc mạng nơ-ron dựa trên cơ chế attention | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Decoder-only | Kiến trúc chỉ dùng phần decoder. Mỗi token chỉ được nhìn các token đứng trước nó | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Embedding (lớp) | Bảng tra đổi mỗi token ID thành một vector | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Self-attention, Q/K/V | Cơ chế giúp mỗi token lấy thông tin ngữ cảnh từ các token khác, thông qua ba vector Query, Key, Value | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Multi-head attention | Nhiều head attention chạy song song cùng lúc | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| GQA / MQA | Nhiều head Query dùng chung head Key và Value, nhờ đó KV cache nhỏ hơn. MQA là trường hợp chỉ có 1 head K/V | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| FFN / MLP | Mạng truyền thẳng (feed-forward) nằm trong mỗi layer | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| ReLU, GELU, SwiGLU, SiLU | Các hàm kích hoạt | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| LayerNorm, RMSNorm | Các lớp chuẩn hóa | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| RoPE | Cách cho model biết vị trí token bằng cách xoay vector Query và Key | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Residual connection | Cộng đầu vào của một khối vào đầu ra của chính khối đó: `x + Khối(x)` | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Layer / decoder block | Một khối gồm các bước norm → attention → residual → norm → FFN → residual | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| LM head, logits | LM head là lớp cuối, biến vector thành điểm thô (logits) cho từng token | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Softmax | Hàm biến một dãy điểm thành các xác suất có tổng bằng 1 | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| `q_proj` … `down_proj` | Tên các lớp linear trong attention và FFN. Đây là những lớp được chọn để gắn LoRA (target modules) | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Dense model | Model dùng toàn bộ tham số để xử lý mỗi token | [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| MoE (Mixture of Experts) | Model có nhiều FFN gọi là expert, nhưng mỗi token chỉ đi qua vài expert | [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| Expert, shared expert | Expert: một FFN trong lớp MoE. Shared expert: expert mà mọi token đều đi qua | [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| Router / gating, top-k routing | Mạng nhỏ quyết định mỗi token đi qua k expert nào | [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| Total / active parameters | Total: tổng số tham số, quyết định bộ nhớ cần có. Active: số tham số dùng cho mỗi token, quyết định lượng tính toán | [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| Ký hiệu `30B-A3B` | Model có 30 tỷ tham số tổng, trong đó 3 tỷ tham số được kích hoạt | [Dense & MoE](/kien-thuc-nen/dense-va-moe), [Danh sách model hỗ trợ](/model-catalog) |

**Nguồn:** giải thích tổng hợp từ các trang được trỏ tới ở cột "Xem"; nguồn gốc của từng khái niệm ghi ở trang đó. Danh sách URL đầy đủ: [Nguồn tham khảo](/nguon).

## Bộ nhớ, độ chính xác và lượng tử hóa

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| Parameter / weight | Các con số model học được. Ví dụ 7B ≈ 7 tỷ tham số | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| dtype | Kiểu số dùng để lưu tham số | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| VRAM | Bộ nhớ card đồ họa | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho), [Cài đặt](/cai-dat) |
| Unified memory | Bộ nhớ mà CPU và GPU dùng chung, ví dụ trên Mac | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| Offload | Khi thiếu VRAM, đẩy một phần model hoặc cache sang RAM hay ổ đĩa | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| OOM (out of memory) | Lỗi hết bộ nhớ | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| KV cache | Bộ nhớ giữ key và value của các token trước để không phải tính lại. Context càng dài, KV cache càng lớn | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho), [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Activation | Các kết quả trung gian sinh ra khi dữ liệu đi qua model (lượt forward) | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| Shard | Model lớn được chia thành nhiều file; mỗi file là một shard (`00001-of-00003`) | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| mmproj | File chứa module chiếu cho phần vision (xử lý ảnh), đi kèm file GGUF | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| FP32, FP16, BF16 | Số dấu phẩy động dài lần lượt 32, 16 và 16 bit. BF16 có khoảng giá trị như FP32 | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Exponent / mantissa | Exponent: bit mũ, quyết định khoảng giá trị. Mantissa: bit định trị, quyết định độ mịn | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Mixed precision | Tính toán ở 16-bit, nhưng giữ một bản trọng số FP32 để cập nhật | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| FP8 (E4M3, E5M2) | Số 8 bit, có hai biến thể: E4M3 ưu tiên độ mịn, E5M2 ưu tiên khoảng giá trị | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| NVFP4 | Định dạng 4-bit của NVIDIA; cứ mỗi khối 16 phần tử dùng chung một hệ số scale | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [Export và deploy](/export-deploy/nvfp4-fp8#nvfp4) |
| MXFP4, microscaling (MX) | Định dạng 4-bit; cứ mỗi khối 32 phần tử dùng chung một hệ số scale | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| INT8 / INT4 | Lưu bằng số nguyên 8 hoặc 4 bit, kèm hệ số scale | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Quantization (lượng tử hóa) | Lưu trọng số bằng ít bit hơn (độ chính xác thấp hơn) để model chiếm ít bộ nhớ hơn | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| PTQ / QAT | PTQ: lượng tử hóa sau khi train xong. QAT: trong lúc train đã mô phỏng lượng tử hóa | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [Fine-tuning](/fine-tuning/) |
| Calibration, imatrix | Calibration: dữ liệu mẫu dùng để hiệu chỉnh khi lượng tử hóa. Imatrix: ma trận độ quan trọng của llama.cpp | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| W4A16, W4A4, W8A8 | Cách ghi số bit: W là trọng số, A là activation | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| GGUF | Định dạng gói model vào một file duy nhất, dùng bởi llama.cpp và ggml | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [Export và deploy](/export-deploy/gguf) |
| K-quant (`Q4_K_M`…), I-quant (`IQ2_XXS`…) | Các kiểu lượng tử hóa trong file GGUF | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| bpw (bits per weight) | Số bit trung bình mỗi trọng số | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Unsloth Dynamic (UD) | Cách lượng tử hóa của Unsloth, chọn mức bit riêng cho từng layer | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [Export và deploy](/export-deploy/gguf#dynamic-gguf) |
| `bnb-4bit`, `unsloth-bnb-4bit` | `bnb-4bit`: safetensors 4-bit dùng BitsAndBytes. `unsloth-bnb-4bit`: bản dynamic 4-bit của Unsloth. Cả hai dùng để fine-tune | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| KL divergence (KLD), perplexity | KLD: mức lệch phân phối so với model gốc. Perplexity: độ "bối rối" của model trên một văn bản | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| NF4, double quantization | NF4: kiểu số 4-bit NormalFloat mà QLoRA dùng. Double quantization: lượng tử hóa luôn cả các hằng số scale | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Compute capability, tensor core | Compute capability: mã cho biết GPU NVIDIA có tính năng phần cứng nào. Tensor core: bộ phận chuyên nhân ma trận | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [Cài đặt](/cai-dat) |
| Blackwell | Một thế hệ GPU NVIDIA, ví dụ RTX 50, B200 | [Cài đặt](/cai-dat), [Export và deploy](/export-deploy/nvfp4-fp8#nvfp4) |

**Nguồn:** giải thích tổng hợp từ các trang được trỏ tới ở cột "Xem"; nguồn gốc của từng khái niệm ghi ở trang đó. Danh sách URL đầy đủ: [Nguồn tham khảo](/nguon).

## Huấn luyện

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| Pretraining | Huấn luyện model từ đầu trên lượng dữ liệu rất lớn | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Fine-tuning (tinh chỉnh) | Train thêm một model đã pretrain bằng dữ liệu của bạn | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen), [Fine-tuning](/fine-tuning/) |
| SFT | Tinh chỉnh có giám sát: model học từ các cặp đầu vào → đầu ra mẫu | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Continued pretraining (CPT) | Pretrain tiếp trên văn bản thô, để model học một lĩnh vực hoặc ngôn ngữ mới | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen), [Fine-tuning](/fine-tuning/) |
| Loss, cross-entropy | Con số đo model sai bao nhiêu. Loss theo token = −ln(xác suất token đúng) | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Gradient, backpropagation | Gradient: đạo hàm của loss theo từng tham số. Backpropagation: thuật toán tính gradient đi ngược từ loss | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Optimizer (SGD, Adam, AdamW) | Thuật toán dùng gradient để cập nhật trọng số | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Optimizer state | Các giá trị optimizer lưu thêm cho mỗi tham số (AdamW lưu 2 giá trị) | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| `adamw_8bit` | AdamW lưu optimizer state ở 8-bit | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Learning rate | Mỗi bước train, tham số được dịch đi nhiều hay ít | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| LR scheduler (linear, cosine), warmup | Scheduler: lịch thay đổi learning rate trong lúc train. Warmup: tăng dần learning rate ở các bước đầu | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Epoch, step | Epoch: một lượt đi qua toàn bộ dataset. Step: một lần cập nhật trọng số | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Batch size | Số mẫu dữ liệu xử lý trong mỗi lượt forward và backward trên một GPU | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Gradient accumulation, effective batch size | Cộng dồn gradient của nhiều micro-batch rồi mới cập nhật. Effective batch size = `batch_size × gradient_accumulation_steps` | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Weight decay, regularization | Weight decay: kéo trọng số về gần 0. Regularization: các kỹ thuật chống overfitting nói chung | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Overfitting / underfitting | Overfitting: model học thuộc dữ liệu train. Underfitting: model học chưa đủ | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Train loss / eval loss | Train loss: loss trên dữ liệu train. Eval loss: loss trên phần dữ liệu tách riêng | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Early stopping | Dừng train sớm khi eval loss không giảm nữa | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Checkpoint | Bản lưu model giữa chừng, dùng để train tiếp | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen), [Fine-tuning](/fine-tuning/) |
| Hyperparameter | Siêu tham số: các thiết lập bạn đặt trước khi train (learning rate, rank…) | [Fine-tuning](/fine-tuning/) |
| Full fine-tuning (FFT) | Train và cập nhật mọi trọng số của model | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| PEFT | Tinh chỉnh tiết kiệm tham số: giữ nguyên (đóng băng) model gốc, chỉ train một phần nhỏ gắn thêm vào | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| LoRA, adapter | Học phần cập nhật dưới dạng tích của hai ma trận hạng thấp. Phần gắn thêm này gọi là adapter | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| Rank `r`, `lora_alpha`, `lora_dropout` | `r`: bề rộng adapter. `lora_alpha`: hệ số scale (alpha/r). `lora_dropout`: dropout của nhánh LoRA | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| rsLoRA | Rank-Stabilized LoRA: biến thể LoRA scale theo alpha/√r | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| `target_modules` | Danh sách các lớp linear được gắn LoRA | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| QLoRA | LoRA gắn trên model gốc đã lượng tử hóa 4-bit | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| Gradient checkpointing | Không giữ hết activation trong bộ nhớ, khi backward thì tính lại. Cách này tiết kiệm VRAM | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| Merge, `save_method` | Merge: cộng LoRA vào trọng số gốc. `save_method`: chọn cách lưu (`merged_16bit`, `merged_4bit`, `lora`) | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora), [Export và deploy](/export-deploy/) |
| LoRA hot swapping | Nạp hoặc gỡ adapter trên vLLM ngay khi đang chạy | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| Multi-GPU, DDP | Multi-GPU: train trên nhiều GPU. DDP: song song hóa dữ liệu giữa các GPU | [Fine-tuning](/fine-tuning/) |
| Dataset format (Alpaca, ShareGPT, ChatML) | Các định dạng dữ liệu dạng instruction hoặc dạng hội thoại | [Các định dạng dữ liệu](/du-lieu/dinh-dang) |
| Synthetic data | Dữ liệu train do LLM sinh ra | [Dữ liệu tổng hợp](/du-lieu/synthetic) |
| Data Recipes | Công cụ trong Unsloth Studio để tạo dataset từ tài liệu | [Data Recipes](/du-lieu/studio#data-recipes) |
| `train_on_responses_only` | Chỉ tính loss trên phần câu trả lời, bỏ qua phần câu hỏi | [Chat template](/du-lieu/chat-template#chi-train-tren-cau-tra-loi), [Token & context](/kien-thuc-nen/token-va-context) |

**Nguồn:** giải thích tổng hợp từ các trang được trỏ tới ở cột "Xem"; nguồn gốc của từng khái niệm ghi ở trang đó. Danh sách URL đầy đủ: [Nguồn tham khảo](/nguon).

## Reinforcement learning và preference

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| Reinforcement Learning (RL) | Model tự sinh câu trả lời, câu trả lời được chấm điểm, rồi model được chỉnh về phía câu điểm cao | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Policy, action, environment | Policy: model đang train. Action: văn bản model sinh ra. Environment: bối cảnh của tác vụ | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Reward, reward function, verifier | Reward: điểm thưởng. Reward function: hàm đổi kết quả thành điểm. Verifier: bộ kiểm tra đúng hay sai | [RL & preference](/kien-thuc-nen/rl-va-preference), [Reward function và verifier](/reinforcement-learning/reward-function) |
| Reward model | Một mạng riêng được train chỉ để chấm điểm | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Reference model, KL penalty (`beta`) | Reference model: bản đóng băng làm mốc so sánh. KL penalty: phạt khi model lệch khỏi mốc | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| RLHF, PPO | RLHF: RL dựa trên phản hồi của con người. PPO: thuật toán RL có cơ chế clip để cập nhật ổn định | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Rollout | Một lần cho model sinh thử một câu trả lời | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| GRPO | Thuật toán RL lấy điểm trung bình của một nhóm câu trả lời làm mốc, thay cho value model | [RL & preference](/kien-thuc-nen/rl-va-preference), [GRPO trong Unsloth](/reinforcement-learning/grpo) |
| Advantage | Một câu trả lời hơn hay kém trung bình nhóm bao nhiêu, chia cho độ lệch chuẩn | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| RLVR | RL với reward kiểm chứng tự động được, ví dụ toán hoặc code | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| DPO, ORPO, KTO | Các phương pháp căn chỉnh model theo sở thích (preference) | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Preference data (chosen / rejected) | Với cùng một prompt, có một câu được ưa thích (chosen) và một câu bị chê (rejected) | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Reward hacking | Model tìm cách lách luật để tăng reward mà không thực sự làm đúng việc | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Unsloth Standby | Cơ chế cho inference (vLLM) và train dùng chung vùng nhớ khi chạy RL | [Memory-efficient RL](/reinforcement-learning/memory-efficient) |

**Nguồn:** giải thích tổng hợp từ các trang được trỏ tới ở cột "Xem"; nguồn gốc của từng khái niệm ghi ở trang đó. Danh sách URL đầy đủ: [Nguồn tham khảo](/nguon).

## Suy luận và sử dụng

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| Inference (suy luận) | Dùng model để sinh kết quả; model không học thêm gì | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling), [Chạy model và gọi API](/inference/) |
| Autoregressive | Sinh từng token một, nối vào chuỗi rồi đoán token tiếp theo | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Prefill / decode | Prefill: pha xử lý cả prompt song song. Decode: pha sinh từng token | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Greedy / sampling | Greedy: luôn chọn token có xác suất cao nhất. Sampling: bốc ngẫu nhiên theo xác suất | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Temperature | Làm phân phối xác suất nhọn hơn hoặc phẳng hơn | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Top-k, top-p, min-p | Các cách loại bớt token ứng viên trước khi bốc ngẫu nhiên | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Repetition / presence / frequency penalty | Phạt các token đã xuất hiện để model bớt lặp lại | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Chat template | Quy tắc đổi danh sách tin nhắn thành một chuỗi token, có chèn các token điều khiển | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling), [Chat template](/du-lieu/chat-template) |
| `add_generation_prompt` | Thêm phần mở đầu lượt của assistant vào cuối prompt, để model biết tới lượt nó trả lời | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Thinking / reasoning mode, `reasoning_effort` | Model viết ra một đoạn suy nghĩ trước khi trả lời. `reasoning_effort` chỉnh mức độ suy nghĩ | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling), [Chạy model và gọi API](/inference/api) |
| Tool calling (function calling) | Model sinh ra yêu cầu gọi hàm (dạng JSON). App chạy hàm đó rồi trả kết quả lại | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling), [Chạy model và gọi API](/inference/tool-calling) |
| Self-healing tool calling | Unsloth tự sửa các tool call bị lỗi | [Chạy model và gọi API](/inference/tool-calling) |
| Server-side tools | Các công cụ Unsloth tự chạy, như Python, bash, web search | [Chạy model và gọi API](/inference/tool-calling) |
| MCP (Model Context Protocol) | Giao thức chuẩn để model kết nối và gọi dịch vụ bên ngoài | [Chạy model và gọi API](/inference/mcp) |
| OpenAI-compatible API, endpoint | API có cùng định dạng với API của OpenAI. Endpoint là địa chỉ HTTP nhận request | [Chạy model và gọi API](/inference/api) |
| API key (`sk-unsloth-…`), Bearer token | Khóa xác thực, gửi kèm request qua header `Authorization: Bearer` | [Chạy model và gọi API](/inference/api) |
| Streaming / SSE | Trả kết quả từng phần ngay khi có, qua một luồng sự kiện | [Chạy model và gọi API](/inference/api) |
| Coding agent | Tác tử (agent) lập trình, ví dụ Claude Code, Codex, OpenCode | [Chạy model và gọi API](/inference/coding-agent) |
| Prompt injection | Lệnh độc hại được chèn vào nội dung mà model đọc | [Chạy model và gọi API](/inference/mcp), [Ứng dụng RAG](/ung-dung-rag) |
| MTP, speculative decoding | Đoán trước nhiều token để decode nhanh hơn | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho), [Export và deploy](/export-deploy/) |
| RAG | Tìm tài liệu liên quan, đưa vào prompt, rồi để model dựa vào đó trả lời | [Ứng dụng RAG](/ung-dung-rag) |
| Vector DB, reranker | Vector DB: cơ sở dữ liệu lưu vector. Reranker: model xếp hạng lại kết quả tìm được | [Ứng dụng RAG](/ung-dung-rag) |
| Hallucination | Model bịa ra thông tin | [Ứng dụng RAG](/ung-dung-rag) |

**Nguồn:** giải thích tổng hợp từ các trang được trỏ tới ở cột "Xem"; nguồn gốc của từng khái niệm ghi ở trang đó. Danh sách URL đầy đủ: [Nguồn tham khảo](/nguon).

## Sản phẩm, công cụ và triển khai

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| Unsloth Desktop | App native cài trên macOS, Windows, Linux | [Tổng quan](/tong-quan) |
| Unsloth Studio | Giao diện web không cần code (no-code), cài thủ công | [Tổng quan](/tong-quan) |
| Unsloth Core | Thư viện Python gốc, dùng bằng cách viết code | [Tổng quan](/tong-quan) |
| `unsloth run`, `unsloth start` | `unsloth run`: nạp model và mở API. `unsloth start`: nối coding agent với model local | [Chạy model và gọi API](/inference/coding-agent) |
| safetensors | Định dạng lưu trọng số chuẩn của Hugging Face | [Export và deploy](/export-deploy/) |
| llama.cpp, llama-server | llama.cpp: engine inference viết bằng C/C++. llama-server: server HTTP của llama.cpp | [Chạy model và gọi API](/inference/), [Export và deploy](/export-deploy/chay-model#llama-server) |
| vLLM, SGLang | Engine inference hiệu năng cao cho production | [Export và deploy](/export-deploy/chay-model#vllm) |
| Ollama, LM Studio | Ứng dụng chạy model local | [Ollama](/export-deploy/chay-model#ollama), [LM Studio](/export-deploy/chay-model#lm-studio) |
| Modelfile | File cấu hình model của Ollama | [Export và deploy](/export-deploy/chay-model#ollama) |
| MLX | Framework ML trên Apple Silicon | [Cài đặt](/cai-dat) |
| CUDA, ROCm, XPU/oneAPI | Nền tảng tính toán trên GPU, lần lượt của NVIDIA, AMD, Intel | [Cài đặt](/cai-dat) |
| WSL | Chạy Linux trong Windows | [Cài đặt](/cai-dat) |
| venv / uv / conda | Công cụ tạo môi trường Python cô lập và quản lý gói | [Cài đặt](/cai-dat) |
| bitsandbytes, xformers, triton | bitsandbytes: thư viện lượng tử hóa. xformers, triton: thư viện kernel tăng tốc | [Cài đặt](/cai-dat) |
| Docker image / container | Image: môi trường đóng gói sẵn. Container: một bản image đang chạy | [Cài đặt](/cai-dat) |
| LAN access, bind `0.0.0.0` | Mở server để máy khác trong mạng nội bộ truy cập được | [Export và deploy](/export-deploy/lan-remote) |
| Cloudflare tunnel, `--secure` | Link HTTPS công khai trỏ tới server chạy trên máy bạn | [Export và deploy](/export-deploy/lan-remote) |
| Push to Hub | Đẩy model lên Hugging Face Hub | [Export và deploy](/export-deploy/) |

**Nguồn:** thuật ngữ và giải thích tổng hợp từ các trang trên website; nguồn gốc của từng khái niệm ghi ở trang được trỏ tới. Danh sách URL đầy đủ: [Nguồn tham khảo](/nguon).

## Đọc tiếp

- [Nguồn tham khảo](/nguon) — Danh sách đầy đủ URL docs Unsloth mà các trang trên website dùng.
- [Kiến thức nền LLM](/kien-thuc-nen/) — Đọc kỹ các khái niệm nếu một câu giải thích ở trên chưa đủ.
- [Tổng quan kiến trúc](/tong-quan) — Xem các thuật ngữ sản phẩm (Desktop, Studio, Core) được đặt vào bức tranh chung.
