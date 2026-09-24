---
title: Thuật ngữ
description: Bảng thuật ngữ Anh – Việt dùng trong website, mỗi thuật ngữ trỏ tới trang giải thích.
---

# Thuật ngữ

Trang này gom các thuật ngữ tiếng Anh dùng trên website, kèm giải thích ngắn bằng tiếng Việt. Cột "Xem" trỏ tới trang giải thích kỹ nhất; trang thuộc route [Kiến thức nền LLM](/kien-thuc-nen/) giải thích khái niệm, trang route Unsloth giải thích cách dùng trong Unsloth. Dùng ô tìm kiếm (góc trên) để tra nhanh.

## Mô hình và kiến trúc

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| LLM (Large Language Model) | Mô hình ngôn ngữ lớn; không có ngưỡng số tham số chính thức | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| SLM (Small Language Model) | Mô hình ngôn ngữ nhỏ; cách gọi tương đối | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| Base model | Model mới pre-train, chỉ biết viết tiếp văn bản | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| Instruct / chat model | Base model đã post-train để làm theo chỉ dẫn, trò chuyện | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| Multimodal, VLM | Model xử lý nhiều loại dữ liệu; VLM nhận ảnh + chữ, sinh chữ | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| TTS / STT / ASR | Chữ thành giọng nói / giọng nói thành chữ / nhận dạng giọng nói | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| Embedding model | Model trả về vector biểu diễn nghĩa thay vì sinh token | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh), [Ứng dụng RAG](/ung-dung-rag) |
| Encoder / decoder model | Model tạo biểu diễn (BERT) / model sinh token tuần tự (Llama) | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) |
| Token | Mảnh văn bản nhỏ nhất model xử lý | [Token & context](/kien-thuc-nen/token-va-context) |
| Tokenizer | Bộ tách văn bản thành token và đổi sang ID | [Token & context](/kien-thuc-nen/token-va-context) |
| Vocabulary | Danh sách cố định các token tokenizer biết | [Token & context](/kien-thuc-nen/token-va-context) |
| BPE, WordPiece, SentencePiece | Các thuật toán/thư viện tách token theo mảnh từ | [Token & context](/kien-thuc-nen/token-va-context) |
| Special token, BOS, EOS | Token điều khiển; token đầu chuỗi; token kết thúc (sinh ra thì dừng) | [Token & context](/kien-thuc-nen/token-va-context) |
| Context window | Số token tối đa model xử lý một lần (prompt + phần sinh) | [Token & context](/kien-thuc-nen/token-va-context) |
| `max_seq_length` | Độ dài chuỗi tối đa đặt khi nạp/train trong Unsloth | [Token & context](/kien-thuc-nen/token-va-context), [Fine-tuning](/fine-tuning) |
| Next-token prediction | Đoán token kế tiếp dựa trên các token trước | [Token & context](/kien-thuc-nen/token-va-context) |
| YaRN, RoPE scaling | Kỹ thuật co giãn mã hóa vị trí để dùng context dài hơn | [Token & context](/kien-thuc-nen/token-va-context) |
| Transformer | Kiến trúc mạng dựa trên attention | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Decoder-only | Chỉ dùng phần decoder, mỗi token chỉ nhìn các token phía trước | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Embedding (lớp) | Bảng tra token ID → vector | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Self-attention, Q/K/V | Cơ chế mỗi token lấy ngữ cảnh từ token khác qua Query, Key, Value | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Multi-head attention | Nhiều head attention chạy song song | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| GQA / MQA | Nhiều head Query dùng chung head K/V (MQA: chỉ 1 head K/V), giảm KV cache | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| FFN / MLP | Mạng truyền thẳng trong mỗi layer | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| ReLU, GELU, SwiGLU, SiLU | Các hàm kích hoạt | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| LayerNorm, RMSNorm | Các lớp chuẩn hóa | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| RoPE | Mã hóa vị trí bằng cách xoay vector Q/K | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Residual connection | Cộng đầu vào vào đầu ra của khối: `x + Khối(x)` | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Layer / decoder block | Một khối norm → attention → residual → norm → FFN → residual | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| LM head, logits | Lớp cuối biến vector thành điểm thô cho từng token | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Softmax | Biến dãy số thành xác suất có tổng bằng 1 | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| `q_proj` … `down_proj` | Tên các lớp linear của attention và FFN, dùng làm LoRA target modules | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) |
| Dense model | Model dùng toàn bộ tham số cho mỗi token | [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| MoE (Mixture of Experts) | Nhiều FFN expert, mỗi token chỉ qua vài expert | [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| Expert, shared expert | Một FFN trong lớp MoE; expert mọi token đều đi qua | [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| Router / gating, top-k routing | Mạng nhỏ chọn k expert cho mỗi token | [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| Total / active parameters | Tổng tham số (quyết định bộ nhớ) / tham số dùng cho mỗi token (quyết định lượng tính) | [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| Ký hiệu `30B-A3B` | 30 tỷ tham số tổng, 3 tỷ tham số kích hoạt | [Dense & MoE](/kien-thuc-nen/dense-va-moe), [Model catalog](/model-catalog) |

## Bộ nhớ, độ chính xác và lượng tử hóa

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| Parameter / weight | Con số model học được; 7B ≈ 7 tỷ tham số | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| dtype | Kiểu số dùng lưu tham số | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| VRAM | Bộ nhớ card đồ họa | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho), [Cài đặt](/cai-dat) |
| Unified memory | Bộ nhớ dùng chung cho CPU và GPU (vd Mac) | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| Offload | Đẩy một phần model/cache sang RAM hoặc đĩa khi thiếu VRAM | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| OOM (out of memory) | Lỗi hết bộ nhớ | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| KV cache | Bộ nhớ lưu key/value của các token trước để không tính lại; tăng theo độ dài context | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho), [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Activation | Kết quả trung gian của lượt forward | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| Shard | Một phần của model bị chia thành nhiều file (`00001-of-00003`) | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| mmproj | File module chiếu cho phần vision, đi kèm GGUF | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| FP32, FP16, BF16 | Số dấu phẩy động 32 / 16 / 16 bit (BF16 có khoảng giá trị như FP32) | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Exponent / mantissa | Bit mũ (khoảng giá trị) / bit định trị (độ mịn) | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Mixed precision | Tính ở 16-bit, giữ bản trọng số FP32 để cập nhật | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| FP8 (E4M3, E5M2) | Số 8 bit, hai biến thể ưu tiên độ mịn / khoảng giá trị | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| NVFP4 | Định dạng 4-bit của NVIDIA, scale theo khối 16 phần tử | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [Export & deploy](/export-deploy) |
| MXFP4, microscaling (MX) | Định dạng 4-bit scale theo khối 32 phần tử | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| INT8 / INT4 | Số nguyên 8 / 4 bit cộng hệ số scale | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Quantization (lượng tử hóa) | Lưu trọng số ở độ chính xác thấp hơn để giảm bộ nhớ | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| PTQ / QAT | Lượng tử hóa sau khi train / train có mô phỏng lượng tử hóa | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [Fine-tuning](/fine-tuning) |
| Calibration, imatrix | Dữ liệu hiệu chỉnh khi lượng tử hóa; ma trận độ quan trọng của llama.cpp | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| W4A16, W4A4, W8A8 | Số bit của trọng số (W) và activation (A) | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| GGUF | Định dạng file một tệp của llama.cpp/ggml | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [Export & deploy](/export-deploy) |
| K-quant (`Q4_K_M`…), I-quant (`IQ2_XXS`…) | Các kiểu quant GGUF | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| bpw (bits per weight) | Số bit trung bình mỗi trọng số | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Unsloth Dynamic (UD) | Quant của Unsloth chọn mức bit riêng cho từng layer | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [Export & deploy](/export-deploy) |
| `bnb-4bit`, `unsloth-bnb-4bit` | Safetensors 4-bit BitsAndBytes / bản dynamic 4-bit của Unsloth, dùng để fine-tune | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| KL divergence (KLD), perplexity | Độ lệch phân phối so với model gốc; độ "bối rối" trên văn bản | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| NF4, double quantization | Kiểu 4-bit NormalFloat của QLoRA; lượng tử hóa luôn các hằng số scale | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Compute capability, tensor core | Mã tính năng phần cứng GPU NVIDIA; đơn vị nhân ma trận chuyên dụng | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), [Cài đặt](/cai-dat) |
| Blackwell | Thế hệ GPU NVIDIA (RTX 50, B200) | [Cài đặt](/cai-dat), [Export & deploy](/export-deploy) |

## Huấn luyện

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| Pretraining | Huấn luyện model từ đầu trên dữ liệu rất lớn | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Fine-tuning (tinh chỉnh) | Train thêm model đã pretrain bằng dữ liệu riêng | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen), [Fine-tuning](/fine-tuning) |
| SFT | Tinh chỉnh có giám sát trên cặp đầu vào → đầu ra | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Continued pretraining (CPT) | Pretrain tiếp trên văn bản thô để học miền/ngôn ngữ mới | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen), [Fine-tuning](/fine-tuning) |
| Loss, cross-entropy | Con số đo mức sai; loss theo token = −ln(xác suất token đúng) | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Gradient, backpropagation | Đạo hàm của loss theo tham số; thuật toán tính gradient ngược từ loss | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Optimizer (SGD, Adam, AdamW) | Thuật toán cập nhật trọng số theo gradient | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Optimizer state | Giá trị optimizer lưu thêm cho mỗi tham số (AdamW: 2) | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) |
| `adamw_8bit` | AdamW lưu optimizer state ở 8-bit | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Learning rate | Mức dịch tham số mỗi bước | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| LR scheduler (linear, cosine), warmup | Lịch thay đổi learning rate; tăng dần learning rate ở các bước đầu | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Epoch, step | Một lượt qua toàn bộ dataset; một lần cập nhật trọng số | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Batch size | Số mẫu mỗi lượt forward/backward trên một GPU | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Gradient accumulation, effective batch size | Cộng dồn gradient nhiều micro-batch; `batch_size × gradient_accumulation_steps` | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Weight decay, regularization | Kéo trọng số về gần 0; các kỹ thuật chống overfitting | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Overfitting / underfitting | Học thuộc dữ liệu train / học chưa đủ | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Train loss / eval loss | Loss trên dữ liệu train / trên dữ liệu tách riêng | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Early stopping | Dừng train khi eval loss không giảm nữa | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| Checkpoint | Bản lưu giữa chừng để train tiếp | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen), [Fine-tuning](/fine-tuning) |
| Hyperparameter | Siêu tham số đặt trước khi train (learning rate, rank…) | [Fine-tuning](/fine-tuning) |
| Full fine-tuning (FFT) | Cập nhật mọi trọng số | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| PEFT | Tinh chỉnh tiết kiệm tham số: đóng băng model gốc, chỉ train phần nhỏ thêm vào | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| LoRA, adapter | Học phần cập nhật dưới dạng tích hai ma trận hạng thấp | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| Rank `r`, `lora_alpha`, `lora_dropout` | Bề rộng adapter; hệ số scale (alpha/r); dropout của nhánh LoRA | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| rsLoRA | Rank-Stabilized LoRA, scale theo alpha/√r | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| `target_modules` | Các lớp linear được gắn LoRA | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| QLoRA | LoRA trên model gốc lượng tử hóa 4-bit | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| Gradient checkpointing | Không giữ hết activation, tính lại khi backward để tiết kiệm VRAM | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| Merge, `save_method` | Cộng LoRA vào trọng số gốc; cách lưu (`merged_16bit`, `merged_4bit`, `lora`) | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora), [Export & deploy](/export-deploy) |
| LoRA hot swapping | Nạp/gỡ adapter trên vLLM khi đang chạy | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) |
| Multi-GPU, DDP | Train trên nhiều GPU; song song hóa dữ liệu | [Fine-tuning](/fine-tuning) |
| Dataset format (Alpaca, ShareGPT, ChatML) | Các định dạng dữ liệu instruction / hội thoại | [Dữ liệu](/du-lieu) |
| Synthetic data | Dữ liệu do LLM sinh ra | [Dữ liệu](/du-lieu) |
| Data Recipes | Công cụ tạo dataset từ tài liệu trong Unsloth Studio | [Dữ liệu](/du-lieu) |
| `train_on_responses_only` | Chỉ tính loss trên phần trả lời | [Dữ liệu](/du-lieu), [Token & context](/kien-thuc-nen/token-va-context) |

## Reinforcement learning và preference

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| Reinforcement Learning (RL) | Model tự sinh câu trả lời, được chấm điểm, rồi được đẩy về phía câu điểm cao | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Policy, action, environment | Model đang train; văn bản sinh ra; bối cảnh tác vụ | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Reward, reward function, verifier | Điểm thưởng; hàm đổi kết quả thành điểm; bộ kiểm tra đúng/sai | [RL & preference](/kien-thuc-nen/rl-va-preference), [Reinforcement Learning](/reinforcement-learning) |
| Reward model | Mạng riêng được train để chấm điểm | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Reference model, KL penalty (`beta`) | Bản đóng băng làm mốc; phạt mức lệch khỏi mốc | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| RLHF, PPO | RL từ phản hồi con người; thuật toán RL có clip để cập nhật ổn định | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Rollout | Một lần cho model sinh thử câu trả lời | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| GRPO | RL lấy trung bình của một nhóm câu trả lời làm mốc thay cho value model | [RL & preference](/kien-thuc-nen/rl-va-preference), [Reinforcement Learning](/reinforcement-learning) |
| Advantage | Mức hơn/kém trung bình nhóm, chia độ lệch chuẩn | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| RLVR | RL với reward kiểm chứng tự động được (toán, code) | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| DPO, ORPO, KTO | Các phương pháp căn chỉnh theo sở thích (preference) | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Preference data (chosen / rejected) | Cặp câu được ưa thích / bị chê cho cùng prompt | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Reward hacking | Model lách luật để tăng reward mà không làm đúng việc | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| Unsloth Standby | Cơ chế dùng chung vùng nhớ giữa inference (vLLM) và train khi RL | [Reinforcement Learning](/reinforcement-learning) |

## Suy luận và sử dụng

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| Inference (suy luận) | Dùng model để sinh kết quả, không học thêm | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling), [Inference & API](/inference) |
| Autoregressive | Sinh từng token, nối vào chuỗi rồi đoán tiếp | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Prefill / decode | Pha xử lý prompt song song / pha sinh từng token | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Greedy / sampling | Luôn chọn token xác suất cao nhất / bốc ngẫu nhiên theo xác suất | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Temperature | Làm phân phối xác suất nhọn hơn hoặc phẳng hơn | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Top-k, top-p, min-p | Các cách cắt bớt token ứng viên trước khi bốc | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Repetition / presence / frequency penalty | Phạt token đã xuất hiện để giảm lặp | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Chat template | Quy tắc đổi danh sách tin nhắn thành chuỗi token có token điều khiển | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling), [Dữ liệu](/du-lieu) |
| `add_generation_prompt` | Thêm phần mở đầu lượt assistant ở cuối prompt | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Thinking / reasoning mode, `reasoning_effort` | Model sinh đoạn suy nghĩ trước khi trả lời; mức độ suy nghĩ | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling), [Inference & API](/inference) |
| Tool calling (function calling) | Model sinh yêu cầu gọi hàm (JSON), app chạy hàm và trả kết quả | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling), [Inference & API](/inference) |
| Self-healing tool calling | Unsloth tự sửa tool call bị lỗi | [Inference & API](/inference) |
| Server-side tools | Công cụ Unsloth tự chạy (Python, bash, web search) | [Inference & API](/inference) |
| MCP (Model Context Protocol) | Giao thức chuẩn để model gọi dịch vụ bên ngoài | [Inference & API](/inference) |
| OpenAI-compatible API, endpoint | API cùng định dạng OpenAI; địa chỉ HTTP nhận request | [Inference & API](/inference) |
| API key (`sk-unsloth-…`), Bearer token | Khóa xác thực gửi qua header `Authorization: Bearer` | [Inference & API](/inference) |
| Streaming / SSE | Trả kết quả từng phần qua luồng sự kiện | [Inference & API](/inference) |
| Coding agent | Tác tử lập trình (Claude Code, Codex, OpenCode) | [Inference & API](/inference) |
| Prompt injection | Chèn lệnh độc hại vào nội dung model đọc | [Inference & API](/inference), [Ứng dụng RAG](/ung-dung-rag) |
| MTP, speculative decoding | Dự đoán/đoán trước nhiều token để decode nhanh hơn | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho), [Export & deploy](/export-deploy) |
| RAG | Truy xuất tài liệu liên quan rồi đưa vào prompt để trả lời | [Ứng dụng RAG](/ung-dung-rag) |
| Vector DB, reranker | Cơ sở dữ liệu vector; model xếp hạng lại kết quả truy xuất | [Ứng dụng RAG](/ung-dung-rag) |
| Hallucination | Model bịa thông tin | [Ứng dụng RAG](/ung-dung-rag) |

## Sản phẩm, công cụ và triển khai

| Thuật ngữ | Giải thích | Xem |
| --- | --- | --- |
| Unsloth Desktop | App native cài trên macOS, Windows, Linux | [Tổng quan](/tong-quan) |
| Unsloth Studio | Web UI no-code, cài thủ công | [Tổng quan](/tong-quan) |
| Unsloth Core | Thư viện Python gốc, dùng bằng code | [Tổng quan](/tong-quan) |
| `unsloth run`, `unsloth start` | Lệnh nạp model và mở API; lệnh nối coding agent với model local | [Inference & API](/inference) |
| safetensors | Định dạng lưu trọng số chuẩn của Hugging Face | [Export & deploy](/export-deploy) |
| llama.cpp, llama-server | Engine inference C/C++; server HTTP của nó | [Inference & API](/inference), [Export & deploy](/export-deploy) |
| vLLM, SGLang | Engine inference hiệu năng cao cho production | [Export & deploy](/export-deploy) |
| Ollama, LM Studio | Ứng dụng chạy model local | [Export & deploy](/export-deploy) |
| Modelfile | File cấu hình model của Ollama | [Export & deploy](/export-deploy) |
| MLX | Framework ML trên Apple Silicon | [Cài đặt](/cai-dat) |
| CUDA, ROCm, XPU/oneAPI | Nền tảng tính toán GPU của NVIDIA, AMD, Intel | [Cài đặt](/cai-dat) |
| WSL | Chạy Linux trong Windows | [Cài đặt](/cai-dat) |
| venv / uv / conda | Môi trường Python cô lập và công cụ quản lý gói | [Cài đặt](/cai-dat) |
| bitsandbytes, xformers, triton | Thư viện lượng tử hóa; thư viện kernel tăng tốc | [Cài đặt](/cai-dat) |
| Docker image / container | Môi trường đóng gói sẵn / bản đang chạy | [Cài đặt](/cai-dat) |
| LAN access, bind `0.0.0.0` | Mở server cho máy khác trong mạng nội bộ | [Export & deploy](/export-deploy) |
| Cloudflare tunnel, `--secure` | Link HTTPS công khai tới server local | [Export & deploy](/export-deploy) |
| Push to Hub | Đẩy model lên Hugging Face Hub | [Export & deploy](/export-deploy) |

**Nguồn:** thuật ngữ và giải thích tổng hợp từ các trang trên website; nguồn gốc của từng khái niệm ghi ở trang được trỏ tới. Danh sách URL đầy đủ: [Nguồn tham khảo](/nguon).
