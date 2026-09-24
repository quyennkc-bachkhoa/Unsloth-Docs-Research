---
title: Dense & MoE
description: Phân biệt model dense và Mixture of Experts (MoE), cách đọc ký hiệu kiểu 30B-A3B, vì sao MoE nhanh nhưng vẫn tốn bộ nhớ, offload expert và những gì Unsloth làm cho MoE.
---

# Dense & MoE

Trang này trả lời ba câu hỏi:

- Vì sao tên model có đuôi kiểu "35B-A3B"?
- Vì sao model MoE chạy nhanh hơn model dense cùng cỡ, nhưng vẫn cần đủ bộ nhớ cho toàn bộ trọng số?
- Khi fine-tune MoE bằng Unsloth, bạn phải chỉnh gì?

Bạn nên đọc trang này sau [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) và trước khi chọn model ở [/model-catalog](/model-catalog).

## Dense model

**Khái niệm.** Model **dense** (đặc) dùng **toàn bộ** tham số cho mỗi token. Mỗi layer có đúng một khối FFN, và token nào cũng đi qua khối đó. Vì vậy số tham số ghi trên tên model (ví dụ "27B") cho bạn biết hai điều cùng lúc: lượng trọng số phải chứa trong bộ nhớ, và lượng tính toán cho mỗi token.

**Ví dụ.** Qwen3.5-27B là model dense. Theo bảng của docs Unsloth, bản BF16 cần 54 GB, bản 4-bit cần 17 GB. Xem [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5). **[Ước tính]** `27B × 2 byte (BF16) = 54 GB`, khớp bảng. Công thức `bộ nhớ ≈ số tham số × số byte mỗi tham số` có ở trang [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho). HF dùng cùng cách tính cho Mixtral: 45B tham số × 2 byte ≈ 90 GB ở half precision (https://huggingface.co/docs/transformers/en/model_doc/mixtral).

**Ảnh hưởng khi dùng Unsloth.** Model dense fine-tune theo cách chuẩn, với `target_modules` gồm `gate_proj`, `up_proj`, `down_proj`. Riêng QLoRA 4-bit cho Qwen3.5 dense, docs chưa nhất quán. Code mẫu trong docs Qwen3.5 ghi QLoRA 4-bit "is fine" cho 27B. Nhưng một đoạn khác lại không khuyến nghị QLoRA cho mọi model Qwen3.5 (xem hộp cảnh báo cuối trang).

**Gặp ở đâu trong Unsloth.** [/model-catalog](/model-catalog), [/fine-tuning](/fine-tuning).

**Nguồn:** Unsloth Qwen3.5; Unsloth Qwen3.5 Fine-tuning Guide; HF Mixtral model doc.

## Mixture of Experts (MoE): expert và router

**Khái niệm.** Model MoE có nhiều khối FFN thay vì một, và với mỗi token chỉ dùng vài khối trong số đó. Cụ thể, model **MoE** (Mixture of Experts, hỗn hợp chuyên gia) thay khối FFN trong mỗi layer (hoặc một số layer) bằng **nhiều FFN song song**. Mỗi FFN gọi là một **expert** (chuyên gia).

Một mạng nhỏ gọi là **router** hay **gating network** (mạng cổng) quyết định gửi mỗi token tới expert nào. Chỉ các expert được chọn mới tính toán cho token đó. Phần attention vẫn dùng chung. Cách làm này gọi là "conditional computation" (tính toán có điều kiện): mỗi đầu vào chỉ kích hoạt một phần mạng. **[Nguồn ngoài]** https://arxiv.org/abs/1701.06538 , https://huggingface.co/blog/moe

Paper Sparsely-Gated MoE (2017) dùng tới hàng nghìn expert. Paper báo dung lượng model tăng hơn 1000 lần mà hiệu năng tính toán chỉ giảm ít. **[Nguồn ngoài]** https://arxiv.org/abs/1701.06538

**Ví dụ.** Trong Transformers v4, expert của Qwen3-MoE được khai báo đúng như "một danh sách FFN":

```python
self.experts = nn.ModuleList(
    [Qwen3MoeMLP(config, intermediate_size) for _ in range(self.num_experts)]
)
```

Từ Transformers v5, trọng số của tất cả expert được gộp thành một tensor duy nhất:

```python
self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))
```

(Hai đoạn code chép từ docs Unsloth [Faster MoE](https://unsloth.ai/docs/basics/faster-moe).)

**Ảnh hưởng khi dùng Unsloth.** Expert là FFN, nên tên module LoRA cho expert là `gate_up_proj` và `down_proj`. Chi tiết ở mục "Unsloth làm gì cho MoE" bên dưới.

**Gặp ở đâu trong Unsloth.** [/fine-tuning](/fine-tuning), [Faster MoE](https://unsloth.ai/docs/basics/faster-moe).

**Nguồn:** arXiv 1701.06538; HF blog MoE (chỉ diễn giải khái niệm); Unsloth Faster MoE.

### Top-k routing

**Khái niệm.** Top-k routing là cách router chọn expert: chấm điểm tất cả, rồi giữ k expert điểm cao nhất. Các bước:

1. Router cho ra một điểm cho mỗi expert (router logits). Điểm đi qua softmax để thành xác suất.
2. Router giữ **k** expert có điểm cao nhất.
3. Đầu ra của token là tổ hợp có trọng số của k expert đó.

k thường rất nhỏ so với tổng số expert. **[Nguồn ngoài]** https://huggingface.co/blog/moe. Switch Transformers đơn giản hóa xuống **k = 1**, tức mỗi token một expert, để giảm chi phí giao tiếp và tính toán. Nhờ đó họ train model hàng nghìn tỷ tham số mà chi phí tính toán mỗi token không đổi. **[Nguồn ngoài]** https://arxiv.org/abs/2101.03961

Router còn phải cân bằng tải. Nếu token dồn vào vài expert "được ưa chuộng", các expert khác bị bỏ phí. Khi train, người ta thêm **auxiliary loss** (loss phụ) để phạt việc phân phối lệch. **[Nguồn ngoài]** https://huggingface.co/blog/moe. Trong Transformers, tham số này là `router_aux_loss_coef`, mặc định `0.001` ở Qwen3MoE và Mixtral. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe

**Ví dụ (số từ docs Unsloth).**

| Model | Tổng expert | Kích hoạt mỗi token | Nguồn |
|---|---|---|---|
| Qwen3-30B-A3B | 128 | 8 | [Faster MoE](https://unsloth.ai/docs/basics/faster-moe) |
| gpt-oss-20b | 32 | 4 | [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) |
| gpt-oss-120b | 128 | 4 | [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) |
| GLM-4.7-Flash | 64 routed expert + 1 shared expert | docs không ghi | [Faster MoE](https://unsloth.ai/docs/basics/faster-moe) |

**Shared expert** (expert dùng chung) là expert mà mọi token đều đi qua, bên cạnh các expert do router chọn. Docs Unsloth gọi đây là cấu hình "kiểu DeepSeek MoE". Mixtral 8x7B (không có trong bảng trên) dùng 8 expert, top-2. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/en/model_doc/mixtral

Trong config HF, `num_experts` là tổng số expert, còn `num_experts_per_tok` là giá trị k. Qwen3MoE mặc định `num_experts = 128`, `num_experts_per_tok = 8`. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe

**Ảnh hưởng khi dùng Unsloth.** Docs Unsloth nói fine-tune router layer "không phải ý hay". Vì vậy Unsloth **tắt mặc định** việc train router. Điều này được nhắc ở Faster MoE, ở Qwen3.5 Fine-tuning ("vì ổn định") và ở Nemotron 3. Không có lý do để tự bật. **[Nhận định]**

**Gặp ở đâu trong Unsloth.** [/fine-tuning](/fine-tuning), [/model-catalog](/model-catalog).

**Nguồn:** HF blog MoE; arXiv 2101.03961; HF Qwen3MoE, Mixtral model doc; Unsloth Faster MoE, gpt-oss, Qwen3.5 Fine-tuning, [Nemotron 3](https://unsloth.ai/docs/models/nemotron-3).

## Tổng tham số và tham số kích hoạt

**Khái niệm.** Model MoE có hai con số tham số, và mỗi con số trả lời một câu hỏi khác nhau:

- **Tổng tham số** (total parameters): tất cả trọng số của model, gồm mọi expert. Con số này quyết định **bộ nhớ** cần để nạp model.
- **Tham số kích hoạt** (active parameters): số tham số thực sự tham gia tính toán cho **một token**. Nó gồm phần dùng chung (embedding, attention, norm, shared expert) cộng k expert được chọn. Con số này quyết định **lượng tính toán** mỗi token.

Với model dense, hai con số bằng nhau. Với MoE, tham số kích hoạt nhỏ hơn nhiều.

**Ví dụ.**
- Qwen3-30B-A3B có 30.5B tham số tổng và 3.3B tham số kích hoạt mỗi token (128 expert, chọn 8, 48 layer). **[Nguồn ngoài]** https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe.
- Tên "8x7B" của Mixtral **không** có nghĩa là 56B. Tổng chỉ khoảng 45B, vì chỉ FFN được nhân thành 8 expert, còn attention dùng chung. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/en/model_doc/mixtral , https://huggingface.co/blog/moe.
- Với gpt-oss-20b, docs Unsloth ghi các expert MoE/MLP chiếm khoảng 19B trong 20B tham số.

**Ảnh hưởng khi dùng Unsloth.** Khi xem bảng bộ nhớ trong docs Unsloth, luôn đọc theo **tổng tham số**, không theo phần "A..." (xem mục tiếp theo).

**Gặp ở đâu trong Unsloth.** [/model-catalog](/model-catalog) (cột "Tổng tham số / Tham số kích hoạt").

**Nguồn:** HF Qwen3MoE, Mixtral model doc; HF blog MoE; Unsloth gpt-oss.

## Đọc ký hiệu "30B-A3B", "35B-A3B", "26B-A4B"

**Khái niệm.** Tên model MoE thường ghi cả hai con số tham số. Số đầu là tổng tham số. Số sau chữ **A** (Active) là tham số kích hoạt mỗi token. "Qwen3-30B-A3B" nghĩa là khoảng 30B tổng và khoảng 3B kích hoạt. Con số trong tên đã được làm tròn: HF ghi 30.5B và 3.3B cho Qwen3-30B-A3B. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe

**Ví dụ (tên model lấy từ docs Unsloth).**

| Tên | Tổng | Kích hoạt | Ghi chú trong docs Unsloth |
|---|---|---|---|
| Qwen3-30B-A3B | ~30B | ~3B | 128 expert, chọn 8 ([Faster MoE](https://unsloth.ai/docs/basics/faster-moe)) |
| Qwen3.5-35B-A3B | ~35B | ~3B | BF16 70 GB, 4-bit 22 GB ([Qwen3.5](https://unsloth.ai/docs/models/qwen3.5)) |
| Qwen3.5-122B-A10B | ~122B | ~10B | BF16 245 GB, 4-bit 70 GB |
| Qwen3.5-397B-A17B | ~397B | ~17B | BF16 810 GB, 4-bit 214 GB |
| Gemma-4-26B-A4B | ~26B | ~4B | "4B active parameters", 4-bit 16–18 GB ([Gemma 4](https://unsloth.ai/docs/models/gemma-4)) |
| Qwen3-235B-A22B | ~235B | ~22B | [Qwen3-2507](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-2507) |
| Qwen3.8-2.4T-A95B | 2.4T | 95B | "2.4T parameter (95B active)" ([Qwen3.8](https://unsloth.ai/docs/models/qwen3.8)) |
| GLM-5.3 | 744B | 40B | Tên không có đuôi A; docs ghi "744B parameter (40B active)" ([GLM-5.3](https://unsloth.ai/docs/models/glm-5.3)) |

Không phải tên nào cũng theo đúng mẫu này:

- Một số tên chỉ ghi phần kích hoạt, ví dụ "Hunyuan-A13B" trong [model catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog).
- gpt-oss không có đuôi A; docs chỉ ghi số expert.
- Không có đuôi A cũng chưa chắc là model dense: GLM-5.3 và gpt-oss đều là MoE.

**Ảnh hưởng khi dùng Unsloth.** Nhìn đuôi A để ước lượng **tốc độ**, nhìn số đầu để ước lượng **bộ nhớ**. Hai trang docs Unsloth so sánh cụ thể:

- Docs Qwen3.5: giữa 27B (dense) và 35B-A3B (MoE), chọn 27B nếu muốn kết quả chính xác hơn một chút, chọn 35B-A3B nếu muốn suy luận nhanh hơn nhiều.
- Docs Gemma 4: 26B-A4B nhanh hơn 31B nhờ MoE với 4B tham số kích hoạt, đổi lại chất lượng thấp hơn một chút.

**Gặp ở đâu trong Unsloth.** [/model-catalog](/model-catalog), [/inference](/inference).

**Nguồn:** Unsloth Qwen3.5, Gemma 4, Qwen3-2507, Qwen3.8, GLM-5.3, Model Catalog, Faster MoE; HF Qwen3MoE model doc.

## Vì sao MoE nhanh nhưng vẫn tốn bộ nhớ

**Khái niệm.** MoE suy luận nhanh như model nhỏ, nhưng cần bộ nhớ như model lớn. Lý do nằm ở hai điểm:

- Mỗi token chỉ đi qua k expert. Vì vậy số phép tính (FLOPs) mỗi token tỉ lệ với **tham số kích hoạt**, không phải tổng tham số.
- Router có thể chọn expert **bất kỳ** cho token tiếp theo. Vì vậy **mọi** expert phải nằm sẵn trong bộ nhớ.

**[Nguồn ngoài]** https://huggingface.co/blog/moe. HF mô tả Mixtral như sau: model có 45B tham số, nhưng lượng tính toán một lượt forward tương đương model 14B. Lý do là mỗi token chỉ được gửi tới 2 expert, trong khi mọi expert đều phải nạp vào RAM. **[Nguồn ngoài]** https://huggingface.co/docs/transformers/en/model_doc/mixtral

**Ví dụ.** Theo bảng docs Unsloth Qwen3.5:

| Model | Kiểu | 4-bit | BF16 |
|---|---|---|---|
| Qwen3.5-27B | Dense | 17 GB | 54 GB |
| Qwen3.5-35B-A3B | MoE | 22 GB | 70 GB |

**[Ước tính]** 35B-A3B cần nhiều bộ nhớ hơn 27B: ở BF16 là `35B × 2 byte = 70 GB` so với `27B × 2 byte = 54 GB`. Nhưng mỗi token chỉ tính khoảng 3B tham số so với 27B, tức ít hơn khoảng 9 lần. Phép chia `27 / 3 ≈ 9` dựa trên nhận xét "compute tỉ lệ với số tham số được dùng mỗi token" của HF Mixtral. Tốc độ thực tế còn phụ thuộc băng thông bộ nhớ, attention và context, nên không nhanh đúng 9 lần. **[Nhận định]**

**Ảnh hưởng khi dùng Unsloth.** Docs Unsloth lặp lại một quy tắc ở nhiều trang: tổng bộ nhớ khả dụng (VRAM + RAM hệ thống) nên lớn hơn kích thước file quant bạn tải. Nếu không đủ, llama.cpp vẫn chạy được nhờ offload ra SSD hoặc HDD, nhưng chậm hơn. Với MoE, quy tắc này áp dụng cho **tổng** tham số. Cách chọn quant: [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).

**Gặp ở đâu trong Unsloth.** [/model-catalog](/model-catalog), [/inference](/inference).

**Nguồn:** HF blog MoE (khái niệm); HF Mixtral model doc; Unsloth Qwen3.5, gpt-oss.

## Offload expert sang RAM/CPU

**Khái niệm.** Offload expert là cách chạy model MoE lớn khi GPU không đủ VRAM. Vì mỗi token chỉ dùng vài expert, bạn có thể giữ **phần dùng chung** (attention, norm, embedding) trên GPU và đẩy **trọng số expert** sang RAM hệ thống cho CPU tính. GPU chỉ cần chứa phần nhỏ, còn phần lớn nằm ở RAM rẻ hơn. **[Nhận định]** (giải thích cơ chế; cách làm cụ thể lấy từ docs Unsloth bên dưới).

**Ví dụ.** Docs Unsloth dùng cờ `-ot` (override tensor) của llama.cpp:

```bash
-ot ".ffn_.*_exps.=CPU"
```

Theo docs, cờ này offload tất cả lớp MoE sang CPU. Nhờ vậy toàn bộ phần không phải MoE nằm vừa trên 1 GPU, và tốc độ sinh tăng. Nếu còn dư VRAM, bạn thu hẹp regex để giữ thêm expert trên GPU:

- `-ot ".ffn_(up|down)_exps.=CPU"`: chỉ offload expert up và down projection.
- `-ot ".ffn_(up)_exps.=CPU"`: chỉ offload expert up projection.
- Regex theo số layer, ví dụ chỉ offload expert gate, up, down từ layer thứ 6 trở đi.

Hai ví dụ thực tế trong docs:

- Qwen3.5-397B-A17B chạy được trên **1 GPU 24 GB + 256 GB RAM** nhờ MoE offloading, đạt 25+ token/s theo docs [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5).
- Unsloth Desktop tự offload sang RAM và tự nhận nhiều GPU, theo docs [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8) và [GLM-5.3](https://unsloth.ai/docs/models/glm-5.3).

**Ảnh hưởng khi dùng Unsloth.** Bạn cần tìm mức offload vừa phải:

- Offload quá nhiều thì chậm, vì CPU phải tính và dữ liệu phải đi qua RAM.
- Offload quá ít thì GPU hết bộ nhớ (OOM).

Theo gợi ý của docs, bắt đầu với `-ot ".ffn_.*_exps.=CPU"`, rồi giảm dần phần offload khi còn VRAM.

**Gặp ở đâu trong Unsloth.** [/inference](/inference); [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune), [Qwen3-2507](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-2507).

**Nguồn:** Unsloth gpt-oss, Qwen3-2507, Qwen3.5, Qwen3.8, GLM-5.3.

## Unsloth làm gì cho MoE (Faster MoE)

**Khái niệm.** Unsloth có một nhóm tối ưu riêng để train MoE nhanh hơn và ít tốn VRAM hơn. Trang [Faster MoE](https://unsloth.ai/docs/basics/faster-moe) mô tả ba thay đổi:

1. **Grouped GEMM**: trước đây expert là `ModuleList`, nên forward phải chạy vòng `for` qua từng expert. Unsloth dùng `torch._grouped_mm` của PyTorch để tính nhiều expert trong một lần gọi, hoặc dùng Triton kernel riêng của Unsloth.
2. **Split LoRA**: PEFT thường gộp LoRA vào trọng số gốc trước khi tính MoE. Như vậy phải tạo ra ma trận `lora_B @ lora_A` cho **mọi** expert, rất tốn bộ nhớ. Unsloth đổi thứ tự phép nhân (nhờ tính kết hợp của phép nhân ma trận) để chỉ tính cho các cặp token–expert mà router chọn. Docs khẳng định loss, gradient và đầu ra giữ nguyên.
3. **Tự chọn backend** theo phần cứng.

| Backend | Theo docs |
|---|---|
| `grouped_mm` | `torch._grouped_mm`, chạy từ T4 đến B200, tối ưu cho H100 trở lên; mặc định |
| `unsloth_triton` | Triton kernel của Unsloth, tự bật trên A100 và PyTorch cũ |
| `native_torch` | PyTorch thuần, chậm hơn 12 lần nhưng vẫn giữ phần giảm VRAM |

```python
os.environ["UNSLOTH_MOE_BACKEND"] = "grouped_mm"
os.environ["UNSLOTH_MOE_BACKEND"] = "unsloth_triton"
os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
```

**Ví dụ: vì sao LoRA trên MoE tốn hơn dense.** Docs lấy Qwen3-30B-A3B làm ví dụ, với các thông số:

- hidden size `m = 2048`;
- intermediate size của mỗi expert `n = 768`;
- `E = 128` expert, `k = 8` kích hoạt mỗi token.

Với LoRA rank `r = 64`, mỗi projection của mỗi expert thêm `r*(m+n) = 64*(2048+768) = 180,224` tham số. Con số này bằng khoảng 11% ma trận `2048×768`. Tỉ lệ `r/n = 64/768` lớn hơn nhiều so với model dense cỡ tương đương (`r/n = 64/25600` ở Qwen3-32B). Nhân lên 128 expert thì bộ nhớ tăng nhanh. Đó là lý do cần Split LoRA.

**Ví dụ: kết quả đo trong docs (kèm điều kiện).**

- Tiêu đề trang ghi: MoE train nhanh khoảng **12 lần**, ít hơn **35%** VRAM, context dài khoảng **6 lần**. Docs tách rõ phần đóng góp: Transformers v5 đã nhanh khoảng 6 lần so với v4 cho MoE; Unsloth thêm khoảng 2 lần so với v5; tổng cộng 12–30 lần so với Transformers v4.
- Benchmark so với Transformers v5 (bản đã dùng `torch._grouped_mm`). Docs ghi dùng LoRA rank 64 trên các module MoE (gate, up, down):
  - **gpt-oss-20b BF16, NVIDIA B200**: ở context 8192, nhanh 7.3 lần và tiết kiệm 35.73% VRAM (47.43 GB so với 73.80 GB). Ở context 16384, Unsloth dùng 55.13 GB còn Transformers v5 bị OOM. Ở context 1024 chỉ nhanh 1.4 lần và tiết kiệm 6.76%. Nghĩa là lợi ích tăng theo độ dài context.
  - **Qwen3-30B-A3B, H100**: nhanh tới 1.77 lần, tiết kiệm khoảng 5.3 GB ở context 4K. Transformers v5 + TRL bị OOM ở 8K.
  - **GLM-4.7-Flash** (30B MoE, 3B active): throughput nhanh 2.6 lần, ít hơn 15% VRAM trên mọi batch size.
- Trên A100, Triton kernel của Unsloth nhanh khoảng 2.5 lần so với `torch._grouped_mm`. Bước autotune mất khoảng 2 phút lúc đầu, nhưng có thể làm cả lượt train nhanh hơn 35%.
- Mức VRAM: gpt-oss-20b fine-tune trong 12.8 GB VRAM. Qwen3-30B-A3B (LoRA 16-bit) dùng 63 GB. Qwen3.5-35B-A3B bf16 LoRA cần 74 GB. Qwen3.5-122B-A10B bf16 LoRA cần 256 GB ([Qwen3.5 Fine-tuning](https://unsloth.ai/docs/models/qwen3.5/fine-tune)).

**Ảnh hưởng khi dùng Unsloth.**
- **Dùng bf16 (LoRA 16-bit hoặc full fine-tuning), không dùng QLoRA 4-bit cho MoE.** Lý do: BitsandBytes chưa hỗ trợ 4-bit cho trọng số expert dạng `nn.Parameter`. Docs ghi đây không phải hạn chế riêng của Unsloth. Code mẫu đặt `load_in_4bit = False`.
- `target_modules` cho MoE (chép từ docs):

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_up_proj", "down_proj", # LoRA on MoE layers!
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)
```

- Router không được train (tắt mặc định).
- Các tối ưu MoE **bật mặc định**. Để có chúng, bạn cập nhật bằng `pip install --upgrade unsloth unsloth_zoo`. Chỉ đổi `UNSLOTH_MOE_BACKEND` khi gặp lỗi tương thích.
- Riêng gpt-oss, docs ghi hai điểm:
  - Nếu dùng float16 làm kiểu autocast, sau một thời gian sẽ sinh ra giá trị vô cực. Vì vậy Unsloth tính phần MoE bằng bfloat16. GPU không hỗ trợ bfloat16 (như T4) thì dùng float32.
  - Trọng số gpt-oss phát hành ở MXFP4 dạng `nn.Parameter`. Unsloth chuyển sang `nn.Linear` để lượng tử hóa BitsandBytes được, chấp nhận chậm hơn một chút.
- Model được hỗ trợ theo docs:
  - Qwen3 (Thinking, Instruct, VL, 2507, Coder);
  - gpt-oss (20B, 120B, safeguard);
  - GLM (4.5, 4.6, 4.6-Air, 4.7, 4.7-Flash);
  - DeepSeek (V3, R1, V3.1, V3.2);
  - Qwen3.5-35B-A3B, 122B-A10B, 397B-A17B dùng cùng bản cập nhật này ([Qwen3.5 Fine-tuning](https://unsloth.ai/docs/models/qwen3.5/fine-tune)).

**Gặp ở đâu trong Unsloth.** [/fine-tuning](/fine-tuning); cơ chế LoRA: [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora).

**Nguồn:** Unsloth Faster MoE; Unsloth Qwen3.5 Fine-tuning Guide; Unsloth gpt-oss.

::: warning Docs chưa thống nhất
Trang [Faster MoE](https://unsloth.ai/docs/basics/faster-moe) và các trang liên quan đưa ra con số khác nhau cho cùng một việc ở vài chỗ. Bên dưới ghi đủ các phiên bản để bạn tự đối chiếu:

- **Qwen3-30B-A3B nhanh bao nhiêu:** đoạn mở đầu benchmark ghi "1.8x". Mục Qwen3 ghi "~1.7x" trên B200 và "up to 1.77x" trên H100. Bảng số liệu ghi 1.7x ở context 1024, giảm còn 1.1x ở 16384.
- **GLM-4.7-Flash:** đoạn mở đầu ghi "2.1x faster on RTX PRO 6000". Mục GLM 4.7 ghi "2.6x faster throughput" (bảng: 2.6x ở 512–1024, 2x ở 4096) và không ghi GPU.
- **LoRA rank khi đo:** văn bản ghi "LoRA rank = 64", nhưng cột ẩn "Rank" trong các bảng ghi 8.
- **Tiết kiệm VRAM:** tiêu đề ghi `>35%`. Nhưng bảng Qwen3-30B-A3B chỉ 2.06%–15.26%, bảng GLM 6.22%–14.83%. Mức ~36% chỉ đạt với gpt-oss ở context 8K.
- **VRAM fine-tune gpt-oss-20b:** Faster MoE ghi 12.8 GB, còn trang [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) ghi 14 GB.
- **QLoRA cho Qwen3.5 dense:** trang [Qwen3.5 Fine-tuning](https://unsloth.ai/docs/models/qwen3.5/fine-tune) ghi không khuyến nghị QLoRA 4-bit cho mọi model Qwen3.5 "no matter MoE or dense". Nhưng comment trong code mẫu lại ghi "dense 27B is fine".
:::

## Gặp ở đâu trong Unsloth

| Khái niệm | Trang Unsloth trên website | Docs gốc |
|---|---|---|
| Dense vs MoE, chọn model | [/model-catalog](/model-catalog) | [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5), [Gemma 4](https://unsloth.ai/docs/models/gemma-4) |
| Expert, router, top-k | [/fine-tuning](/fine-tuning) | [Faster MoE](https://unsloth.ai/docs/basics/faster-moe), [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) |
| Tổng tham số vs kích hoạt, ký hiệu "A3B" | [/model-catalog](/model-catalog) | [Model Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog), [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8), [GLM-5.3](https://unsloth.ai/docs/models/glm-5.3) |
| Bộ nhớ MoE, offload expert (`-ot`) | [/inference](/inference) | [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune), [Qwen3-2507](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-2507), [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5) |
| Faster MoE, `UNSLOTH_MOE_BACKEND`, `gate_up_proj` | [/fine-tuning](/fine-tuning) | [Faster MoE](https://unsloth.ai/docs/basics/faster-moe), [Qwen3.5 Fine-tuning](https://unsloth.ai/docs/models/qwen3.5/fine-tune) |
