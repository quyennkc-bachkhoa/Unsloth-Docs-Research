// Công thức ước tính bộ nhớ dùng cho VramEstimator.vue.
// Nguồn từng thành phần ghi trong docs/kien-thuc-nen/tham-so-va-bo-nho.md.

export type Precision = 'bf16' | 'fp8' | 'int8' | '4bit'
export type Mode = 'run' | 'lora' | 'qlora'

export interface Arch {
  layers: number
  hidden: number
  intermediate: number
  heads: number
  kvHeads: number
  headDim: number
  vocab: number
  tiedEmbeddings: boolean
}

export interface Preset extends Arch {
  id: string
  label: string
  source: string
}

// Lấy từ config.json chính thức trên Hugging Face.
export const PRESETS: Preset[] = [
  { id: 'llama-3.2-3b', label: 'Llama 3.2 3B', layers: 28, hidden: 3072, intermediate: 8192, heads: 24, kvHeads: 8, headDim: 128, vocab: 128256, tiedEmbeddings: true, source: 'https://huggingface.co/unsloth/Llama-3.2-3B-Instruct/blob/main/config.json' },
  { id: 'llama-3.1-8b', label: 'Llama 3.1 8B', layers: 32, hidden: 4096, intermediate: 14336, heads: 32, kvHeads: 8, headDim: 128, vocab: 128256, tiedEmbeddings: false, source: 'https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/blob/main/config.json' },
  { id: 'qwen3-14b', label: 'Qwen3 14B', layers: 40, hidden: 5120, intermediate: 17408, heads: 40, kvHeads: 8, headDim: 128, vocab: 151936, tiedEmbeddings: false, source: 'https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json' },
  { id: 'qwen3-32b', label: 'Qwen3 32B', layers: 64, hidden: 5120, intermediate: 25600, heads: 64, kvHeads: 8, headDim: 128, vocab: 151936, tiedEmbeddings: false, source: 'https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json' },
  { id: 'llama-3.3-70b', label: 'Llama 3.3 70B', layers: 80, hidden: 8192, intermediate: 28672, heads: 64, kvHeads: 8, headDim: 128, vocab: 128256, tiedEmbeddings: false, source: 'https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/config.json' }
]

// Số byte mỗi tham số theo độ rộng bit của kiểu dữ liệu (16, 8, 4 bit).
export const BYTES_PER_PARAM: Record<Precision, number> = { bf16: 2, fp8: 1, int8: 1, '4bit': 0.5 }

const GB = 1e9

/** Đếm tham số của một decoder-only dense (Llama/Qwen): embedding + attention + MLP SwiGLU + norm. */
export function countParams(a: Arch): number {
  const q = a.hidden * a.heads * a.headDim
  const kv = 2 * a.hidden * a.kvHeads * a.headDim
  const o = a.heads * a.headDim * a.hidden
  const mlp = 3 * a.hidden * a.intermediate
  const norms = 2 * a.hidden
  const embed = a.vocab * a.hidden * (a.tiedEmbeddings ? 1 : 2)
  return a.layers * (q + kv + o + mlp + norms) + embed + a.hidden
}

/** Số tham số LoRA khi gắn vào đủ 7 target module: r × (d_in + d_out) cho mỗi ma trận. */
export function loraParams(a: Arch, r: number): number {
  const qOut = a.heads * a.headDim
  const kvOut = a.kvHeads * a.headDim
  const perLayer =
    r * (a.hidden + qOut) + // q_proj
    2 * r * (a.hidden + kvOut) + // k_proj, v_proj
    r * (qOut + a.hidden) + // o_proj
    2 * r * (a.hidden + a.intermediate) + // gate_proj, up_proj
    r * (a.intermediate + a.hidden) // down_proj
  return a.layers * perLayer
}

export interface Input {
  paramsB: number // tổng tham số, tỷ
  precision: Precision // chỉ dùng ở chế độ chạy
  context: number // số token
  batch: number
  mode: Mode
  arch: Arch
  loraRank: number
  overheadPct: number // phần phụ, % của tổng các phần trên
}

export interface Breakdown {
  weights: number
  kvCache: number
  trainable: number
  activations: number
  overhead: number
  total: number
  loraParams: number
}

export function estimate(i: Input): Breakdown {
  const P = i.paramsB * 1e9
  const a = i.arch
  let weights = 0
  let kvCache = 0
  let trainable = 0
  let activations = 0
  let lp = 0

  if (i.mode === 'run') {
    weights = P * BYTES_PER_PARAM[i.precision]
    // 2 (K và V) × số layer × số KV head × head_dim × số token × 2 byte (FP16)
    kvCache = 2 * a.layers * a.kvHeads * a.headDim * i.context * i.batch * 2
  } else {
    // LoRA: model gốc 16-bit; QLoRA: model gốc 4-bit (đóng băng, không có gradient)
    weights = P * (i.mode === 'qlora' ? BYTES_PER_PARAM['4bit'] : BYTES_PER_PARAM.bf16)
    lp = loraParams(a, i.loraRank)
    // Tham số được train: 6 byte trọng số mixed precision + 4 byte gradient + 2 byte AdamW 8-bit
    trainable = lp * (6 + 4 + 2)
    // Gradient checkpointing: giữ đầu vào mỗi layer (2 byte) + activation đầy đủ của 1 layer khi tính lại
    const sbh = i.context * i.batch * a.hidden
    activations = a.layers * sbh * 2 + 34 * sbh
  }
  const sub = weights + kvCache + trainable + activations
  const overhead = (sub * i.overheadPct) / 100
  return {
    weights: weights / GB,
    kvCache: kvCache / GB,
    trainable: trainable / GB,
    activations: activations / GB,
    overhead: overhead / GB,
    total: (sub + overhead) / GB,
    loraParams: lp
  }
}
