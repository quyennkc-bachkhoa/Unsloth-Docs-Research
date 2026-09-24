<script setup lang="ts">
import { withBase } from 'vitepress'

const legend = [
  {
    kind: 'docs',
    name: 'Nguồn',
    meaning: 'Lấy từ docs chính thức hoặc GitHub của Unsloth. Cuối mỗi mục có dòng liệt kê URL.',
    example: 'Studio đặt sẵn learning rate 2e-4 và LoRA rank 16.',
    where: '/tong-quan'
  },
  {
    kind: 'ext',
    name: 'Nguồn ngoài',
    meaning: 'Lấy từ paper arXiv hoặc tài liệu chính thức của Hugging Face, PyTorch, llama.cpp, NVIDIA.',
    example: 'LoRA chỉ train hai ma trận hạng thấp A và B, giữ nguyên trọng số gốc.',
    where: '/kien-thuc-nen/lora-va-qlora'
  },
  {
    kind: 'estimate',
    name: 'Ước tính',
    meaning: 'Con số tự tính từ một công thức có nguồn, không phải số đo.',
    example: 'Model 8B ở BF16 cần khoảng 8 tỷ × 2 byte = 16 GB cho trọng số.',
    where: '/kien-thuc-nen/tham-so-va-bo-nho'
  },
  {
    kind: 'opinion',
    name: 'Nhận định',
    meaning: 'Ý kiến của người viết, không có trong nguồn nào.',
    example: 'Với dữ liệu thay đổi hằng ngày, RAG thường thực tế hơn fine-tune lại liên tục.',
    where: '/fine-tuning/'
  },
  {
    kind: 'conflict',
    name: 'Docs chưa thống nhất',
    meaning: 'Các trang docs ghi giá trị khác nhau. Website liệt kê đủ, kèm link, không tự chọn.',
    example: 'lora_alpha mặc định: 32 trong Studio, 16 trong notebook Core.',
    where: '/fine-tuning/hyperparameter'
  }
]

const unsloth = [
  {
    group: 'Nhập môn',
    items: [
      { t: 'Tổng quan kiến trúc', d: 'Desktop, Studio, Core; CLI và API', l: '/tong-quan' },
      { t: 'Lộ trình học', d: 'Năm bước từ đọc nền đến deploy', l: '/lo-trinh-hoc' }
    ]
  },
  {
    group: 'Sử dụng',
    items: [
      { t: 'Cài đặt & phần cứng', d: 'Windows, macOS, Linux, AMD, Intel, CPU', l: '/cai-dat' },
      { t: 'Inference & API', d: 'Chạy model, endpoint /v1, agent, MCP', l: '/inference' },
      { t: 'Model catalog', d: 'Tham số, kiến trúc, bộ nhớ theo docs', l: '/model-catalog' }
    ]
  },
  {
    group: 'Huấn luyện',
    items: [
      { t: 'Fine-tuning', d: 'LoRA, QLoRA, full; hyperparameter', l: '/fine-tuning/' },
      { t: 'Reinforcement Learning', d: 'GRPO, DPO, ORPO, KTO', l: '/reinforcement-learning' },
      { t: 'Dữ liệu', d: 'Định dạng dataset, chat template, Data Recipes', l: '/du-lieu' }
    ]
  },
  {
    group: 'Triển khai & ứng dụng',
    items: [
      { t: 'Export & deploy', d: 'GGUF, NVFP4; Ollama, vLLM; LAN', l: '/export-deploy' },
      { t: 'Ứng dụng RAG', d: 'Chatbot RAG chạy local', l: '/ung-dung-rag' }
    ]
  }
]

const nen = [
  {
    group: 'A. Cấu tạo mô hình',
    items: [
      { t: 'Token & context', d: 'Tokenizer, context window', l: '/kien-thuc-nen/token-va-context' },
      { t: 'Phân loại mô hình', d: 'Base, instruct, vision, embedding', l: '/kien-thuc-nen/phan-loai-mo-hinh' },
      { t: 'Kiến trúc Transformer', d: 'Attention, FFN, RoPE, layer', l: '/kien-thuc-nen/kien-truc-transformer' },
      { t: 'Dense & MoE', d: 'Expert, "30B-A3B"', l: '/kien-thuc-nen/dense-va-moe' }
    ]
  },
  {
    group: 'B. Số học & bộ nhớ',
    items: [
      { t: 'Tham số & bộ nhớ', d: 'Ước tính VRAM, đọc tên model', l: '/kien-thuc-nen/tham-so-va-bo-nho' },
      { t: 'Độ chính xác & lượng tử hóa', d: 'BF16, FP8, NVFP4, GGUF quant', l: '/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa' }
    ]
  },
  {
    group: 'C. Huấn luyện',
    items: [
      { t: 'Quá trình huấn luyện', d: 'Loss, optimizer, learning rate', l: '/kien-thuc-nen/qua-trinh-huan-luyen' },
      { t: 'LoRA & QLoRA', d: 'Rank, alpha, merge', l: '/kien-thuc-nen/lora-va-qlora' },
      { t: 'RL & preference', d: 'Reward, DPO, GRPO', l: '/kien-thuc-nen/rl-va-preference' }
    ]
  },
  {
    group: 'D. Suy luận',
    items: [{ t: 'Suy luận & sampling', d: 'KV cache, temperature, tool calling', l: '/kien-thuc-nen/suy-luan-va-sampling' }]
  }
]

const steps = [
  { t: 'Đọc kiến thức nền', d: 'Nhóm A và B trước khi cài' },
  { t: 'Cài đặt Unsloth', d: 'Chọn Desktop, Studio hoặc Core' },
  { t: 'Chạy model và API', d: 'Chat, gọi endpoint /v1' },
  { t: 'Dữ liệu và fine-tune', d: 'Đọc nhóm C trước bước này' },
  { t: 'Export và deploy', d: 'GGUF, Ollama, vLLM, LAN' }
]
</script>

<template>
  <div class="rh">
    <header class="rh-head">
      <h1 class="rh-title">Unsloth Research</h1>
      <p class="rh-lead">
        Unsloth là framework mã nguồn mở để chạy và huấn luyện LLM trên phần cứng của bạn. Website này ghi lại cách dùng
        nó bằng tiếng Việt, và cho biết mỗi câu đến từ đâu.
      </p>
      <div class="rh-actions">
        <a class="rh-btn rh-btn-primary" :href="withBase('/tong-quan')">Đọc tổng quan Unsloth</a>
        <a class="rh-btn" :href="withBase('/kien-thuc-nen/')">Học kiến thức nền trước</a>
      </div>
    </header>

    <section class="rh-legend" aria-labelledby="rh-legend-h">
      <h2 id="rh-legend-h" class="rh-h2">Cách đọc một câu trên website</h2>
      <p class="rh-note">Mỗi thông tin mang một trong năm dấu dưới đây. Dấu cho biết nên tin câu đó đến đâu.</p>
      <dl class="rh-marks">
        <div v-for="m in legend" :key="m.kind" class="rh-mark-row" :data-kind="m.kind">
          <dt><span class="rh-mark">{{ m.name }}</span></dt>
          <dd>
            <p class="rh-meaning">{{ m.meaning }}</p>
            <p class="rh-example">
              Ví dụ: <a :href="withBase(m.where)">{{ m.example }}</a>
            </p>
          </dd>
        </div>
      </dl>
    </section>

    <section class="rh-routes" aria-label="Mục lục">
      <div class="rh-route">
        <h2 class="rh-h2">Unsloth</h2>
        <p class="rh-note">Thao tác: cài, chạy, fine-tune, triển khai. Chỉ dùng docs và GitHub của Unsloth.</p>
        <div v-for="g in unsloth" :key="g.group" class="rh-group">
          <h3 class="rh-h3">{{ g.group }}</h3>
          <ul>
            <li v-for="i in g.items" :key="i.l">
              <a :href="withBase(i.l)"><span class="rh-t">{{ i.t }}</span><span class="rh-d">{{ i.d }}</span></a>
            </li>
          </ul>
        </div>
      </div>
      <div class="rh-route">
        <h2 class="rh-h2">Kiến thức nền LLM</h2>
        <p class="rh-note">Khái niệm docs Unsloth dùng mà không giải thích. Đủ để tự chọn tham số.</p>
        <div v-for="g in nen" :key="g.group" class="rh-group">
          <h3 class="rh-h3">{{ g.group }}</h3>
          <ul>
            <li v-for="i in g.items" :key="i.l">
              <a :href="withBase(i.l)"><span class="rh-t">{{ i.t }}</span><span class="rh-d">{{ i.d }}</span></a>
            </li>
          </ul>
        </div>
      </div>
    </section>

    <section class="rh-path" aria-labelledby="rh-path-h">
      <h2 id="rh-path-h" class="rh-h2">Lộ trình năm bước</h2>
      <ol class="rh-steps">
        <li v-for="s in steps" :key="s.t">
          <span class="rh-step-t">{{ s.t }}</span>
          <span class="rh-step-d">{{ s.d }}</span>
        </li>
      </ol>
      <p class="rh-note"><a :href="withBase('/lo-trinh-hoc')">Xem chi tiết từng bước</a></p>
    </section>
  </div>
</template>

<style scoped>
.rh {
  max-width: 1080px;
  margin: 0 auto;
  padding: 72px 32px 24px;
  color: var(--ink);
  font-family: var(--font-ui);
}

.rh a {
  color: inherit;
}

/* Đầu trang */
.rh-head {
  max-width: 720px;
  padding-bottom: 56px;
}

.rh-title {
  font-size: clamp(40px, 6vw, 64px);
  line-height: 1.05;
  font-weight: 700;
  letter-spacing: -0.035em;
  margin: 0 0 22px;
}

.rh-lead {
  font-family: var(--font-body);
  font-size: 21px;
  line-height: 1.6;
  color: var(--ink-2);
  margin: 0 0 30px;
  max-width: 34em;
}

.rh-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.rh-btn {
  display: inline-block;
  font-size: 15px;
  font-weight: 600;
  padding: 10px 18px;
  border: 1px solid var(--ink);
  border-radius: 4px;
  text-decoration: none;
}

.rh-btn-primary {
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  color: #fff !important;
}

.rh-btn:hover {
  border-color: var(--vp-c-brand-1);
}

.rh-btn-primary:hover {
  background: var(--vp-c-brand-2);
}

/* Tiêu đề mục */
.rh-h2 {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.01em;
  margin: 0 0 6px;
}

.rh-h3 {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink-3);
  margin: 26px 0 4px;
}

.rh-note {
  font-size: 15px;
  color: var(--ink-2);
  margin: 0 0 18px;
}

.rh-note a {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

/* Chú giải: điểm nhấn của trang */
.rh-legend {
  border-top: 2px solid var(--ink);
  padding-top: 28px;
  margin-bottom: 72px;
}

.rh-marks {
  margin: 0;
}

.rh-mark-row {
  --c: var(--mark-docs);
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 24px;
  padding: 18px 0;
  border-bottom: 1px solid var(--rule);
}

.rh-mark-row[data-kind='ext'] {
  --c: var(--mark-ext);
}
.rh-mark-row[data-kind='estimate'] {
  --c: var(--mark-estimate);
}
.rh-mark-row[data-kind='opinion'] {
  --c: var(--mark-opinion);
}
.rh-mark-row[data-kind='conflict'] {
  --c: var(--mark-conflict);
}

.rh-mark-row dt {
  margin: 0;
}

.rh-mark {
  display: inline-block;
  color: var(--c);
  border: 1px solid currentColor;
  border-radius: 3px;
  font-size: 14px;
  font-weight: 600;
  line-height: 1;
  padding: 6px 10px 7px;
}

.rh-mark-row[data-kind='docs'] .rh-mark {
  border-color: transparent;
  border-left: 2px solid var(--rule);
  border-radius: 0;
  padding-left: 10px;
}

.rh-mark-row[data-kind='conflict'] .rh-mark {
  border-left-width: 3px;
}

.rh-mark-row[data-kind='conflict'] .rh-mark::before {
  content: '≠ ';
}

.rh-mark-row dd {
  margin: 0;
}

.rh-meaning {
  font-size: 15px;
  line-height: 1.55;
  margin: 0 0 6px;
}

.rh-example {
  font-family: var(--font-body);
  font-size: 16px;
  line-height: 1.55;
  color: var(--ink-2);
  margin: 0;
}

.rh-example a {
  text-decoration: underline;
  text-decoration-color: color-mix(in srgb, var(--c) 45%, transparent);
  text-underline-offset: 3px;
}

.rh-example a:hover {
  color: var(--c);
  text-decoration-color: currentColor;
}

/* Mục lục hai route */
.rh-routes {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 56px;
  border-top: 2px solid var(--ink);
  padding-top: 28px;
  margin-bottom: 72px;
}

.rh-route ul {
  list-style: none;
  margin: 0;
  padding: 0;
}

.rh-route li a {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr);
  gap: 16px;
  padding: 9px 0;
  border-bottom: 1px solid var(--rule);
  text-decoration: none;
}

.rh-t {
  font-size: 15.5px;
  font-weight: 600;
}

.rh-d {
  font-size: 14px;
  color: var(--ink-3);
}

.rh-route li a:hover .rh-t {
  color: var(--vp-c-brand-1);
}

/* Lộ trình: chuỗi bước thật, nên được đánh số */
.rh-path {
  border-top: 2px solid var(--ink);
  padding-top: 28px;
}

.rh-steps {
  list-style: none;
  counter-reset: step;
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  margin: 12px 0 20px;
  padding: 0;
}

.rh-steps li {
  counter-increment: step;
  position: relative;
  padding: 14px 16px 0 0;
  border-top: 3px solid var(--rule);
}

.rh-steps li::before {
  content: counter(step);
  display: block;
  font-size: 28px;
  font-weight: 700;
  line-height: 1;
  color: var(--vp-c-brand-1);
  margin-bottom: 10px;
}

.rh-step-t {
  display: block;
  font-size: 15.5px;
  font-weight: 600;
  margin-bottom: 2px;
}

.rh-step-d {
  display: block;
  font-size: 14px;
  color: var(--ink-3);
}

@media (max-width: 860px) {
  .rh {
    padding: 44px 20px 16px;
  }
  .rh-mark-row {
    grid-template-columns: 1fr;
    gap: 10px;
  }
  .rh-routes {
    grid-template-columns: 1fr;
    gap: 40px;
  }
  .rh-steps {
    grid-template-columns: 1fr;
  }
  .rh-steps li {
    display: grid;
    grid-template-columns: 36px 1fr;
    border-top: none;
    border-left: 3px solid var(--rule);
    padding: 8px 0 12px 14px;
  }
  .rh-steps li::before {
    grid-row: span 2;
    font-size: 22px;
    margin: 0;
  }
}

@media (max-width: 480px) {
  .rh-route li a {
    grid-template-columns: 1fr;
    gap: 2px;
  }
  .rh-title {
    font-size: 36px;
  }
  .rh-lead {
    font-size: 18px;
  }
}
</style>
