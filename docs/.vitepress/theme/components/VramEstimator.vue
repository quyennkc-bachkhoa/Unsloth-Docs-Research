<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import { PRESETS, countParams, estimate, type Arch, type Mode, type Precision } from './vramFormula'

const presetId = ref(PRESETS[1].id)
const arch = reactive<Arch>({ ...PRESETS[1] })
const paramsB = ref(+(countParams(PRESETS[1]) / 1e9).toFixed(2))
const precision = ref<Precision>('bf16')
const mode = ref<Mode>('run')
const context = ref(2048)
const batch = ref(1)
const loraRank = ref(16)
const overheadPct = ref(10)

watch(presetId, (id) => {
  const p = PRESETS.find((x) => x.id === id)
  if (!p) return
  Object.assign(arch, p)
  paramsB.value = +(countParams(p) / 1e9).toFixed(2)
})

watch(mode, (m) => {
  // Theo docs Unsloth: QLoRA dùng 4-bit, LoRA dùng 16-bit; batch mặc định của notebook là 2.
  if (m === 'lora') precision.value = 'bf16'
  if (m === 'qlora') precision.value = '4bit'
  batch.value = m === 'run' ? 1 : 2
})

const preset = computed(() => PRESETS.find((x) => x.id === presetId.value))

const result = computed(() =>
  estimate({
    paramsB: Math.max(0, Number(paramsB.value) || 0),
    precision: precision.value,
    context: Math.max(1, Number(context.value) || 1),
    batch: Math.max(1, Number(batch.value) || 1),
    mode: mode.value,
    arch,
    loraRank: Math.max(1, Number(loraRank.value) || 1),
    overheadPct: Math.max(0, Number(overheadPct.value) || 0)
  })
)

const rows = computed(() => {
  const r = result.value
  const list = [{ key: 'weights', label: mode.value === 'run' ? 'Trọng số model' : 'Trọng số model gốc (đóng băng)', value: r.weights }]
  if (mode.value === 'run') list.push({ key: 'kv', label: 'KV cache', value: r.kvCache })
  else {
    list.push({ key: 'train', label: 'LoRA adapter + gradient + optimizer', value: r.trainable })
    list.push({ key: 'act', label: 'Activation (có gradient checkpointing)', value: r.activations })
  }
  list.push({ key: 'over', label: `Phần phụ (${overheadPct.value}%)`, value: r.overhead })
  return list
})

const fmt = (x: number) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2))
const pct = (x: number) => (result.value.total > 0 ? (x / result.value.total) * 100 : 0)
</script>

<template>
  <div class="vram">
    <div class="grid">
      <label>
        <span>Chế độ</span>
        <select v-model="mode">
          <option value="run">Chạy (inference)</option>
          <option value="lora">Fine-tune LoRA (16-bit)</option>
          <option value="qlora">Fine-tune QLoRA (4-bit)</option>
        </select>
      </label>
      <label>
        <span>Kiến trúc mẫu</span>
        <select v-model="presetId">
          <option v-for="p in PRESETS" :key="p.id" :value="p.id">{{ p.label }}</option>
        </select>
      </label>
      <label>
        <span>Số tham số (tỷ)</span>
        <input v-model.number="paramsB" type="number" min="0" step="0.1" />
      </label>
      <label>
        <span>Độ chính xác trọng số</span>
        <select v-model="precision" :disabled="mode !== 'run'">
          <option value="bf16">FP16 / BF16 (2 byte)</option>
          <option value="fp8">FP8 (1 byte)</option>
          <option value="int8">INT8 (1 byte)</option>
          <option value="4bit">4-bit (0,5 byte)</option>
        </select>
      </label>
      <label>
        <span>Độ dài context (token)</span>
        <input v-model.number="context" type="number" min="1" step="512" />
      </label>
      <label>
        <span>Batch size</span>
        <input v-model.number="batch" type="number" min="1" step="1" />
      </label>
      <label v-if="mode !== 'run'">
        <span>LoRA rank r</span>
        <input v-model.number="loraRank" type="number" min="1" step="8" />
      </label>
      <label>
        <span>Phần phụ (%)</span>
        <input v-model.number="overheadPct" type="number" min="0" step="5" />
      </label>
    </div>

    <details class="arch">
      <summary>Thông số kiến trúc (dùng cho KV cache, LoRA, activation)</summary>
      <div class="grid">
        <label><span>Số layer</span><input v-model.number="arch.layers" type="number" min="1" /></label>
        <label><span>Hidden size</span><input v-model.number="arch.hidden" type="number" min="1" /></label>
        <label><span>Intermediate size</span><input v-model.number="arch.intermediate" type="number" min="1" /></label>
        <label><span>Attention heads</span><input v-model.number="arch.heads" type="number" min="1" /></label>
        <label><span>KV heads</span><input v-model.number="arch.kvHeads" type="number" min="1" /></label>
        <label><span>Head dim</span><input v-model.number="arch.headDim" type="number" min="1" /></label>
      </div>
      <p v-if="preset" class="src">
        Giá trị mặc định lấy từ <a :href="preset.source" target="_blank" rel="noopener">config.json của {{ preset.label }}</a>.
      </p>
    </details>

    <div class="result">
      <div class="total">
        <span>Ước tính</span>
        <strong>≈ {{ fmt(result.total) }} GB</strong>
      </div>
      <div class="bar" role="img" aria-label="Tỷ lệ các thành phần bộ nhớ">
        <span v-for="r in rows" :key="r.key" :class="'seg-' + r.key" :style="{ width: pct(r.value) + '%' }" />
      </div>
      <table>
        <tbody>
          <tr v-for="r in rows" :key="r.key">
            <td><i :class="'dot seg-' + r.key" />{{ r.label }}</td>
            <td class="num">{{ fmt(r.value) }} GB</td>
          </tr>
          <tr v-if="mode !== 'run'">
            <td>Số tham số LoRA (7 target module)</td>
            <td class="num">{{ (result.loraParams / 1e6).toFixed(1) }} triệu</td>
          </tr>
        </tbody>
      </table>
      <p class="note">[Ước tính] Không phải số đo. Bộ nhớ thực tế phụ thuộc model, engine và cấu hình.</p>
    </div>
  </div>
</template>

<style scoped>
.vram {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 16px;
  margin: 16px 0;
  background: var(--vp-c-bg-soft);
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
}
label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
input,
select {
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 14px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  width: 100%;
}
select:disabled {
  opacity: 0.6;
}
.arch {
  margin-top: 12px;
  font-size: 14px;
}
.arch summary {
  cursor: pointer;
  color: var(--vp-c-brand-1);
  margin-bottom: 8px;
}
.src {
  font-size: 12px;
  margin: 8px 0 0;
}
.result {
  margin-top: 16px;
}
.total {
  display: flex;
  align-items: baseline;
  gap: 12px;
}
.total strong {
  font-size: 28px;
  color: var(--vp-c-brand-1);
}
.bar {
  display: flex;
  height: 12px;
  border-radius: 6px;
  overflow: hidden;
  margin: 10px 0;
  background: var(--vp-c-divider);
}
.result table {
  display: table;
  width: 100%;
  margin: 0;
}
.result td {
  border: none;
  padding: 4px 0;
  font-size: 14px;
}
.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  margin-right: 8px;
}
.seg-weights { background: #e30613; }
.seg-kv { background: #f59e0b; }
.seg-train { background: #8b5cf6; }
.seg-act { background: #0ea5e9; }
.seg-over { background: #9ca3af; }
.note {
  font-size: 12px;
  color: var(--vp-c-text-2);
  margin: 8px 0 0;
}
</style>
