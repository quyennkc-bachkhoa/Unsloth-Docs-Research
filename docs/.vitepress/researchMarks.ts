// Plugin markdown-it: biến các nhãn nguồn gốc thông tin trong nội dung thành phần tử có class,
// để CSS hiển thị chúng như một hệ thống ký hiệu thống nhất. Nội dung markdown không đổi.
import type { MarkdownRenderer } from 'vitepress'

const MARKS: Record<string, string> = {
  'Nhận định': 'nhan-dinh',
  'Nguồn ngoài': 'nguon-ngoai',
  'Ước tính': 'uoc-tinh'
}
const MARK_RE = /\[(Nhận định|Nguồn ngoài|Ước tính)\]/g

function shortUrl(href: string): string {
  return href.replace(/^https?:\/\//, '').replace(/^www\./, '').replace(/\/$/, '')
}

export function researchMarks(md: MarkdownRenderer) {
  md.core.ruler.push('research-marks', (state) => {
    const tokens = state.tokens
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i]

      // Hộp container đặc biệt: gắn data-kind để CSS phân biệt với hộp tip/warning thường.
      if (t.type === 'container_warning_open' && /^warning\s+Docs chưa thống nhất/.test(t.info.normalize('NFC').trim())) {
        t.attrSet('data-kind', 'conflict')
      }
      if (t.type === 'container_tip_open' && /^tip\s+Kiến thức nền/.test(t.info.normalize('NFC').trim())) {
        t.attrSet('data-kind', 'kb')
      }
      if (t.type === 'container_tip_open' && /^tip\s+Tóm tắt/.test(t.info.normalize('NFC').trim())) {
        t.attrSet('data-kind', 'summary')
      }

      if (t.type !== 'inline' || !t.children) continue
      const kids = t.children

      // Đoạn bắt đầu bằng **Nguồn:** là dòng trích dẫn (bỏ qua token text rỗng ở đầu).
      const lead = kids.filter((c) => !(c.type === 'text' && c.content === ''))
      if (
        tokens[i - 1]?.type === 'paragraph_open' &&
        lead[0]?.type === 'strong_open' &&
        /^Nguồn:?$/.test((lead[1]?.content ?? '').normalize('NFC').trim())
      ) {
        tokens[i - 1].attrJoin('class', 'src-line')
      }

      const out: typeof kids = []
      for (let k = 0; k < kids.length; k++) {
        const c = kids[k]
        // URL trần (linkify): hiển thị gọn, bỏ giao thức. href giữ nguyên.
        if (c.type === 'link_open' && c.markup === 'linkify') {
          const txt = kids[k + 1]
          if (txt?.type === 'text') txt.content = shortUrl(txt.content)
        }
        if (c.type === 'text') c.content = c.content.normalize('NFC')
        if (c.type === 'text' && MARK_RE.test(c.content)) {
          MARK_RE.lastIndex = 0
          let last = 0
          let m: RegExpExecArray | null
          while ((m = MARK_RE.exec(c.content))) {
            if (m.index > last) {
              const tt = new state.Token('text', '', 0)
              tt.content = c.content.slice(last, m.index)
              out.push(tt)
            }
            const h = new state.Token('html_inline', '', 0)
            h.content = `<span class="mark mark-${MARKS[m[1]]}">${m[1]}</span>`
            out.push(h)
            last = m.index + m[0].length
          }
          if (last < c.content.length) {
            const tt = new state.Token('text', '', 0)
            tt.content = c.content.slice(last)
            out.push(tt)
          }
          continue
        }
        out.push(c)
      }
      t.children = out
    }
  })
}
