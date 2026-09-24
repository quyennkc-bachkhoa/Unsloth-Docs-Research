import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { researchMarks } from './researchMarks'

const unslothSidebar = [
  {
    text: 'Nhập môn',
    items: [
      { text: 'Trang chủ', link: '/' },
      { text: 'Tổng quan kiến trúc', link: '/tong-quan' },
      { text: 'Lộ trình học', link: '/lo-trinh-hoc' }
    ]
  },
  {
    text: 'Sử dụng',
    items: [
      { text: 'Cài đặt & phần cứng', link: '/cai-dat' },
      {
        text: 'Inference & API',
        link: '/inference/',
        collapsed: false,
        items: [
          { text: 'Studio Chat', link: '/inference/studio-chat' },
          { text: 'API OpenAI & Anthropic', link: '/inference/api' },
          { text: 'Coding agent', link: '/inference/coding-agent' },
          { text: 'Connections', link: '/inference/connections' },
          { text: 'MCP', link: '/inference/mcp' },
          { text: 'Tool calling', link: '/inference/tool-calling' },
          { text: 'Cạm bẫy', link: '/inference/cam-bay' }
        ]
      },
      { text: 'Model catalog', link: '/model-catalog' }
    ]
  },
  {
    text: 'Huấn luyện',
    items: [
      {
        text: 'Fine-tuning',
        link: '/fine-tuning/',
        collapsed: false,
        items: [
          { text: 'LoRA, QLoRA hay full', link: '/fine-tuning/lora-qlora-full' },
          { text: 'Chọn model', link: '/fine-tuning/chon-model' },
          { text: 'Quy trình', link: '/fine-tuning/quy-trinh' },
          { text: 'Hyperparameter', link: '/fine-tuning/hyperparameter' },
          { text: 'Đánh giá & overfitting', link: '/fine-tuning/danh-gia' },
          { text: 'Notebooks', link: '/fine-tuning/notebooks' },
          { text: 'Hiệu năng & benchmark', link: '/fine-tuning/benchmark' },
          { text: 'Mở rộng', link: '/fine-tuning/mo-rong' },
          { text: 'Cạm bẫy', link: '/fine-tuning/cam-bay' }
        ]
      },
      { text: 'Reinforcement Learning', link: '/reinforcement-learning' },
      { text: 'Dữ liệu', link: '/du-lieu' }
    ]
  },
  {
    text: 'Triển khai & ứng dụng',
    items: [
      { text: 'Export & deploy', link: '/export-deploy' },
      { text: 'Ứng dụng RAG', link: '/ung-dung-rag' }
    ]
  },
  {
    text: 'Tham khảo',
    items: [
      { text: 'Thuật ngữ', link: '/thuat-ngu' },
      { text: 'Nguồn tham khảo', link: '/nguon' }
    ]
  }
]

const kienThucNenSidebar = [
  {
    text: 'Tổng quan',
    items: [{ text: 'Đọc route này khi nào', link: '/kien-thuc-nen/' }]
  },
  {
    text: 'A. Cấu tạo mô hình',
    items: [
      { text: 'Token & context', link: '/kien-thuc-nen/token-va-context' },
      { text: 'Phân loại mô hình', link: '/kien-thuc-nen/phan-loai-mo-hinh' },
      { text: 'Kiến trúc Transformer', link: '/kien-thuc-nen/kien-truc-transformer' },
      { text: 'Dense & MoE', link: '/kien-thuc-nen/dense-va-moe' }
    ]
  },
  {
    text: 'B. Số học & bộ nhớ',
    items: [
      { text: 'Tham số & bộ nhớ', link: '/kien-thuc-nen/tham-so-va-bo-nho' },
      { text: 'Độ chính xác & lượng tử hóa', link: '/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa' }
    ]
  },
  {
    text: 'C. Huấn luyện',
    items: [
      { text: 'Quá trình huấn luyện', link: '/kien-thuc-nen/qua-trinh-huan-luyen' },
      { text: 'LoRA & QLoRA', link: '/kien-thuc-nen/lora-va-qlora' },
      { text: 'RL & preference', link: '/kien-thuc-nen/rl-va-preference' }
    ]
  },
  {
    text: 'D. Suy luận',
    items: [{ text: 'Suy luận & sampling', link: '/kien-thuc-nen/suy-luan-va-sampling' }]
  }
]

export default withMermaid(
  defineConfig({
    title: 'Unsloth Research',
    description: 'Nghiên cứu Unsloth — chạy và fine-tune LLM local, viết bằng tiếng Việt',
    lang: 'vi-VN',
    base: '/Unsloth-Docs-Research/',
    cleanUrls: true,
    lastUpdated: true,
    appearance: true,
    vite: { build: { chunkSizeWarningLimit: 4000 } },
    head: [
      ['meta', { name: 'theme-color', content: '#E30613' }],
      ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
      ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
      [
        'link',
        {
          rel: 'stylesheet',
          href: 'https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap'
        }
      ]
    ],
    markdown: {
      config: (md) => {
        md.use(researchMarks)
      },
      container: {
        tipLabel: 'Mẹo',
        warningLabel: 'Cảnh báo',
        dangerLabel: 'Nguy hiểm',
        infoLabel: 'Thông tin',
        detailsLabel: 'Chi tiết'
      }
    },
    themeConfig: {
      nav: [
        { text: 'Unsloth', link: '/', activeMatch: '^/(?!kien-thuc-nen/)' },
        { text: 'Kiến thức nền LLM', link: '/kien-thuc-nen/', activeMatch: '^/kien-thuc-nen/' }
      ],
      sidebar: {
        '/kien-thuc-nen/': kienThucNenSidebar,
        '/': unslothSidebar
      },
      search: {
        provider: 'local',
        options: {
          translations: {
            button: { buttonText: 'Tìm kiếm', buttonAriaLabel: 'Tìm kiếm' },
            modal: {
              noResultsText: 'Không có kết quả cho',
              resetButtonTitle: 'Xóa',
              footer: { selectText: 'chọn', navigateText: 'di chuyển', closeText: 'đóng' }
            }
          }
        }
      },
      socialLinks: [{ icon: 'github', link: 'https://github.com/unslothai/unsloth' }],
      outline: { level: [2, 3], label: 'Trên trang này' },
      docFooter: { prev: 'Trang trước', next: 'Trang sau' },
      lastUpdated: { text: 'Cập nhật lần cuối' },
      darkModeSwitchLabel: 'Giao diện',
      lightModeSwitchTitle: 'Chuyển sang sáng',
      darkModeSwitchTitle: 'Chuyển sang tối',
      sidebarMenuLabel: 'Menu',
      returnToTopLabel: 'Lên đầu trang',
      footer: {
        message: 'Nội dung tổng hợp từ docs chính thức của Unsloth; phần [Nhận định] là ý kiến người viết.',
        copyright: 'Unsloth Research'
      }
    }
  })
)
