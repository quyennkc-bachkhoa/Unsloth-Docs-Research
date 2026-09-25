import { defineConfig } from 'vitepress'
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
      { text: 'Cài đặt và phần cứng', link: '/cai-dat' },
      {
        text: 'Chạy model và gọi API',
        link: '/inference/',
        collapsed: false,
        items: [
          { text: 'Chat trong Studio', link: '/inference/studio-chat' },
          { text: 'Gọi qua API (OpenAI, Anthropic)', link: '/inference/api' },
          { text: 'Dùng với coding agent', link: '/inference/coding-agent' },
          { text: 'Kết nối server khác', link: '/inference/connections' },
          { text: 'Thêm công cụ qua MCP', link: '/inference/mcp' },
          { text: 'Cho model gọi hàm (tool calling)', link: '/inference/tool-calling' }
        ]
      },
      { text: 'Danh sách model hỗ trợ', link: '/model-catalog' }
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
          { text: 'Chọn cách train và model', link: '/fine-tuning/chon-cach-train' },
          { text: 'Quy trình từng bước', link: '/fine-tuning/quy-trinh' },
          { text: 'Chọn hyperparameter', link: '/fine-tuning/hyperparameter' },
          { text: 'Đánh giá và overfitting', link: '/fine-tuning/danh-gia' },
          { text: 'Notebook chạy sẵn', link: '/fine-tuning/notebooks' },
          { text: 'Hiệu năng và benchmark', link: '/fine-tuning/benchmark' },
          { text: 'Kỹ thuật nâng cao', link: '/fine-tuning/mo-rong' },
          { text: 'Lỗi thường gặp', link: '/fine-tuning/loi-thuong-gap' }
        ]
      },
      {
        text: 'Reinforcement Learning',
        link: '/reinforcement-learning/',
        collapsed: false,
        items: [
          { text: 'Train bằng GRPO', link: '/reinforcement-learning/grpo' },
          { text: 'Viết reward function', link: '/reinforcement-learning/reward-function' },
          { text: 'Train theo cặp tốt/xấu (DPO, ORPO, KTO)', link: '/reinforcement-learning/dpo-orpo-kto' },
          { text: 'Tiết kiệm VRAM khi chạy RL', link: '/reinforcement-learning/memory-efficient' },
          { text: 'Train AI agent', link: '/reinforcement-learning/agent' },
          { text: 'Lỗi thường gặp', link: '/reinforcement-learning/loi-thuong-gap' }
        ]
      },
      {
        text: 'Dữ liệu',
        link: '/du-lieu/',
        collapsed: false,
        items: [
          { text: 'Định dạng dữ liệu', link: '/du-lieu/dinh-dang' },
          { text: 'Chat template', link: '/du-lieu/chat-template' },
          { text: 'Làm dữ liệu trong Studio', link: '/du-lieu/studio' },
          { text: 'Sinh dữ liệu tổng hợp', link: '/du-lieu/synthetic' },
          { text: 'Checklist chuẩn bị dữ liệu', link: '/du-lieu/checklist' }
        ]
      }
    ]
  },
  {
    text: 'Triển khai và ứng dụng',
    items: [
      {
        text: 'Export và deploy',
        link: '/export-deploy/',
        collapsed: false,
        items: [
          { text: 'Xuất file GGUF', link: '/export-deploy/gguf' },
          { text: 'Định dạng trọng số NVFP4 và FP8', link: '/export-deploy/nvfp4-fp8' },
          { text: 'Chạy model đã xuất', link: '/export-deploy/chay-model' },
          { text: 'Mở cho máy khác (LAN, Cloudflare)', link: '/export-deploy/lan-remote' },
          { text: 'Lỗi thường gặp', link: '/export-deploy/loi-thuong-gap' }
        ]
      },
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
    items: [{ text: 'Giới thiệu', link: '/kien-thuc-nen/' }]
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

const baiToanSidebar = [
  {
    text: 'Bài toán 2×L40S',
    items: [
      { text: 'Đề bài và đáp án nhanh', link: '/bai-toan/' },
      { text: 'Chọn model cho tiếng Việt', link: '/bai-toan/chon-model' },
      { text: 'Fine-tune bằng Unsloth', link: '/bai-toan/fine-tune' },
      { text: 'Dữ liệu văn bản hành chính', link: '/bai-toan/van-ban-hanh-chinh' },
      { text: 'Host model trên máy chủ', link: '/bai-toan/hosting' }
    ]
  }
]

export default defineConfig({
  title: 'Unsloth Research',
  description: 'Nghiên cứu Unsloth — chạy và fine-tune LLM local, viết bằng tiếng Việt',
  lang: 'vi-VN',
  base: '/Unsloth-Docs-Research/',
  cleanUrls: true,
  lastUpdated: true,
  appearance: true,
  vite: {
    build: { chunkSizeWarningLimit: 4000 }
  },
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
      { text: 'Unsloth', link: '/', activeMatch: '^/(?!kien-thuc-nen/|bai-toan/)' },
      { text: 'Kiến thức nền LLM', link: '/kien-thuc-nen/', activeMatch: '^/kien-thuc-nen/' },
      { text: 'Bài toán 2×L40S', link: '/bai-toan/', activeMatch: '^/bai-toan/' }
    ],
    sidebar: {
      '/kien-thuc-nen/': kienThucNenSidebar,
      '/bai-toan/': baiToanSidebar,
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
