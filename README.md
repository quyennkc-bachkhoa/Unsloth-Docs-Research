# Unsloth-Docs-Research

Website research về [Unsloth](https://github.com/unslothai/unsloth), framework mã nguồn mở để chạy và fine-tune LLM trên máy local. Website viết bằng tiếng Việt, dựng bằng [VitePress](https://vitepress.dev).

Website có 2 route:

- **Unsloth** (`/`): các sản phẩm Desktop, Studio, Core; cài đặt theo nền tảng; inference và API; model catalog; fine-tuning, RL, dữ liệu; export và deploy; ứng dụng RAG; thuật ngữ và nguồn. Nội dung chỉ lấy từ [docs chính thức](https://unsloth.ai/docs) và repo GitHub của Unsloth.
- **Kiến thức nền LLM** (`/kien-thuc-nen/`): giải thích những khái niệm docs Unsloth dùng mà không giải thích (token, Transformer, MoE, bộ nhớ, lượng tử hóa, LoRA, RL, sampling). Có công cụ ước tính VRAM. Nguồn ngoài docs Unsloth giới hạn ở paper arXiv và tài liệu chính thức của Hugging Face, PyTorch, llama.cpp, NVIDIA và model card chính thức.

Quy ước nhãn trong nội dung:

- **[Nhận định]:** ý kiến của người viết, không có trong nguồn.
- **[Nguồn ngoài]:** nội dung lấy từ nguồn ngoài docs Unsloth.
- **[Ước tính]:** con số tự tính từ một công thức có nguồn.
- Hộp **"Docs chưa thống nhất"**: các trang docs Unsloth ghi giá trị khác nhau; website liệt kê đủ các giá trị, mỗi giá trị kèm link, và không tự chọn giá trị nào.

## Chạy local

Yêu cầu: Node.js LTS (đã thử với Node 24).

```bash
npm install
npm run docs:dev
```

Mở địa chỉ mà terminal in ra, mặc định là `http://localhost:5173/Unsloth-Docs-Research/`.

## Build

```bash
npm run docs:build     # xuất ra docs/.vitepress/dist
npm run docs:preview   # xem bản build tại http://localhost:4173/Unsloth-Docs-Research/
```

## Cấu trúc

```
docs/
├── .vitepress/
│   ├── config.mts                 # title, nav, multi-sidebar, search, base
│   └── theme/
│       ├── index.ts               # đăng ký component global
│       ├── custom.css             # màu chủ đạo #E30613
│       └── components/
│           ├── VramEstimator.vue  # công cụ ước tính bộ nhớ
│           └── vramFormula.ts     # công thức ước tính
├── *.md                           # route Unsloth
└── kien-thuc-nen/*.md             # route Kiến thức nền LLM
.github/workflows/deploy.yml       # build + deploy GitHub Pages
```

## Deploy lên GitHub Pages

Workflow `.github/workflows/deploy.yml` tự build và deploy mỗi lần push lên `main`. Bạn cũng có thể chạy tay ở tab Actions (`workflow_dispatch`). Sau khi push, cần làm các bước sau trên GitHub:

1. Kiểm tra repo đang để **Public**.
2. Vào **Settings → Pages → Build and deployment → Source**, chọn **GitHub Actions**.
3. Vào tab **Actions**, kiểm tra workflow "Deploy VitePress site to Pages" chạy xanh.
4. Mở website tại `https://<username>.github.io/Unsloth-Docs-Research/`.

`base` trong `docs/.vitepress/config.mts` đang đặt là `/Unsloth-Docs-Research/`, phải khớp chính xác tên repo, kể cả hoa thường. Nếu đổi tên repo thì sửa `base` theo.
