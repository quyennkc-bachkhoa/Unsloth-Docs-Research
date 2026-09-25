---
title: Cài đặt & phần cứng
description: Yêu cầu phần cứng/phần mềm và lệnh cài Unsloth Desktop, Studio, Core trên Windows, macOS, Linux/WSL, AMD, Intel, CPU, Docker, pip/uv và Google Colab.
---

# Cài đặt & phần cứng

## Ba cách dùng Unsloth, ba bộ yêu cầu

Trước khi cài, bạn cần chọn dùng Unsloth theo cách nào. Unsloth có ba "sản phẩm" dùng chung một lõi. Mỗi sản phẩm có cách cài và yêu cầu hệ thống riêng. Phần giới thiệu chung nằm ở trang [Tổng quan](/tong-quan).

| Sản phẩm | Là gì | Cách cài | Ghi chú yêu cầu |
|---|---|---|---|
| **Unsloth Desktop** | App native (ứng dụng cài như phần mềm thường) cho macOS, Windows, Linux | Tải file cài `.dmg` / `.exe` / `.deb` | Trang nguồn không liệt kê yêu cầu riêng; tải model rồi chat ngay, "no setup required" |
| **Unsloth Studio** | Web UI (giao diện chạy trong trình duyệt), mặc định ở cổng 8888 | Script `install.sh` / `install.ps1` | Có bảng yêu cầu theo OS: Python 3.11 đến dưới 3.14, NVIDIA driver để train trên Windows/Linux… |
| **Unsloth Core** | Thư viện Python gốc, dùng bằng code (notebook, script) | `pip` / `uv` / Conda / Docker | Linux và Windows; GPU NVIDIA từ 2018+, CUDA Capability tối thiểu 7.0; AMD/Intel theo hướng dẫn riêng |

Cách chọn:
- Muốn bắt đầu nhanh: dùng Desktop. Docs gọi đây là cách dễ nhất để có Studio.
- Muốn chỉnh tùy chọn nâng cao: cài Studio thủ công.
- Muốn tự viết code để train: dùng Core.

<div class="dg">
<div class="dg-tree">
<div class="dg-node is-q">Bạn muốn gì?</div>
<div class="dg-kids">
<div class="dg-kid"><div class="dg-col">
<div class="dg-node">Chat/train bằng giao diện, không cấu hình</div>
<div class="dg-node is-end">Unsloth Desktop</div>
</div></div>
<div class="dg-kid"><div class="dg-col">
<div class="dg-node">Web UI, truy cập từ máy khác, tùy chọn nâng cao</div>
<div class="dg-node is-end">Unsloth Studio<small><code>install.sh</code> / <code>install.ps1</code> / Docker</small></div>
</div></div>
<div class="dg-kid"><div class="dg-col">
<div class="dg-node">Viết code Python train/inference</div>
<div class="dg-node is-end">Unsloth Core<small>uv / pip / Conda / Docker / Colab</small></div>
</div></div>
</div>
</div>
</div>

**Nguồn:** https://unsloth.ai/docs/get-started/install, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements, https://unsloth.ai/docs/new/studio/install

## Bảng tóm tắt theo nền tảng

Bảng này giúp bạn tra nhanh: máy của bạn chạy được gì, train được không, và nên cài theo cách nào. Chi tiết từng nền tảng nằm ở các mục bên dưới.

Ô ghi "—" nghĩa là trang nguồn không nói rõ. Ô ghi "Docs chưa thống nhất" nghĩa là các trang docs nói khác nhau; các hộp cảnh báo ngay sau bảng liệt kê từng chỗ.

| Nền tảng | OS / phiên bản | GPU / driver | Python | Chạy (chat/inference)? | Train? | Sản phẩm hỗ trợ | Cách cài khuyến nghị |
|---|---|---|---|---|---|---|---|
| **Windows + NVIDIA** | Windows 10 hoặc 11 (64-bit) | GPU NVIDIA đã cài driver; CUDA Toolkit được script cài Studio tự cài khớp driver | 3.11 đến dưới 3.14 (Studio); ví dụ Conda của Core dùng 3.12 | Có | Có | Desktop, Studio, Core (Core qua Conda, Docker hoặc WSL) | Desktop (`.exe`), hoặc Studio: `irm https://unsloth.ai/install.ps1 \| iex` |
| **macOS** | macOS 12 Monterey trở lên (Intel hoặc Apple Silicon) | — (dùng MLX) | 3.11 đến dưới 3.14 | Có (MLX và GGUF) | Docs chưa thống nhất (xem bên dưới) | Desktop, Studio; Core: Docs chưa thống nhất (xem bên dưới) | Desktop (`.dmg`), hoặc Studio: `install.sh` |
| **Linux / WSL** | Ubuntu 20.04+ hoặc distro tương tự (64-bit) | GPU NVIDIA đã cài driver; CUDA toolkit: Docs chưa thống nhất (xem mục Linux & WSL); Core: CUDA Capability ≥ 7.0 | 3.11 đến dưới 3.14 (Studio); ví dụ uv của Core dùng 3.13 | Có | Có | Desktop (Linux: `.deb`/AppImage), Studio, Core | Desktop, hoặc Studio: `install.sh`; Core: `uv pip install unsloth --torch-backend=auto` |
| **AMD** | Windows và Linux | Dòng GPU và ROCm: Docs chưa thống nhất (xem bên dưới); Docker cần driver `amdgpu` | Docs chưa thống nhất (xem bên dưới) | Có (Studio trong Docker `unsloth-rocm` có chat) | Có | Desktop (Windows, Linux), Studio (train: Docs chưa thống nhất, xem bên dưới), Core, Docker `unsloth/unsloth-rocm` | Desktop; thủ công: `install.sh` / `install.ps1`; Core: `uv pip install unsloth[amd]` |
| **Intel** | Linux (khuyến nghị Ubuntu 22.04+) hoặc Windows 11 (khuyến nghị) | Data Center GPU Max, Arc, Intel Ultra AIPC; driver Intel Graphics mới nhất; Windows cần oneAPI Base Toolkit 2025.2.1 | Docs chưa thống nhất (xem bên dưới) | — | Có (Core) | Core (build từ source); Studio: Docs chưa thống nhất (xem bên dưới) | Core: `pip install .[intel-gpu-torch290]` |
| **CPU-only** | Như Linux (bỏ NVIDIA driver) và macOS | Không cần GPU | Như Linux/macOS | Có — Chat với model GGUF | Không được nêu (chỉ Chat + Data Recipes) | Studio | Studio: `install.sh` |

::: tip Kiến thức nền
GGUF, MLX, quantization là gì? Xem [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).
:::

::: warning Docs chưa thống nhất — Mac: train được không, Core có chạy trên Mac không
| Thông số | [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | [new/studio/install](https://unsloth.ai/docs/new/studio/install) | [install/mac](https://unsloth.ai/docs/get-started/install/mac) |
|---|---|---|---|
| Train trên Mac | Đầu trang: "Mac: Training, MLX and GGUF inference are ALL supported"; mục Training: Studio Training chạy trên "MLX" | Đầu trang: "Mac: Like CPU - Chat + Data Recipes works for now. **MLX** training now works!"; mục System Requirements: "Chat and training and all features" | Desktop dùng để "run and train MLX or GGUF models on your Mac" |
| Unsloth Core trên Mac | Mục Core Requirements: OS "Works on Linux and Windows"; "Apple/Silicon/MLX is in the works" | — | Trang chỉ cài Desktop và Studio |
| Docker trên Mac | — | "We're working on Mac compatibility" | — |
| Gói phải cài thêm cho Studio | Homebrew, `brew install git`, `brew install cmake`, `brew install openssl`, Python 3.11 đến dưới 3.14 | Mục MacOS chỉ liệt kê macOS 12+ và môi trường Python (uv/venv/conda) | — |
:::

::: warning Docs chưa thống nhất — Studio có train trên AMD/Intel không
Có trang nói Studio train được trên AMD và Intel. Có trang chỉ nhắc GPU NVIDIA, hoặc chỉ hướng dẫn Core. Bảng sau giữ nguyên lời từng trang:

| Nguồn | Nội dung |
|---|---|
| [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) (mục Training) | "Unsloth Studio Training currently works on NVIDIA, AMD, MLX, Intel devices" |
| [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) (cùng mục) | "You can still use the original Unsloth Core to train on AMD and Intel devices" |
| [new/studio/install](https://unsloth.ai/docs/new/studio/install) (đầu trang) | "Training: Works on NVIDIA, Intel, AMD GPUs and Mac devices" |
| [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) và [new/studio/install](https://unsloth.ai/docs/new/studio/install) (yêu cầu Windows, Linux & WSL) | Chỉ ghi "NVIDIA GPU with drivers installed", không có dòng AMD/Intel |
| [install/amd](https://unsloth.ai/docs/get-started/install/amd) | Cài Studio cho AMD bằng `install.sh` (Linux) / `install.ps1` (Windows); Desktop là cách cài dễ nhất trên AMD |
| [install/intel](https://unsloth.ai/docs/get-started/install/intel) | Chỉ hướng dẫn Core (build từ source), không nhắc Studio |
:::

::: warning Docs chưa thống nhất — dòng GPU AMD được hỗ trợ
| Thông số | [install/amd](https://unsloth.ai/docs/get-started/install/amd) | [install/docker](https://unsloth.ai/docs/get-started/install/docker) (image `unsloth/unsloth-rocm`) |
|---|---|---|
| GPU Radeon | "RDNA 3/3.5/4 (RX 6000–9000 series)" trên Windows và Linux | "covers RDNA1 and newer, plus CDNA" |
| GPU data center | MI300X (192GB) | CDNA |
| ROCm | "ROCm 6.0 or newer is required" (cài PyTorch thủ công) | Image "Built against ROCm 7.2" |
:::

::: warning Docs chưa thống nhất — phiên bản Python
Mỗi trang docs ghi một mức Python khác nhau, tùy sản phẩm và cách cài. Bảng sau liệt kê đủ để bạn đối chiếu với trường hợp của mình:

| Nguồn | Giá trị | Ngữ cảnh |
|---|---|---|
| [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | "version 3.11 up to, but not including, 3.14" | Yêu cầu Windows, MacOS, Linux & WSL |
| [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | "Python 3.11–3.13 is required" | Mục Training (Studio) |
| [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | "Python 3.13 is supported!" (không nêu mức tối thiểu) | Mục Unsloth Core |
| [new/studio/install](https://unsloth.ai/docs/new/studio/install) | "version 3.11 up to, but not including, 3.14"; bảng Troubleshooting gợi ý `sudo apt install python3.12 python3.12-venv` | Studio |
| [install/windows-installation](https://unsloth.ai/docs/get-started/install/windows-installation) | `conda create --name unsloth_env python==3.12 -y` | Core qua Conda trên Windows |
| [install/linux](https://unsloth.ai/docs/get-started/install/linux), [install/pip-install](https://unsloth.ai/docs/get-started/install/pip-install) | `uv venv unsloth_env --python 3.13` | Core qua uv |
| [install/pip-install](https://unsloth.ai/docs/get-started/install/pip-install) | `apt install python3.10-venv python3.11-venv python3.12-venv python3.13-venv -y` | Core qua venv |
| [install/pip-install](https://unsloth.ai/docs/get-started/install/pip-install) | `winget install -e --id Python.Python.3.13 --source winget` | Studio developer/nightly trên Windows |
| [install/amd](https://unsloth.ai/docs/get-started/install/amd) | "any 3.11-3.13 works everywhere (3.10 works for manual installs"; Windows: "use 3.12 if installing the unsloth[rocm72-torch291] extra" | AMD |
| [install/intel](https://unsloth.ai/docs/get-started/install/intel) | "Python: 3.10+"; `conda create -n unsloth-xpu python==3.10` | Intel (Core) |
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements, https://unsloth.ai/docs/new/studio/install, https://unsloth.ai/docs/get-started/install/amd, https://unsloth.ai/docs/get-started/install/intel, https://unsloth.ai/docs/get-started/install/docker, https://unsloth.ai/docs

## Yêu cầu VRAM cho fine-tuning

Bảng này cho biết card đồ họa của bạn cần ít nhất bao nhiêu VRAM (bộ nhớ card đồ họa) để fine-tune một model. Mức VRAM phụ thuộc vào hai thứ: model lớn cỡ nào (số tham số) và bạn dùng phương pháp nào.

Hai phương pháp trong bảng:
- QLoRA (LoRA trên model nén 4-bit): dùng 4-bit, tốn ít VRAM hơn.
- LoRA: dùng 16-bit.

Docs nhấn mạnh đây là **mức tối thiểu tuyệt đối**. Tùy model, bạn có thể cần nhiều hơn. Nguồn cũng không ghi rõ độ dài context hay batch size đã dùng khi đo.

| Số tham số model | QLoRA (4-bit) VRAM | LoRA (16-bit) VRAM |
|---|---|---|
| 3B | 3.5 GB | 8 GB |
| 7B | 5 GB | 19 GB |
| 8B | 6 GB | 22 GB |
| 9B | 6.5 GB | 24 GB |
| 11B | 7.5 GB | 29 GB |
| 14B | 8.5 GB | 33 GB |
| 27B | 22 GB | 64 GB |
| 32B | 26 GB | 76 GB |
| 40B | 30 GB | 96 GB |
| 70B | 41 GB | 164 GB |
| 81B | 48 GB | 192 GB |
| 90B | 53 GB | 212 GB |
| 405B | 237 GB | 950 GB |

Cách đọc bảng: model 8B với QLoRA cần tối thiểu 6 GB. Cùng model đó với LoRA 16-bit cần 22 GB, tức gấp khoảng 3,7 lần.

::: tip Mẹo tránh OOM
OOM (out of memory, hết bộ nhớ) thường xảy ra khi batch size quá cao. Để giảm VRAM, docs khuyên đặt batch size là 1, 2 hoặc 3.
:::

::: tip Kiến thức nền
Muốn hiểu "8B" nghĩa là gì và cách ước tính VRAM? Xem [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho). LoRA/QLoRA: xem [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora). Batch size: xem [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen).
:::

**[Nhận định]** Bảng có một bước nhảy lạ. Với QLoRA, 14B cần 8.5 GB nhưng 27B lên tới 22 GB, tức gần gấp 2,6. Trong khi đó số tham số chỉ gấp 1,9. Vì vậy bạn nên coi số liệu của các model lớn là để tham khảo.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements

## Windows

Trên Windows, bạn có ba lựa chọn: Desktop (dễ nhất), Studio cài bằng một lệnh PowerShell, hoặc Core cho người viết code. Core lại có ba cách cài: Conda, Docker hoặc WSL. Mục này đi lần lượt từng cách, rồi đến các lỗi hay gặp và cách gỡ cài đặt.

### Yêu cầu (Studio, để train)

- Hệ điều hành: Windows 10 hoặc 11 (64-bit).
- GPU: GPU NVIDIA đã cài driver.
- Công cụ: App Installer (có sẵn `winget`) và Git. Cài Git bằng `winget install --id Git.Git -e --source winget`.
- Python: 3.11 đến dưới 3.14. Bạn nên làm việc trong môi trường ảo (virtual environment) như uv, venv, conda hoặc mamba.

Studio chạy thẳng trên Windows, **không cần WSL** (Windows Subsystem for Linux, lớp chạy Linux trong Windows). Script cài tự lo phần lớn công cụ: Git, CMake (qua `winget`) và CUDA Toolkit khớp với driver. C++ compiler là Visual Studio Build Tools 2022.

::: warning Docs chưa thống nhất — Git/CUDA trên Windows
Cùng trang [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) nhưng hai chỗ nói khác nhau. Một chỗ bảo bạn tự cài Git, chỗ kia nói script cài lo việc này:

| Thông số | Danh sách "Requirements" (Windows) | Bảng mục Training (Windows) |
|---|---|---|
| Git | Tự cài: `winget install --id Git.Git -e --source winget` | "Installed by setup script (`winget`)" |
| CUDA Toolkit | Không nhắc | "Installed by setup script (matched to driver)" |
:::

### Desktop

1. Tải bản Windows: https://unsloth.ai/download/windows
2. Mở file `.exe` và làm theo hướng dẫn. Sau đó mở app và chờ cài xong.
3. Vào **Select model** hoặc **Model hub**, chọn model và mức quantization. Tải về rồi chat.

### Studio (cài thủ công)

Chạy lệnh sau trong PowerShell để cài:

```bash
irm https://unsloth.ai/install.ps1 | iex
```

Khi muốn cập nhật, bạn chạy lại đúng lệnh này. Mỗi lần muốn mở Studio, chạy:

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

Sau khi cài, mở `http://127.0.0.1:8888` trong trình duyệt. Lần đầu vào, Studio yêu cầu bạn tạo mật khẩu.

::: warning Docs chưa thống nhất — lệnh chạy Studio, địa chỉ bind và cổng
Lệnh chạy ở trên mở Studio cho mọi máy trong mạng. Trang Studio chính lại mặc định chỉ cho máy của bạn truy cập, và khuyên cách an toàn hơn khi mở ra ngoài. Cổng cũng khác nhau giữa cài native và Docker:

| Thông số | [install/windows-installation](https://unsloth.ai/docs/get-started/install/windows-installation), [install/mac](https://unsloth.ai/docs/get-started/install/mac), [install/linux](https://unsloth.ai/docs/get-started/install/linux), [get-started/install](https://unsloth.ai/docs/get-started/install), [install/pip-install](https://unsloth.ai/docs/get-started/install/pip-install) | [new/studio/install](https://unsloth.ai/docs/new/studio/install) | [install/docker](https://unsloth.ai/docs/get-started/install/docker) |
|---|---|---|---|
| Lệnh chạy | `unsloth studio -H 0.0.0.0 -p 8888` | Hint: `unsloth studio -p 8888`, thêm `-H 0.0.0.0` nếu cần truy cập từ máy khác; bản developer: `unsloth studio -p 8888`; truy cập từ xa "recommended": `unsloth studio --secure -p 8888` | Studio có sẵn trong container |
| Địa chỉ bind | `0.0.0.0` (mọi interface mạng) | Mặc định `127.0.0.1`; `-H 0.0.0.0` "Only use this on a trusted network" | Cổng publish "listen on every interface" |
| Cổng Studio | 8888 | 8888 (cài native); 8000 (mục Docker) | 8000 (Studio), 8888 (JupyterLab) |
| Tool chạy code phía server | — | Bật mặc định; nên thêm `--disable-tools` khi mở Studio ra ngoài | Bật mặc định trong container |
:::

### Core — cách 1: Conda

Cài Miniconda trong PowerShell:

```ps
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile ".\miniconda.exe"
Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S" -Wait
del .\miniconda.exe
```

Mở **Anaconda Powershell Prompt** và tạo môi trường:

```bash
conda create --name unsloth_env python==3.12 -y
conda activate unsloth_env
```

Chạy `nvidia-smi` để xác nhận máy có GPU. Ở góc phải trên của kết quả, ghi lại "CUDA Version".

Sau đó cài PyTorch. Trong lệnh dưới, đổi `130` theo phiên bản CUDA của bạn. Phiên bản đó phải có trên pytorch.org:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Kiểm tra PyTorch theo mục "Kiểm tra sau khi cài" ở cuối trang. **Chỉ khi PyTorch chạy được**, bạn mới cài Unsloth:

```bash
pip install unsloth
```

Nếu máy đã có PyTorch, docs nói chỉ cần `pip install unsloth` có thể là đủ.

### Core — cách 2: Docker

Docs gọi Docker là cách dễ nhất cho người dùng Windows, vì bạn không phải tự xử lý dependency. Lệnh cụ thể nằm ở mục [Docker](#docker).

::: warning Docs chưa thống nhất — NVIDIA Container Toolkit trên Windows
Hai trang nói ngược nhau về việc có phải cài NVIDIA Container Toolkit trên Windows hay không:

| Nguồn | Nội dung |
|---|---|
| [install/windows-installation](https://unsloth.ai/docs/get-started/install/windows-installation) (Method #2) | Cài `nvidia-container-toolkit` bằng `sudo apt-get install` với phiên bản `1.17.8-1`; container publish `-p 8888:8888 -p 2222:22` |
| [install/docker](https://unsloth.ai/docs/get-started/install/docker) | Windows: cài Docker Desktop + NVIDIA driver, `wsl --update`, bật WSL 2 engine; "No separate NVIDIA Container Toolkit installation is needed"; container publish cổng 8000 và 8888 |

Bảng so sánh đầy đủ ở mục Docker.
:::

### Core — cách 3: WSL

Cách này chạy Core bên trong một bản Ubuntu nằm trong Windows. Trước tiên, cài và mở Ubuntu:

```bash
wsl.exe --install Ubuntu-24.04
wsl.exe -d Ubuntu-24.04
```

Nếu máy đã có WSL, bạn chỉ cần gõ:

```bash
wsl
```

Trong WSL, cài lần lượt Python, PyTorch, rồi Unsloth và Jupyter:

```bash
sudo apt update
sudo apt install python3 python3-full python3-pip python3-venv -y
```

```bash
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu130
```

```bash
pip install unsloth jupyter
```

::: warning Lệnh trong docs có lỗi đánh máy
Khi gặp lỗi quyền, trang [install/windows-installation](https://unsloth.ai/docs/get-started/install/windows-installation) (mục WSL) gợi ý thêm một cờ vào lệnh. Nhưng cờ này bị in bằng dấu gạch dài (en dash) thay vì dấu gạch ngang ASCII:
- Với PyTorch: `pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu130 –break-system-packages`
- Với Unsloth: `pip install unsloth jupyter –-break-system-packages`

**[Nhận định]** Tùy chọn của pip viết bằng hai dấu gạch ngang ASCII (`--`). Nếu copy nguyên văn hai lệnh trên, pip có thể không nhận ra cờ. Bạn nên gõ lại phần cờ.
:::

Cài xong, mở Jupyter:

```bash
jupyter notebook
```

::: warning Cạm bẫy trên Windows
- Nếu `nvidia-smi` không chạy, bạn phải cài lại NVIDIA driver.
- Nếu PyTorch không chạy được với CUDA, docs nói có thể phải cài lại CUDA driver. Đừng cài Unsloth trước khi PyTorch chạy được.
- **vLLM không hỗ trợ Windows trực tiếp**, chỉ chạy qua WSL hoặc Linux. Bạn cần vLLM khi dùng GRPO (một kiểu reinforcement learning).
- Trên WSL, NVIDIA driver được cài **trên Windows**, không phải trong WSL. Về CUDA toolkit thì docs chưa thống nhất; xem hộp ở mục "Linux & WSL".
- Docker trên Windows không tự nhận GPU. Phiên bản `nvcc --version` trong container nên khớp với bản CUDA mà `nvidia-smi` hiển thị trên máy host. Xem thêm hướng dẫn GPU của Docker Desktop.
:::

::: details Khắc phục sự cố nâng cao (Core trên Windows)
Các bước docs đề xuất khi Core vẫn lỗi:
1. Cài `torch` và `triton`, ví dụ `pip install torch torchvision torchaudio triton`.
2. Chạy `nvcc` để xác nhận có CUDA. Nếu lỗi, cài `cudatoolkit` hoặc CUDA driver.
3. Với GPU Intel: làm theo phần Windows của hướng dẫn Intel (mục [Intel](#intel)).
4. Cài `xformers` thủ công và kiểm tra bằng `python -m xformers.info`. GPU Ampere có thể dùng `flash-attn`.
5. Đảm bảo các phiên bản Python, CUDA, CUDNN, `torch`, `triton`, `xformers` tương thích nhau. Tra bảng PyTorch Compatibility Matrix.
6. Cài `bitsandbytes` và kiểm tra bằng `python -m bitsandbytes`.
:::

### Gỡ cài đặt

- Desktop: vào Settings → Apps → Installed apps → Unsloth → Uninstall. Model đã tải và file của bạn không bị xóa.
- Studio:

```ps1
irm https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.ps1 | iex 
```

::: danger Xóa sạch dữ liệu
Lệnh `Remove-Item -Recurse -Force "$HOME\.unsloth"` xóa toàn bộ lịch sử, chat, checkpoint model và file export. Dữ liệu này không thể khôi phục. Các model trong cache Hugging Face (`%USERPROFILE%\.cache\huggingface\hub\`) không bị ảnh hưởng.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/install/windows-installation, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements, https://unsloth.ai/docs/new/studio/install

## macOS

Trên Mac, bạn dùng Desktop hoặc Studio. Việc train trên Mac và chạy Core trên Mac thì các trang docs nói chưa khớp nhau; hộp cảnh báo trong mục này liệt kê chi tiết.

### Yêu cầu

- Hệ điều hành: macOS 12 Monterey trở lên, chip Intel hoặc Apple Silicon.
- Công cụ: Homebrew, rồi chạy `brew install git`, `brew install cmake`, `brew install openssl`.
- Python: 3.11 đến dưới 3.14, trong môi trường uv, venv, conda hoặc mamba.

### Desktop

1. Tải: https://unsloth.ai/download/mac
2. Mở file `.dmg` và kéo Unsloth vào **Applications**. Sau đó mở app và chờ cài xong.
3. Vào **Select model** hoặc **Model hub**, chọn model và mức quantization phù hợp với máy. Tải về rồi chat.

Trang cài đặt Mac ghi Desktop dùng để "run and train MLX or GGUF models". Các trang khác nói khác; xem hộp "Docs chưa thống nhất" bên dưới.

### Studio (cài thủ công)

Lệnh dưới chỉ cài Studio. Nó không cài Desktop hay Core:

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

Khi muốn cập nhật, bạn chạy lại đúng lệnh này. Mỗi lần muốn mở Studio, chạy:

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

::: warning Docs chưa thống nhất — Mac: train được không, Core có chạy trên Mac không
| Thông số | [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | [new/studio/install](https://unsloth.ai/docs/new/studio/install) | [install/mac](https://unsloth.ai/docs/get-started/install/mac) |
|---|---|---|---|
| Train trên Mac | Đầu trang: "Mac: Training, MLX and GGUF inference are ALL supported"; mục Training: Studio Training chạy trên "MLX" | Đầu trang: "Mac: Like CPU - Chat + Data Recipes works for now. **MLX** training now works!"; mục System Requirements: "Chat and training and all features" | Desktop dùng để "run and train MLX or GGUF models on your Mac" |
| Unsloth Core trên Mac | Mục Core Requirements: OS "Works on Linux and Windows"; "Apple/Silicon/MLX is in the works" | — | Trang chỉ cài Desktop và Studio |
| Docker trên Mac | — | "We're working on Mac compatibility" | — |
| Gói phải cài thêm cho Studio | Homebrew, `brew install git`, `brew install cmake`, `brew install openssl`, Python 3.11 đến dưới 3.14 | Mục MacOS chỉ liệt kê macOS 12+ và môi trường Python (uv/venv/conda) | — |
:::

::: warning Docs chưa thống nhất — lệnh chạy Studio
Hai trang dùng hai lệnh khác nhau:
- [install/mac](https://unsloth.ai/docs/get-started/install/mac) dùng `unsloth studio -H 0.0.0.0 -p 8888`. Lệnh này mở Studio cho cả mạng.
- [new/studio/install](https://unsloth.ai/docs/new/studio/install) dùng `unsloth studio -p 8888`, mặc định chỉ nghe ở `127.0.0.1`. Trang này gọi `--secure` là cách "recommended" để truy cập từ xa.

Bảng đầy đủ ở mục Windows.
:::

### Gỡ cài đặt

- Desktop: mở Finder → Applications, chuột phải vào Unsloth → Move to Trash.
- Studio:

```shellscript
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh | sh
```

**Nguồn:** https://unsloth.ai/docs/get-started/install/mac, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements, https://unsloth.ai/docs/new/studio/install

## Linux & WSL

Trên Linux, bạn dùng được cả ba sản phẩm: Desktop, Studio và Core. Mục này cũng áp dụng cho WSL, tức Linux chạy bên trong Windows.

### Yêu cầu

- Hệ điều hành: Ubuntu 20.04+ hoặc distro tương tự (64-bit).
- GPU: GPU NVIDIA đã cài driver. Về CUDA toolkit, xem hộp bên dưới. Blackwell trong hộp đó là dòng GPU mới của NVIDIA, ví dụ RTX 50.
- Công cụ: Git (`sudo apt install git`). CMake thường có sẵn, nếu chưa thì `sudo apt install cmake`. C++ compiler: `build-essential`.
- Python: 3.11 đến dưới 3.14, trong uv, venv, conda hoặc mamba.

Riêng Core có yêu cầu GPU cụ thể hơn:
- GPU NVIDIA từ 2018 trở đi, CUDA Capability tối thiểu 7.0. CUDA Capability là chỉ số thế hệ kiến trúc GPU của NVIDIA.
- Ví dụ GPU đạt yêu cầu: V100, T4, Titan V, RTX 20 & 50, A100, H100, L40.
- GTX 1070, 1080 chạy được nhưng chậm.
- Thiết bị cần hỗ trợ `xformers`, `torch`, `BitsandBytes` và `triton`.

::: warning Docs chưa thống nhất — Git và CUDA toolkit trên Linux/WSL
Các chỗ trong docs nói khác nhau về việc CUDA toolkit là bắt buộc, tùy chọn, hay phải cài trong WSL:

| Thông số | [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) — danh sách "Linux & WSL" (cũng có ở [new/studio/install](https://unsloth.ai/docs/new/studio/install)) | [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) — bảng mục Training | [new/studio/install](https://unsloth.ai/docs/new/studio/install) — hint WSL |
|---|---|---|---|
| CUDA toolkit | "CUDA toolkit (12.4+ recommended, 12.8+ for blackwell)" | "Optional; `nvcc` auto-detected" | "make sure ... the **CUDA toolkit** is installed inside your WSL distro" |
| Git | `sudo apt install git` | "Usually preinstalled" | — |
:::

### Desktop

1. Tải: https://unsloth.ai/download/linux
2. Mở file `.deb` và chọn Install. Sau đó mở app và chờ cài xong.

### Studio (cài thủ công)

Cài Studio:

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

Mở Studio:

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

::: warning Docs chưa thống nhất — lệnh chạy Studio
Hai trang dùng hai lệnh khác nhau:
- [install/linux](https://unsloth.ai/docs/get-started/install/linux) dùng `unsloth studio -H 0.0.0.0 -p 8888`. Lệnh này mở Studio cho cả mạng.
- [new/studio/install](https://unsloth.ai/docs/new/studio/install) dùng `unsloth studio -p 8888`, mặc định chỉ nghe ở `127.0.0.1`. Trang này gọi `--secure` là cách "recommended" để truy cập từ xa.

Bảng đầy đủ ở mục Windows.
:::

Nếu muốn cài Studio từ repo chính (bản developer), dùng khối lệnh đầu. Muốn bản nightly, dùng khối lệnh thứ hai:

```bash
git clone https://github.com/unslothai/unsloth
cd unsloth
./install.sh --local
unsloth studio -H 0.0.0.0 -p 8888
```

```bash
git clone https://github.com/unslothai/unsloth
cd unsloth
git checkout nightly
./install.sh --local
unsloth studio -H 0.0.0.0 -p 8888
```

### Core với uv

Bốn lệnh dưới lần lượt: cài uv, tạo môi trường ảo, bật môi trường, rồi cài Unsloth:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```

Chi tiết thêm ở mục "pip & uv" bên dưới.

::: warning Cạm bẫy Linux/WSL
- Trên WSL, NVIDIA driver được cài **trên Windows**. Về CUDA toolkit, xem hộp "Docs chưa thống nhất" ở trên.
- Báo `nvidia-smi not found`: cài NVIDIA driver.
- Báo `nvcc not found`: chạy `sudo apt install nvidia-cuda-toolkit`, hoặc thêm `/usr/local/cuda/bin` vào PATH.
- Lỗi phiên bản Python: chạy `sudo apt install python3.12 python3.12-venv`. Studio cần Python 3.11 đến dưới 3.14.
- Báo `llama-server build failed`: lỗi này không làm dừng cài đặt. Unsloth vẫn chạy nhưng **mất inference GGUF**. Cài `cmake` rồi chạy lại setup.
- Build failed: xóa `~/.unsloth/llama.cpp` rồi chạy lại setup.
:::

### Gỡ cài đặt

- Desktop bản `.deb`: chạy `sudo apt remove unsloth`.
- Desktop bản AppImage: xóa file `.AppImage`.
- Studio: chạy script `uninstall.sh` như ở mục macOS.

**Nguồn:** https://unsloth.ai/docs/get-started/install/linux, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements, https://unsloth.ai/docs/new/studio/install

## AMD

Mục này dành cho máy dùng GPU AMD. Docs Unsloth quảng cáo fine-tune nhanh tới 2x và ít hơn khoảng 70% bộ nhớ trên phần cứng AMD. Cách dễ nhất là Desktop. Nếu cài Core thủ công, bạn cần thêm vài bước riêng cho AMD: PyTorch bản ROCm, bitsandbytes bản pre-release và biến môi trường.

### Yêu cầu

- GPU theo trang AMD: Radeon RDNA 3/3.5/4 (RX 6000–9000 series) trên Windows và Linux. Ngoài ra có GPU data center như MI300X (192GB).
- ROCm 6.0 trở lên khi bạn tự cài PyTorch. ROCm là nền tảng tính toán GPU của AMD, tương đương CUDA. Xem phiên bản bằng `amd-smi version`.
- Python theo trang AMD: 3.11–3.13 dùng được mọi nơi, 3.10 dùng được khi cài thủ công. Các trang khác ghi khác; xem hộp phiên bản Python ở mục "Bảng tóm tắt theo nền tảng".

::: warning Docs chưa thống nhất — dòng GPU AMD được hỗ trợ
| Thông số | [install/amd](https://unsloth.ai/docs/get-started/install/amd) | [install/docker](https://unsloth.ai/docs/get-started/install/docker) (image `unsloth/unsloth-rocm`) |
|---|---|---|
| GPU Radeon | "RDNA 3/3.5/4 (RX 6000–9000 series)" trên Windows và Linux | "covers RDNA1 and newer, plus CDNA" |
| GPU data center | MI300X (192GB) | CDNA |
| ROCm | "ROCm 6.0 or newer is required" (cài PyTorch thủ công) | Image "Built against ROCm 7.2" |
:::

### Cài nhanh

Cách dễ nhất là tải Desktop app.

::: warning Link tải trong docs
Trên [install/amd](https://unsloth.ai/docs/get-started/install/amd), nút "Download for Linux" lại trỏ tới `https://unsloth.ai/download/windows`. Trong khi đó, các trang [install/linux](https://unsloth.ai/docs/get-started/install/linux) và [get-started/install](https://unsloth.ai/docs/get-started/install) dùng `https://unsloth.ai/download/linux` cho Linux. Bản Windows là `https://unsloth.ai/download/windows`.
:::

Nếu muốn cài Studio thủ công, dùng các lệnh dưới. Việc Studio có train được trên AMD không thì docs chưa thống nhất; xem hộp "Docs chưa thống nhất — Studio có train trên AMD/Intel không" ở mục "Bảng tóm tắt theo nền tảng".

Linux:

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://unsloth.ai/install.ps1 | iex
```

### Core thủ công

Cài Core trên AMD gồm năm bước. Bước 4 là bắt buộc với AMD; đừng bỏ qua.

1) Tạo môi trường ảo (tùy chọn). Khối đầu cho Linux, khối sau cho Windows:

```bash
# Linux — swap 3.13 for whichever 3.10-3.13 you have
apt update && apt install python3.13-venv -y
python3.13 -m venv unsloth_env
source unsloth_env/bin/activate
pip install uv
```

```shellscript
py -3.13 -m venv unsloth_env
unsloth_env\Scripts\Activate.ps1
pip install uv
```

2) Cài PyTorch cho ROCm. Bạn bỏ qua bước này nếu cài Unsloth kèm extra AMD như `unsloth[rocm72-torch291]`, vì extra đó đã kèm PyTorch khớp phiên bản. Trên Linux, đổi `rocm7.1` theo phiên bản ROCm của bạn:

```bash
uv pip install "torch>=2.4,<2.11.0" "torchvision<0.26.0" "torchaudio<2.11.0" \
    --index-url https://download.pytorch.org/whl/rocm7.1 --upgrade --force-reinstall
```

Nếu dùng ROCm 7.2 (wheel torch 2.11), dùng lệnh này thay thế:

```bash
uv pip install "torch>=2.11.0,<2.12.0" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm7.2 --upgrade --force-reinstall
```

Tag index là đoạn cuối URL trong lệnh, ví dụ `rocm7.1`. Các tag có sẵn: `rocm6.0`, `rocm6.1`, `rocm6.2`, `rocm6.3`, `rocm6.4`, `rocm7.0`, `rocm7.1`, `rocm7.2`. Nếu máy bạn có ROCm 6.5–6.9, dùng `rocm6.4`. Nếu có ROCm 7.3+, dùng `rocm7.2`.

::: details Lệnh tự dò phiên bản ROCm và cài PyTorch
```bash
ROCM_TAG="$({ command -v amd-smi >/dev/null 2>&1 && amd-smi version 2>/dev/null | awk -F'ROCm version: ' 'NF>1{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}'; } || { [ -r /opt/rocm/.info/version ] && awk -F. '{print "rocm"$1"."$2; exit}' /opt/rocm/.info/version; } || { command -v hipconfig >/dev/null 2>&1 && hipconfig --version 2>/dev/null | awk -F': *' '/HIP version/{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}'; } || { command -v dpkg-query >/dev/null 2>&1 && ver="$(dpkg-query -W -f="${Version}\n" rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F'[.-]' '{print "rocm"$1"."$2; exit}' <<<"$ver"; } || { command -v rpm >/dev/null 2>&1 && ver="$(rpm -q --qf '%{VERSION}\n' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F'[.-]' '{print "rocm"$1"."$2; exit}' <<<"$ver"; })"; [ -n "$ROCM_TAG" ] && uv pip install "torch>=2.4,<2.11.0" "torchvision<0.26.0" "torchaudio<2.11.0" --index-url "https://download.pytorch.org/whl/$ROCM_TAG" --upgrade --force-reinstall
```
:::

3) Cài Unsloth với extra AMD:

```bash
uv pip install unsloth[amd]
```

4) **Bắt buộc với AMD**: cài bitsandbytes bản pre-release. bitsandbytes là thư viện nén model xuống 4-bit hoặc 8-bit. Ở bước này phải dùng `pip`, không dùng `uv`, vì uv từ chối wheel này do phiên bản trong tên file bị lệch:

```bash
# x86_64 systems:
pip install --force-reinstall --no-cache-dir --no-deps \
    "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl"

# aarch64 systems: replace x86_64 with aarch64 in the URL above

# Fallback if the URL is unreachable:
# pip install --force-reinstall --no-cache-dir --no-deps "bitsandbytes>=0.49.1"
```

5) Đặt biến môi trường trước khi train:

```bash
export HSA_OVERRIDE_GFX_VERSION=9.4.2  # Required for AMD MI300X
export HF_HUB_DISABLE_XET=1            # Fixes HuggingFace download issues on AMD
```

::: warning Cạm bẫy AMD
- bitsandbytes bản ≤ 0.49.2 có **lỗi NaN khi decode 4-bit trên mọi GPU AMD**. Vì vậy phải dùng bản pre-release ở trên.
- `HSA_OVERRIDE_GFX_VERSION=9.4.2` bảo ROCm coi GPU là gfx942 (MI300X). Thiếu biến này, một số kernel có thể không biên dịch hoặc không chạy được.
- Flash Attention 2 không có trên AMD. Unsloth tự chuyển sang Xformers, nên bạn có thể bỏ qua cảnh báo về việc này.
- Trên Windows, dùng Python 3.12 nếu cài extra `unsloth[rocm72-torch291]`.
:::

::: warning Docs chưa thống nhất — ROCm 7.2+, bitsandbytes và extra AMD
Ngay trong cùng trang [install/amd](https://unsloth.ai/docs/get-started/install/amd) có những câu mâu thuẫn nhau. Cột "Câu A" và "Câu B" là hai câu khác nhau trên trang đó:

| Thông số | Câu A | Câu B |
|---|---|---|
| ROCm 7.2 | "on ROCm 7.2 use this instead" → index `rocm7.2`, `torch>=2.11.0,<2.12.0` | "If your ROCm version is 7.2 or higher, replace `$ROCM_TAG` ... with `rocm7.1`, no PyTorch wheels exist yet for 7.2+" |
| ROCm 7.3+ | "ROCm 7.3+ uses `rocm7.2`" | Như trên: dùng `rocm7.1` |
| bitsandbytes | "versions ≤ 0.49.2 have a 4-bit decode NaN bug on every AMD GPU" | Lệnh dự phòng: `"bitsandbytes>=0.49.1"` |
| Extra khi cài Unsloth | Mục Install PyTorch nhắc extra `unsloth[rocm72-torch291]` (Windows: "use 3.12" khi cài extra này) | Mục Install Unsloth dùng `uv pip install unsloth[amd]` |
:::

AMD còn có notebook one-click trên AMD Dev Cloud. Notebook này cho dùng miễn phí GPU MI300X 192GB VRAM, không cần đăng ký hay thẻ.

**Nguồn:** https://unsloth.ai/docs/get-started/install/amd, https://unsloth.ai/docs/get-started/install/docker

## Intel

Với GPU Intel, trang hướng dẫn Intel chỉ nói tới Core, và bạn cài Core bằng cách build từ source. Trên Windows cần thêm vài bước cấu hình runtime.

### Yêu cầu

- GPU Intel: Data Center GPU Max Series, Arc Series, hoặc Intel Ultra AIPC.
- Hệ điều hành: Linux (khuyến nghị Ubuntu 22.04+) hoặc Windows 11 (khuyến nghị).
- Chỉ trên Windows: Intel oneAPI Base Toolkit 2025.2.1. Chọn đúng bản 2025.2.1.
- Driver: nên dùng driver Intel Graphics mới nhất.
- Python: 3.10+.

### Cài Unsloth với hỗ trợ Intel (Core)

Tạo và bật môi trường conda:

```bash
conda create -n unsloth-xpu python==3.10
conda activate unsloth-xpu
```

Tải mã nguồn Unsloth và cài kèm extra cho GPU Intel:

```bash
git clone https://github.com/unslothai/unsloth.git
cd unsloth
pip install .[intel-gpu-torch290]
```

Chỉ trên Linux: bạn có thể cài thêm vLLM để dùng cho inference và RL. Làm theo hướng dẫn Intel XPU của vLLM.

### Cấu hình runtime chỉ cho Windows

Mở Command Prompt với quyền Administrator và bật long path. Mỗi máy chỉ cần làm một lần:

```bash
powershell -Command "Set-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" -Name "LongPathsEnabled" -Value 1
```

Tiếp theo, tải và giải nén `level-zero-win-sdk-1.20.2.zip` từ GitHub (repo oneapi-src/level-zero, release v1.20.2). Sau đó mở Command Prompt, bật môi trường conda `unsloth-xpu` và chạy:

```bash
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" -
set ZE_PATH=path\to\the\unzipped\level-zero-win-sdk-1.20.2
```

::: warning Lệnh trong docs có thể sai cú pháp
Trên [install/intel](https://unsloth.ai/docs/get-started/install/intel):
- Lệnh `powershell -Command "Set-ItemProperty -Path "HKLM:..." -Name "LongPathsEnabled" -Value 1` có ngoặc kép lồng trong ngoặc kép, và thiếu ngoặc kép đóng ở cuối. **[Nhận định]** Copy nguyên văn có thể lỗi; cần kiểm tra lại cú pháp.
- Dòng `call "...\oneAPI\setvars.bat" -` kết thúc bằng ` -`. Nguồn không giải thích ý nghĩa của nó.
:::

::: warning Cạm bẫy Intel
- Intel Ultra AIPC trên Windows: iGPU dùng chung bộ nhớ với hệ thống, mặc định khoảng **57%** RAM. Một số trường hợp có thể cần tăng tỷ lệ này: model lớn (ví dụ Qwen3-32B), context dài, batch lớn hoặc LoRA rank lớn. Bạn tăng qua registry, ở khóa `SystemPartitionCommitLimitPercentage` tại `Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager`.
- Khi gặp OOM, bạn có thể: giảm `per_device_train_batch_size`, dùng model nhỏ hơn, giảm `max_seq_length`, giảm LoRA rank (`r=8` thay vì 16 hoặc 32). Với GRPO, giảm thêm `num_generations`.
- Phiên bản Python: trang Intel dùng 3.10 (conda), các trang khác ghi khác. Xem hộp phiên bản Python ở mục "Bảng tóm tắt theo nền tảng".
:::

**Nguồn:** https://unsloth.ai/docs/get-started/install/intel, https://unsloth.ai/docs/get-started/install/windows-installation

## CPU-only

Máy không có GPU vẫn dùng được Studio, nhưng chỉ cho một số việc:
- **Chat với model GGUF**.
- **Data Recipes** (tạo dataset từ tài liệu).
- **Export**: docs ghi "coming very soon".

Yêu cầu giống Linux (trừ NVIDIA driver) và macOS. Bạn cài như Studio trên Linux hoặc macOS, bằng `install.sh`.

::: info Không train trên CPU
Nguồn chỉ nêu Chat + Data Recipes cho CPU, không nhắc tới training. Danh sách thiết bị train được chỉ gồm NVIDIA, AMD, Intel và Mac.
:::

Có hai biến môi trường liên quan khi cài và chạy:
- `UNSLOTH_NO_TORCH`: bỏ qua PyTorch, tức chế độ chỉ dùng GGUF. **[Nhận định]** Nguồn không nói chế độ GGUF-only dành riêng cho CPU.
- `UNSLOTH_CPU_THREADS`: giới hạn số luồng CPU trên máy nhiều nhân, ví dụ `UNSLOTH_CPU_THREADS=8 unsloth studio -p 8888`.

Lệnh cài với `UNSLOTH_NO_TORCH` trên Linux hoặc macOS, rồi trên Windows:

```bash
curl -fsSL https://unsloth.ai/install.sh | UNSLOTH_NO_TORCH=1 sh
```

```powershell
$env:UNSLOTH_NO_TORCH=1; irm https://unsloth.ai/install.ps1 | iex
```

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements, https://unsloth.ai/docs/new/studio/install

## Docker

Docker cho bạn chạy Unsloth trong một container, tức môi trường đóng gói sẵn mọi dependency. Nhờ vậy bạn không phải tự xử lý dependency trên máy.

Có hai image chính thức:
- `unsloth/unsloth` cho GPU NVIDIA. Blackwell và RTX 50 cũng dùng image này.
- `unsloth/unsloth-rocm` cho GPU AMD.

Container đã có sẵn Unsloth Studio (cổng 8000) và JupyterLab (cổng 8888). Từ tháng 9/2026, image đã cập nhật Studio và AMD.

### NVIDIA — terminal (bash)

Yêu cầu: NVIDIA driver **570.26 trở lên**. Chạy lệnh từ thư mục dự án; bên trong container, thư mục đó sẽ là `/workspace/host`:

```bash
docker run -d --name unsloth --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -p 8888:8888 \
  -e JUPYTER_PASSWORD="mypassword" \
  -v "$PWD":/workspace/host \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  -v unsloth-studio:/opt/unsloth-studio \
  unsloth/unsloth
```

### NVIDIA — PowerShell

Cùng lệnh trên, viết cho PowerShell:

```powershell
docker run -d --name unsloth --gpus all --ipc=host `
  --ulimit memlock=-1 --ulimit stack=67108864 `
  -p 8000:8000 -p 8888:8888 `
  -e JUPYTER_PASSWORD="mypassword" `
  -v "${PWD}:/workspace/host" `
  -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface" `
  -v "unsloth-studio:/opt/unsloth-studio" `
  unsloth/unsloth
```

### Chuẩn bị

Trước khi chạy các lệnh trên, máy cần có Docker và cách để container dùng GPU NVIDIA.

Cài Docker (Linux):

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
```

NVIDIA Container Toolkit (Linux):

```bash
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/install_nvidia_toolkit.sh -o install_nvidia_toolkit.sh && sudo -E bash install_nvidia_toolkit.sh
```

Trên Windows:
1. Cài Docker Desktop và NVIDIA driver mới nhất.
2. Chạy `wsl --update`.
3. Trong Docker Desktop, bật **Use the WSL 2 based engine**.

Trên Windows, trang Docker ghi **không cần** cài NVIDIA Container Toolkit riêng.

### AMD — Linux

Yêu cầu: driver `amdgpu` phải hoạt động. Image được build với ROCm 7.2, hỗ trợ RDNA1 trở lên và CDNA. Bạn không cần container toolkit.

Đoạn đầu của lệnh dưới tự gom các thiết bị GPU và nhóm quyền cần thiết vào biến `GPU_FLAGS`, rồi mới chạy container:

```bash
GPU_FLAGS="--device /dev/kfd"
[ -e /dev/dri ] && GPU_FLAGS="$GPU_FLAGS --device /dev/dri"
for g in video render; do
  gid=$(getent group "$g" | cut -d: -f3)
  [ -n "$gid" ] && GPU_FLAGS="$GPU_FLAGS --group-add $gid"
done
docker run -d --name unsloth \
  $GPU_FLAGS \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -p 8888:8888 \
  -e JUPYTER_PASSWORD="mypassword" \
  -v "$PWD":/workspace/host \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  -v unsloth-studio:/opt/unsloth-studio \
  unsloth/unsloth-rocm
```

::: details AMD — trong WSL2 và PowerShell
WSL2 không có `/dev/kfd`, nên GPU được truyền vào container qua `/dev/dxg`. Chạy lệnh bên trong distro WSL2, ví dụ `wsl -d Ubuntu`. Máy cần driver AMD cho Windows có hỗ trợ WSL2.

```bash
docker run -d --name unsloth --device /dev/dxg --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm/lib \
  -p 8000:8000 -p 8888:8888 \
  -e JUPYTER_PASSWORD="mypassword" \
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro \
  -v "$PWD":/workspace/host \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  -v unsloth-studio:/opt/unsloth-studio \
  unsloth/unsloth-rocm
```

PowerShell (cần Docker Desktop dùng WSL 2 engine):

```powershell
docker run -d --name unsloth --device /dev/dxg --ipc=host `
  --ulimit memlock=-1 --ulimit stack=67108864 `
  -e HSA_ENABLE_DXG_DETECTION=1 `
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm/lib `
  -p 8000:8000 -p 8888:8888 `
  -e JUPYTER_PASSWORD="mypassword" `
  -v "/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro" `
  -v "${PWD}:/workspace/host" `
  -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface" `
  -v "unsloth-studio:/opt/unsloth-studio" `
  unsloth/unsloth-rocm
```
:::

### Sau khi chạy container

- Xem mật khẩu: chạy `docker logs -f unsloth` và tìm dòng **Unsloth container ready**.
- Mở Studio: vào `http://localhost:8000`, đăng nhập bằng user `unsloth`.
- Mở JupyterLab: vào `http://localhost:8888`, dùng mật khẩu đã đặt trong `JUPYTER_PASSWORD`.
- Quên mật khẩu Studio: chạy `docker exec unsloth unsloth studio reset-password --username unsloth`.
- Cập nhật: chạy `docker pull unsloth/unsloth` (hoặc `unsloth/unsloth-rocm`), rồi `docker rm -f unsloth`. Sau đó chạy lại lệnh Quickstart với cùng các cờ `-v`. Model, file và dữ liệu Studio vẫn được giữ.

::: warning Cạm bẫy Docker
- Studio và JupyterLab **dùng chung GPU**. Model đã load trong Studio giữ VRAM cho tới khi bạn unload. Hãy unload trước khi train trong notebook.
- Lưu kết quả dưới `/workspace/host`. Thứ gì ghi ra ngoài volume sẽ mất khi xóa container.
- Dùng **named volume** cho `/opt/unsloth-studio`, không dùng thư mục host. Lý do: Studio cần tạo symlink, mà host Windows hoặc macOS có thể không cho phép, làm container dừng ngay khi khởi động.
- Nếu chưa đổi mật khẩu tự sinh, Studio tự tắt sau 3600 giây (`UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT`). Khi đó chạy `docker restart unsloth` rồi đổi mật khẩu.
- Cổng bị chiếm: đổi cổng phía host, ví dụ `-p 8001:8000`.
- Container chạy bằng root. Nếu cần file thuộc về user trên máy host, dùng `unsloth/unsloth:core` với `--user`.
- Cổng publish lắng nghe trên mọi interface, và Studio cùng JupyterLab dùng HTTP thường. Trên cloud, hãy bind `127.0.0.1` hoặc đặt `-e UNSLOTH_STUDIO_SECURE=1`. Để truy cập từ xa, dùng `ssh -L 8000:localhost:8000 -L 8888:localhost:8888 user@your-server`.
:::

::: warning Docs chưa thống nhất — lệnh và yêu cầu Docker
Ba trang đưa ra ba lệnh Docker khác nhau về cổng, thư mục mount và cờ. Hướng dẫn ở trên theo trang [install/docker](https://unsloth.ai/docs/get-started/install/docker):

| Thông số | [install/windows-installation](https://unsloth.ai/docs/get-started/install/windows-installation) (Method #2) | [new/studio/install](https://unsloth.ai/docs/new/studio/install) (mục Docker) | [install/docker](https://unsloth.ai/docs/get-started/install/docker) |
|---|---|---|---|
| NVIDIA Container Toolkit trên Windows | Cài bằng `apt-get` với `NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1` | — | "No separate NVIDIA Container Toolkit installation is needed" (Windows); Linux dùng script `install_nvidia_toolkit.sh` |
| Cổng publish | `-p 8888:8888 -p 2222:22` | `-p 8888:8888 -p 8000:8000 -p 2222:22` | `-p 8000:8000 -p 8888:8888`; cổng 22 chỉ khi đặt `SSH_KEY` |
| Mount | `-v $(pwd)/work:/workspace/work` | `-v $(pwd)/work:/workspace/work` | `"$PWD":/workspace/host`, cache Hugging Face, volume `unsloth-studio:/opt/unsloth-studio` |
| Cờ khác | `--gpus all` | `--gpus all` | `--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864` |
| Truy cập | Jupyter Lab `http://localhost:8888` | Studio `http://localhost:8000` | Studio `http://localhost:8000`, JupyterLab `http://localhost:8888` |
| Yêu cầu driver | — | — | NVIDIA driver 570.26 trở lên |
| Mac | — | "We're working on Mac compatibility" | — |
:::

::: warning Docs chưa thống nhất — dòng GPU AMD được hỗ trợ
| Thông số | [install/amd](https://unsloth.ai/docs/get-started/install/amd) | [install/docker](https://unsloth.ai/docs/get-started/install/docker) (image `unsloth/unsloth-rocm`) |
|---|---|---|
| GPU Radeon | "RDNA 3/3.5/4 (RX 6000–9000 series)" trên Windows và Linux | "covers RDNA1 and newer, plus CDNA" |
| GPU data center | MI300X (192GB) | CDNA |
| ROCm | "ROCm 6.0 or newer is required" (cài PyTorch thủ công) | Image "Built against ROCm 7.2" |
:::

**Nguồn:** https://unsloth.ai/docs/get-started/install/docker, https://unsloth.ai/docs/get-started/install/windows-installation, https://unsloth.ai/docs/new/studio/install, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements

## pip & uv: cài Unsloth Core bằng code

Mục này dành cho người muốn dùng Unsloth Core trong code Python của mình. Bạn cài Core như một thư viện Python bình thường, bằng pip hoặc uv. uv là trình quản lý gói Python nhanh, dùng thay pip.

Cài uv (khối đầu cho Linux và macOS, khối sau cho Windows PowerShell):

```shellscript
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Cài Core bằng uv (docs khuyến nghị):

```bash
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```

Hoặc chỉ dùng pip:

```bash
pip install unsloth
```

Cài kèm vLLM (engine inference hiệu năng cao):

```bash
uv pip install unsloth vllm --torch-backend=auto
```

Cài bản mới nhất từ nhánh main:

```bash
uv pip install unsloth --torch-backend=auto
pip uninstall unsloth unsloth_zoo -y && pip install --no-deps git+https://github.com/unslothai/unsloth_zoo.git && pip install --no-deps git+https://github.com/unslothai/unsloth.git
```

Để không làm hỏng các gói Python của hệ thống, bạn nên cô lập Unsloth trong một venv:

```bash
apt install python3.10-venv python3.11-venv python3.12-venv python3.13-venv -y
python -m venv unsloth_env
source unsloth_env/bin/activate
pip install --upgrade pip && pip install uv
uv pip install unsloth --torch-backend=auto
```

Trong Jupyter hoặc Colab, thêm tiền tố `!` trước lệnh. Nếu máy đã có torch hoặc transformers ở phiên bản khác, `pip install unsloth` sẽ tự cài bản mới nhất của các thư viện đó.

Nếu vẫn còn lỗi dependency, ép cài lại:

```bash
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth_zoo
```

::: details Cài pip nâng cao theo torch/CUDA (không dùng nếu có Conda)
Lệnh thay đổi theo phiên bản torch và CUDA trên máy bạn. Ví dụ với torch 2.4 và CUDA 12.1:

```bash
pip install --upgrade pip
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

Với torch 2.5 và CUDA 12.4:

```bash
pip install --upgrade pip
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

GPU Ampere (A100, H100, RTX3090) trở lên dùng biến thể `-ampere`. Nếu không chắc chọn lệnh nào, chạy lệnh dưới; nó in ra câu lệnh pip tối ưu cho máy bạn:

```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```
:::

::: warning Docs chưa thống nhất — CUDA/torch cho cài pip nâng cao
Trên cùng trang [install/pip-install](https://unsloth.ai/docs/get-started/install/pip-install), phần văn bản và script dò tự động hỗ trợ các bản CUDA, torch khác nhau:

| Thông số | Văn bản "Advanced Pip Installation" | Script dò tự động (`_auto_install.py` / đoạn Python REPL) |
|---|---|---|
| CUDA | `cu118`, `cu121`, `cu124` | 11.8, 12.1, 12.4, 12.6, 12.8, 13.0 |
| torch | `torch211`, `torch212`, `torch220`, `torch230`, `torch240` (ví dụ thêm `torch250`) | Tới `torch291` (torch 2.9.x); mới hơn báo "too new" |
| Biến thể Ampere | Dùng `cu118-ampere` / `cu121-ampere` / `cu124-ampere` cho A100, H100, RTX3090 trở lên | Luôn tắt `-ampere`: "is_ampere is broken due to flash-attn" |
:::

::: warning Docs chưa thống nhất — phiên bản Python
| Nguồn | Giá trị | Ngữ cảnh |
|---|---|---|
| [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | "version 3.11 up to, but not including, 3.14" | Yêu cầu Windows, MacOS, Linux & WSL |
| [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | "Python 3.11–3.13 is required" | Mục Training (Studio) |
| [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | "Python 3.13 is supported!" (không nêu mức tối thiểu) | Mục Unsloth Core |
| [new/studio/install](https://unsloth.ai/docs/new/studio/install) | "version 3.11 up to, but not including, 3.14"; bảng Troubleshooting gợi ý `sudo apt install python3.12 python3.12-venv` | Studio |
| [install/windows-installation](https://unsloth.ai/docs/get-started/install/windows-installation) | `conda create --name unsloth_env python==3.12 -y` | Core qua Conda trên Windows |
| [install/linux](https://unsloth.ai/docs/get-started/install/linux), [install/pip-install](https://unsloth.ai/docs/get-started/install/pip-install) | `uv venv unsloth_env --python 3.13` | Core qua uv |
| [install/pip-install](https://unsloth.ai/docs/get-started/install/pip-install) | `apt install python3.10-venv python3.11-venv python3.12-venv python3.13-venv -y` | Core qua venv |
| [install/pip-install](https://unsloth.ai/docs/get-started/install/pip-install) | `winget install -e --id Python.Python.3.13 --source winget` | Studio developer/nightly trên Windows |
| [install/amd](https://unsloth.ai/docs/get-started/install/amd) | "any 3.11-3.13 works everywhere (3.10 works for manual installs"; Windows: "use 3.12 if installing the unsloth[rocm72-torch291] extra" | AMD |
| [install/intel](https://unsloth.ai/docs/get-started/install/intel) | "Python: 3.10+"; `conda create -n unsloth-xpu python==3.10` | Intel (Core) |
:::

**Nguồn:** https://unsloth.ai/docs/get-started/install/pip-install, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements

## Google Colab

Colab cho dùng GPU T4 miễn phí. **[Nhận định]** Vì vậy đây là cách thử Unsloth khi máy bạn không có GPU phù hợp, mà không cần cài gì trên máy.

**Chạy notebook Unsloth:**
- Bấm nút Play ở từng cell theo thứ tự, không bỏ cell nào. Hoặc chọn **Runtime → Run all** để chạy hết.
- Cell đầu tiên cài Unsloth từ GitHub cùng các gói phụ thuộc.

**Chạy Unsloth Studio trên Colab:**
- Có notebook Colab miễn phí cho **Unsloth Studio**: https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb
- Trên T4, notebook này train và chạy được hầu hết model tới **22B tham số**. Model lớn hơn thì cần đổi sang GPU lớn hơn.
- Cách mở: bấm "Run all", cuộn tới **Start Unsloth Studio** rồi bấm **Open Unsloth Studio**.

::: warning Lưu ý Colab
- Link mở Studio có thể báo lỗi nếu bạn tắt cookie, dùng adblocker hoặc dùng Mozilla. Khi đó bạn vẫn có thể cuộn xuống dưới nút để thấy giao diện.
- Colab có thể tắt phiên GPU nếu thấy bạn không hoạt động trên trang.
:::

::: details Ví dụ code Colab (fine-tune gpt-oss-20b)
```python
from unsloth import FastLanguageModel, FastModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
max_seq_length = 2048 # Supports RoPE Scaling internally, so choose any!
# Get LAION dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files = {"train" : url}, split = "train")

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit", #or choose any model

] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4-bit quantization. False = 16-bit LoRA.
    load_in_8bit = False, # 8-bit quantization
    load_in_16bit = False, # [NEW!] 16-bit LoRA
    full_finetuning = False, # Use for full fine-tuning.
    # token = "hf_...", # use one if using gated models
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    args = SFTConfig(
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)
trainer.train()

# Go to https://docs.unsloth.ai for advanced tips like
# (1) Saving to GGUF / merging to 16bit for vLLM
# (2) Continued training from a saved LoRA adapter
# (3) Adding an evaluation loop / OOMs
# (4) Customized chat templates
```
:::

Ý nghĩa từng tham số trong code trên được giải thích ở trang [Fine-tuning](/fine-tuning/).

**Nguồn:** https://unsloth.ai/docs/get-started/install/google-colab, https://unsloth.ai/docs/new/studio/install

## Cập nhật

Mỗi sản phẩm có cách cập nhật riêng. Desktop cập nhật độc lập với Studio và Core cài thủ công, nên bạn cần cập nhật từng cái mình đang dùng.

### Desktop

Desktop tự kiểm tra cập nhật mỗi khi khởi động. Bạn cũng có thể kiểm tra tay ở **Settings → General → Check for updates**.
- macOS, Windows: chọn **Update now**. App tự cài bản mới và mở lại.
- Linux AppImage: cập nhật được ngay trong app.
- Linux `.deb`: không tự cập nhật. Bạn tải gói mới từ trang download Linux và cài đè lên.

### Studio

Chạy lại chính lệnh cài. Khối đầu cho Linux và macOS, khối sau cho Windows:

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

```bash
irm https://unsloth.ai/install.ps1 | iex
```

Với bản developer (cài từ repo), chạy `git pull` rồi chạy lại `./install.sh --local`.

### Core

Cập nhật Unsloth và thư viện đi kèm `unsloth_zoo`:

```bash
pip install --upgrade unsloth unsloth_zoo
```

Cập nhật mà không cập nhật dependency:

```bash
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth_zoo
```

Để quay về phiên bản cũ, dùng lệnh dưới và thay `2025.1.5` bằng số phiên bản của release trên GitHub:

```bash
pip install --force-reinstall --no-cache-dir --no-deps unsloth==2025.1.5
```

### Docker

Xem phần "Sau khi chạy container" ở mục [Docker](#docker).

**Nguồn:** https://unsloth.ai/docs/get-started/install/updating, https://unsloth.ai/docs/new/studio/install, https://unsloth.ai/docs/get-started/install/docker

## Kiểm tra sau khi cài

Mục này gom các lệnh để bạn xác nhận từng phần đã cài đúng: Studio, GPU, PyTorch và các thư viện phụ.

**Studio:** mở `http://127.0.0.1:8888`. Nếu dùng Docker thì mở `http://localhost:8000`. Lần đầu, trình duyệt chuyển tới trang tạo mật khẩu (`/change-password`), sau đó vào trang Chat.

**GPU NVIDIA:** chạy `nvidia-smi`. Nếu không có lệnh này hoặc không hiện bảng, cài lại NVIDIA driver.
- Trong Docker: `docker exec unsloth nvidia-smi`.
- AMD trong Docker: `docker exec unsloth rocm-smi`. Nếu báo lỗi quyền trên `/dev/kfd`, nghĩa là lệnh chạy container thiếu các id `--group-add`.

**PyTorch + CUDA (Core):** chạy đoạn code sau trong `python`. Kết quả đúng là `True` và một ma trận toàn số 10:

```python
import torch
print(torch.cuda.is_available())
A = torch.ones((10, 10), device = "cuda")
B = torch.ones((10, 10), device = "cuda")
A @ B
```

**Intel XPU:**

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
print(f"XPU device count: {torch.xpu.device_count()}")
print(f"XPU device name: {torch.xpu.get_device_name(0)}")
```

**Thư viện phụ:** kiểm tra bằng `python -m xformers.info`, `python -m bitsandbytes` và `nvcc --version`.

::: details Script train thử Unsloth (từ trang Windows)
```python
from unsloth import FastLanguageModel, FastModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
max_seq_length = 512
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files = {"train" : url}, split = "train")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-270m-it",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4-bit quantization. False = 16-bit LoRA.
    load_in_8bit = False, # 8-bit quantization
    load_in_16bit = False, # 16-bit LoRA
    full_finetuning = False, # Use for full fine-tuning.
    trust_remote_code = False, # Enable to support new models
    # token = "hf_...", # use one if using gated models
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    args = SFTConfig(
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
        dataset_num_proc = 1,
    ),
)
trainer.train()
```

Khi chạy đúng, log in ra banner `🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.`. Kèm theo là thông tin GPU và phiên bản Torch, CUDA, Triton. Sau đó quá trình train bắt đầu. Ví dụ trong docs: RTX 3060 12 GB, Torch 2.10.0+cu130, Platform: Windows.
:::

::: warning Docs chưa thống nhất — bảo mật khi mở Studio ra ngoài
Các trang hướng dẫn cài đều dùng lệnh mở Studio cho cả mạng. Trang Studio chính và trang Docker lại khuyên hạn chế truy cập từ ngoài:

| Nguồn | Nội dung |
|---|---|
| [install/windows-installation](https://unsloth.ai/docs/get-started/install/windows-installation), [install/mac](https://unsloth.ai/docs/get-started/install/mac), [install/linux](https://unsloth.ai/docs/get-started/install/linux), [install/pip-install](https://unsloth.ai/docs/get-started/install/pip-install) | Lệnh chạy trong hướng dẫn: `unsloth studio -H 0.0.0.0 -p 8888` |
| [new/studio/install](https://unsloth.ai/docs/new/studio/install) | Mặc định bind `127.0.0.1`; `--secure` (HTTPS qua Cloudflare tunnel) là "recommended"; `-H 0.0.0.0` "Only use this on a trusted network"; tool phía server (web search, chạy code Python/terminal) bật mặc định, nên thêm `--disable-tools` khi mở ra ngoài |
| [install/docker](https://unsloth.ai/docs/get-started/install/docker) | Cổng publish nghe trên mọi interface; trên cloud bind `127.0.0.1` hoặc `-e UNSLOTH_STUDIO_SECURE=1` |
:::

**Nguồn:** https://unsloth.ai/docs/new/studio/install, https://unsloth.ai/docs/get-started/install/windows-installation, https://unsloth.ai/docs/get-started/install/intel, https://unsloth.ai/docs/get-started/install/docker
