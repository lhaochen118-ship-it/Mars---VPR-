# Environment summary — VPR-Space reproduction

*Cập nhật 06/11/2026.*

## Môi trường đánh giá (giống tác giả)

| Thành phần | Phiên bản | Nguồn |
|---|---|---|
| Python | 3.13.16 (Kaggle). Tác giả không nêu; các gói họ ghim yêu cầu ≥ 3.11 | Kaggle image |
| torch / torchvision | **2.6.0+cu124 / 0.21.0+cu124** (đúng `requirements.txt` của tác giả) | download.pytorch.org, cài offline trong venv `/tmp/vprenv` |
| numpy / pillow / scipy / opencv | **2.4.4 / 12.2.0 / 1.17.1 / 5.0.0.93** (đúng file ghim) | PyPI, cài offline |
| huggingface_hub / safetensors / einops | 1.23.0 / 0.8.0 / 0.8.2 (đúng file ghim) | PyPI |
| Bổ sung cho các phương pháp khác | pytorch_lightning 2.6.6, lightning_utilities, torchmetrics, pytorch_metric_learning 3.0.1, fast_pytorch_kmeans, psutil, timm 1.0.30 (`--no-deps`) | PyPI |
| DenseVLAD, kiểm tra môi trường | Môi trường mặc định Kaggle: torch 2.11.0+cu130, kornia 0.9.0 | Kaggle image |
| Mặc định TF32 | `matmul.allow_tf32 = False`, `cudnn.allow_tf32 = True` (mặc định PyTorch, tác giả không thay đổi) | – |

## Code upstream ghim theo commit

| Repo | Commit | Ngày |
|---|---|---|
| rfechner/VPR-Space (code đánh giá) | `eae17bf` | 27/10/2026 |
| amaralibey/Bag-of-Queries | `1a4965e` | 16/03/2025 |
| gmberton/MegaLoc | `5fe0dd6` | 27/07/2026 |
| facebookresearch/dinov2 | `7764ea0` | 03/06/2026 |
| serizba/salad | `6aede13` | 27/10/2025 |
| AnyLoc/DINO | `eea7d4c` | 29/07/2024 |
| gmberton/CosPlace | `52b56e9` | 15/03/2026 |
| gmberton/VPR-methods-evaluation (MixVPR, NetVLAD) | `9c8dfd4` | 13/08/2026 |

## Phần cứng

| Loại phiên | GPU | CPU / RAM | Dùng cho |
|---|---|---|---|
| L4×4 (qua cuộc thi `gemma-4-developer-agent`) | 4× L4 23 GB (sm_89), driver 580 | 46 vCPU / 188 GB | Toàn bộ suy luận |
| T4×2 | 2× T4 16 GB (sm_75) | – | Bản sao Nhóm A, để đo độ nhạy phần cứng |
| CPU + Internet | – | – | Tải dữ liệu, kiểm chứng, đóng gói deps |

## Dữ liệu

- `rfechner/VPR-Space` v0.1.0: 11 phần, 20.288.675.840 byte. SHA-256 khớp 11/11.
- Thư mục `vpr-space-v0/`: `images` (8.784 PNG), `poses` (25), `vpr_gt/VPR-GT-RPE-0` (16), `misc/flagged_errors`, `misc/rpe_estimates`.
