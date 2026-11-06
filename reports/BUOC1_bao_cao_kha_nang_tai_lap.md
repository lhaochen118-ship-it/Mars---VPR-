# BÁO CÁO BƯỚC 1 — Khả năng tái lập chính xác bài báo VPR-Space

| | |
|---|---|
| **Bài báo** | *VPR-Space: A Benchmark for Visual Place Recognition in Space Analog Domains* — Richard Fechner, Peer Neubert (Institute for Computational Visualistics, Univ. Koblenz). arXiv:[2610.35642](https://arxiv.org/abs/2610.35642) v1 (cs.RO), ngày 26/10/2026 |
| **Repo tác giả** | [github.com/rfechner/VPR-Space](https://github.com/rfechner/VPR-Space): giấy phép MIT, HEAD `eae17bf` (27/10/2026), pre-release `v0.1.0` (15/10/2026) |
| **Ngày lập** | 06/11/2026 |
| **Phạm vi** | Bước 1: xác định chính xác phần nào **tái lập được chính xác**, phần nào **chỉ tái lập gần đúng**, phần nào **không công bố**. Cuối báo cáo có kế hoạch đề xuất cho Bước 2, bám theo tài liệu vận hành *Reproduce VPR - Formatted Headings.docx* trên nhánh `main`, và danh sách các quyết định cần bạn chốt. |

---

## 0. Tóm tắt

1. **Bài báo không huấn luyện mô hình nào.** Đóng góp gồm: (i) bộ benchmark 16 cặp database/query, (ii) cách dựng ground-truth (GT) mới tên **VPR-GT-RPE-0**, (iii) đánh giá 9 phương pháp VPR có sẵn. Vì vậy **không tồn tại "trọng số riêng của tác giả"**. Mọi trọng số đều là trọng số pretrained công khai của các phương pháp gốc. Tái lập kết quả nghĩa là chạy suy luận (inference) các mô hình pretrained trên 16 bộ dữ liệu đã phát hành, rồi tính metric bằng code của tác giả.
2. **Tái lập CHÍNH XÁC được** (mọi thứ cần thiết đã công bố và đã kiểm chứng):
   dữ liệu của 16 cặp, ma trận GT `VPR-GT-RPE-0`, code đánh giá và metric, cùng **2 trong 9 phương pháp có cấu hình đầy đủ: MegaLoc và BoQ**. Tương ứng với cột **MeLo** và **BoQ** của Bảng II và III, các giá trị similarity ở **Hình 6** (BoQ), và **Bảng IV** (thống kê dữ liệu).
3. **Chỉ tái lập GẦN ĐÚNG được**: phương pháp và trọng số công khai, nhưng tác giả **không công bố cấu hình** đã dùng: SALAD, AnyLoc (global), AnyLoc (unstructured), MixVPR, CosPlace, NetVLAD, DINOv3-CLS (trọng số bị khóa, phải xin quyền), DenseVLAD (không có bản Python chính thức hay vocabulary sẵn). Trong số này **chỉ SALAD, AnyLoc-global và DenseVLAD có số liệu trong bài** để đối chiếu. Năm cấu hình còn lại không có số nào được công bố.
4. **KHÔNG công bố, không thể tái lập chính xác**: code dựng benchmark từ dữ liệu SLAM gốc (chọn chuỗi, lấy mẫu con, làm sạch), pipeline sinh GT bằng tổ hợp 5 bộ ước lượng tư thế tương đối (RPE), các GT thay thế ở Hình 5, kết quả chi tiết của 5 cấu hình không có trong bảng, website benchmark, và Hình 1 trái (Oxford RobotCar). Tác giả ghi rõ những phần này *"sẽ công bố khi bài báo được xuất bản"*.
5. **Tài nguyên Kaggle đã xác minh**: 10 tài khoản đều hoạt động (8 tài khoản L4, 2 tài khoản T4). L4 có được qua cuộc thi `gemma-4-developer-agent` (id 149921, mọi tài khoản đã tham gia). Kaggle API truy cập được trực tiếp từ phiên làm việc này, nên Bước 2 có thể tự động hoàn toàn.

**Bảng tổng hợp mức tái lập**

| Thành phần | Công bố? | Tái lập |
|---|---|---|
| Dữ liệu 16 cặp DB/Q (ảnh, pose) | ✅ release v0.1.0 | ✅ **Chính xác** (xem §5) |
| GT `VPR-GT-RPE-0` (hard/soft) | ✅ trong release | ✅ **Chính xác** (dùng lại; không sinh lại được) |
| Code đánh giá (similarity, AUPRC, R@100p, R@K) | ✅ `evaluate.py`, `metrics.py` | ✅ **Chính xác** |
| MegaLoc, BoQ | ✅ cấu hình trong `methods.py` | ✅ **Chính xác** |
| SALAD, AnyLoc-global, DenseVLAD | ⚠️ chỉ có số liệu, không có cấu hình | 🟡 Gần đúng, **có số để đối chiếu** |
| AnyLoc-unstructured, MixVPR, CosPlace, NetVLAD, DINOv3 | ⚠️ không có cấu hình, không có số | 🟡 Gần đúng, **không có số để đối chiếu** |
| Pipeline sinh GT (DUSt3R, MUSt3R, MapAnything, MADPose, SP+LG, Depth Anything V2) | ❌ | ❌ chỉ có thể làm xấp xỉ, rất tốn kém |
| Code dựng benchmark từ dữ liệu gốc, nhãn làm sạch | ❌ | ❌ |
| Kết quả chi tiết, website benchmark | ❌ | ❌ |

---

## 1. Nội dung bài báo và các con số cần tái lập

### 1.1 16 "datasets" = 16 cặp chuỗi database–query

Mười cặp kiểm tra **thay đổi góc nhìn (VC)** và sáu cặp kiểm tra **thay đổi chiếu sáng (IC)**. ID là tên trong `dataset.py` của tác giả, nhãn là tên trong Bảng II/IV. Số ảnh lấy từ Bảng IV.

| # | ID (`dataset.py`) | Nhãn bài báo | #DB | #Q |
|---|---|---|---|---|
| 1 | `MadMax-A0--MadMax-A2` | MM A0 – A2 | 171 | 180 |
| 2 | `MadMax-A2--MadMax-A4` | MM A2 – A4 | 180 | 412 |
| 3 | `MadMax-C0--MadMax-C1` | MM C0 – C1 | 502 | 463 |
| 4 | `MadMax-E0--MadMax-E1` | MM E0 – E1 | 517 | 507 |
| 5 | `MadMax-F0--MadMax-F3` | MM F0 – F3 | 117 | 229 |
| 6 | `MadMax-G0--MadMax-G2` | MM G0 – G2 | 258 | 290 |
| 7 | `S3LI-Etna-crater--S3LI-Etna-crater_inout` | S3E crater – crater inout | 1129 | 1683 |
| 8 | `S3LI-Etna-traverse_1--S3LI-Etna-loops` | S3E traverse 1 – loops | 485 | 780 |
| 9 | `LunarLoc-9--LunarLoc-10` | LL 9 – 10 | 119 | 183 |
| 10 | `LunarLoc-15--LunarLoc-17` | LL 15 – 17 | 118 | 54 |
| 11 | `SPICE-A-dawn-off-spad@5.0us--SPICE-A-night-on-spad@5.0us` | HL³ SPAD dawn – night | 98 | 97 |
| 12 | `SPICE-A-dawn-off-spad@5.0us--SPICE-A-noon-off-spad@5.0us` | HL³ SPAD dawn – noon | 98 | 97 |
| 13 | `SPICE-A-noon-off-spad@5.0us--SPICE-A-night-on-spad@5.0us` | HL³ SPAD noon – night | 97 | 97 |
| 14 | `SPICE-A-dawn-off@2500.0us--SPICE-A-night-on@2500.0us` | HL³ FLIR dawn – night | 99 | 97 |
| 15 | `SPICE-A-dawn-off@2500.0us--SPICE-A-noon-off@2500.0us` | HL³ FLIR dawn – noon | 99 | 99 |
| 16 | `SPICE-A-noon-off@2500.0us--SPICE-A-night-on@2500.0us` | HL³ FLIR noon – night | 99 | 97 |

Có **25 chuỗi (trajectory) khác nhau**: 11 MADMAX, 4 S3LI, 4 LunarLoc, 6 SPICE. Vì nhiều cặp dùng chung chuỗi, tổng số ảnh là **8.784** (theo Bảng IV).

### 1.2 Các con số được công bố (mục tiêu đối chiếu)

Toàn bộ giá trị đã được chép vào file máy đọc được [`vprspace_repro/paper_reference.json`](../vprspace_repro/paper_reference.json). Tôi đã tự kiểm tra lại: trung bình các cột tính lại khớp dòng "Avg. mean" của Bảng II (MeLo 7,51/16 = 0,47; BoQ 0,47; AnyL 0,40; DenV 0,35).

| Mục | Nội dung | Phương pháp có số |
|---|---|---|
| Bảng II | AUPRC cho 16 cặp, cùng Worst case, Avg. mean và Avg. std | MeLo, BoQ, SLD, AnyL(global), DenV |
| Bảng III | R@1/5/10 và R@100p, **chỉ ở dạng tổng hợp** (worst case, mean±std). Không có số theo từng cặp | 5 phương pháp như trên |
| Bảng IV | #ảnh, % query có match, % cặp dương | (dữ liệu) |
| Bảng V | Siêu tham số của GT (d_max, τ_t, τ_R, n) | (GT) |
| Hình 6 | 6 giá trị similarity của BoQ, 3 chữ số thập phân (ví dụ MM A0–A2, query 56: đúng 0,246 / sai 0,296) | BoQ |
| Hình 1 phải, Hình 5 | Histogram similarity (BoQ), PR-curve (MegaLoc, MM A0–A2) | chỉ đối chiếu định tính |

**Các điểm mơ hồ trong bài báo cần lưu ý:**

- Bài nói "nine" phương pháp nhưng liệt kê **10 cấu hình** (AnyLoc có 2 vocabulary).
- **NetVLAD** được nêu tên là "baseline", nhưng không có trong bảng và cũng **không nằm trong danh sách "loại khỏi bảng"**. Mọi nơi khác trong bài đều không nhắc lại.
- "Worst case" ở Bảng III không được định nghĩa: có thể là min theo từng metric, hoặc giá trị của cặp khó nhất. Bước 2 sẽ thử cả hai.
- Không rõ std ở "Avg. std" dùng ddof=0 hay ddof=1. Với MeLo, cả hai cách đều làm tròn ra 0,20.
- Chỉ số query trong Hình 6 không rõ đếm từ 0 hay từ 1.

---

## 2. Những gì tác giả đã công bố (đã kiểm tra trực tiếp)

### 2.1 Repo GitHub `rfechner/VPR-Space`

- **10 commit** từ 14/10 đến 27/10/2026. Gồm 4 file Python nhỏ (`dataset.py`, `evaluate.py`, `methods.py`, `metrics.py`), `requirements.txt`, ảnh `resources/bench_overview.png` và `LICENSE` (MIT).
- `methods.py` **chỉ chứa 2 phương pháp**:
  - `boq`: `torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name="dinov2", output_dim=12288)`, ảnh **322×322**, batch 48.
  - `megaloc`: `torch.hub.load("gmberton/MegaLoc", "get_trained_model")`, ảnh **518×518**, batch 24.
  - Tiền xử lý chung: `ToTensor → Resize(size, BICUBIC, antialias=True) → Normalize(ImageNet)`. Ảnh bị resize thẳng về hình vuông (không giữ tỉ lệ), resize chạy trên CPU trong `DataLoader` (8 worker), không dùng mixed precision.
- `evaluate.py`:
  - Similarity: chuẩn hóa từng chiều bằng mean/std của **database** (std +1e-8), chuẩn hóa L2, rồi `S = db @ query.T`.
  - Descriptor được cache theo tên trajectory, nên mỗi chuỗi chỉ trích xuất một lần.
- `metrics.py`, chuyển thể từ VPR Tutorial của S. Schubert. Lưu ý: phần primitive của metric mang giấy phép **GPL-3.0+**, trong khi repo là MIT.
  - AUPRC dùng **100 ngưỡng** (`linspace(S.max, S.min)`). Các ô chỉ thuộc soft-GT được gán `S.min()`. Tích phân bằng `np.trapezoid` (cần numpy ≥ 2.0).
  - R@100p = recall lớn nhất mà precision = 1.
  - R@K dùng GT = **hard ∪ soft** và chỉ tính các query có ít nhất 1 match.
- `requirements.txt` là bản `pip freeze` của 64 gói: `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, `numpy==2.4.4`, `pillow==12.2.0`, `huggingface_hub==1.23.0`, `safetensors==0.8.0`… Các gói `numpy/pandas/scipy/matplotlib` ghim ở phiên bản yêu cầu **Python ≥ 3.11**.
  - **Thiếu** `pytorch_lightning` (SALAD cần), `fast_pytorch_kmeans` (AnyLoc cần), cùng `timm/transformers`, `gdown`, `faiss`, `xformers`. Như vậy môi trường được công bố **chỉ phục vụ BoQ và MegaLoc**, không phải môi trường đã chạy 7 phương pháp còn lại.
  - Có `rosbags`, `mcap`, `roma`, `opencv-python`: dấu vết của code dựng benchmark và tính sai số tư thế. **Code đó không có trong repo.**
- Tag `v0.1.0` trỏ vào commit `da8d599` (14/10). Phiên bản `dataset.py` lúc đó đọc file `setups.json`. Commit `f7fa1f6` (15/10) chuyển sang danh sách `SETUPS` cố định trong code. Hai cách cho cùng 16 cặp.
- **Issues/PR**: chỉ có 1 issue (#1, 27/10/2026, Hugging Face đề nghị đưa dữ liệu lên HF Hub, chưa ai trả lời) và 0 PR. Không có thông tin kỹ thuật nào bổ sung.
- **Website benchmark**: chưa tồn tại. Tìm kiếm không ra kết quả, và bài báo ghi rõ "will be made available … upon paper publication".

### 2.2 Release dữ liệu `v0.1.0` (pre-release)

- Ngày 15/10/2026, ghi chú phát hành: *"VPR-Space 0.1.0 (pre-release)"*.
- **11 phần** `vpr-space.tar.0000 … .0010`: 10 phần × 1,86 GB và 1 phần 349 MB, tổng ≈ 18,9 GB. Ghép lại bằng `cat vpr-space.tar.* | tar xvf -`.
- Trang release có công bố **SHA-256 cho từng phần**. Tôi đã tải lại cả 11 phần và đối chiếu (§5).

### 2.3 Dữ liệu gốc mà benchmark được dựng từ đó (chỉ cần nếu muốn dựng lại benchmark)

| Bộ dữ liệu | Nơi tải | Giấy phép (theo nguồn tìm được) |
|---|---|---|
| MADMAX (Sao Hỏa – Ma-rốc) | DLR: [rmc.dlr.de/morocco2018](https://rmc.dlr.de/morocco2018) | Không tìm thấy điều khoản rõ ràng |
| S3LI Etna (Mặt Trăng – Etna) | DLR: [datasets.arches-projekt.de/s3li_dataset](https://datasets.arches-projekt.de/s3li_dataset/) | Không tìm thấy (toolkit là MIT) |
| LunarLoc (mô phỏng) | [github.com/mit-acl/lunarloc-data](https://github.com/mit-acl/lunarloc-data) | Chưa xác minh |
| SPICE-HL³ (LunaLab) | Zenodo [10.5281/zenodo.13970078](https://zenodo.org/records/13970078) (~68,9 GB) | CC BY 4.0 |

Ngay cả khi tải hết dữ liệu gốc, vẫn **không dựng lại chính xác** được benchmark: thiếu code chọn và lấy mẫu con, thiếu nhãn làm sạch của 3 người gán nhãn (≈ 9.200 điểm bị đánh dấu), và thiếu pipeline sinh GT.

---

## 3. Kiểm kê chi tiết theo từng thành phần

| Thành phần | Công bố? | Nguồn / phiên bản ghim | Tái lập chính xác? | Ghi chú, bằng chứng |
|---|---|---|---|---|
| Ảnh 30 trajectory (đã lấy mẫu con, đã làm sạch) | ✅ | Release v0.1.0, `images/<traj>/*.png` | ✅ | Kiểm chứng SHA-256 và số ảnh ở §5 |
| Pose camera | ✅ | `poses/<traj>.npy` | ✅ | Không cần cho đánh giá, chỉ dùng để kiểm tra |
| GT `VPR-GT-RPE-0` (hard, soft) | ✅ | `vpr_gt/VPR-GT-RPE-0/<setup>.npz` | ✅ (dùng lại) | Thống kê khớp Bảng IV (§5) |
| GT "pose thresholding" / "frustum overlap" (Hình 5) | ❌ | – | ❌ | Bài không cho ngưỡng cụ thể |
| Nhãn làm sạch dữ liệu ("found errors are provided as part of the published data") | ⚠️ | Xem §5 | – | Bài nói có kèm theo; §5 kiểm tra thực tế |
| Code dựng benchmark (chọn chuỗi, lấy mẫu con 1 m / 10°, trích ảnh từ rosbag) | ❌ | – | ❌ | Bài chỉ mô tả bằng lời (Phụ lục) |
| Pipeline sinh GT (5 RPE, Depth Anything V2, RANSAC, bỏ phiếu đa số) | ❌ | – | ❌ | Bảng V chỉ cho ngưỡng. Thiếu checkpoint, độ phân giải, tham số RANSAC, xử lý intrinsics, danh sách cặp được đánh giá |
| Code đánh giá và metric | ✅ | `evaluate.py`, `metrics.py` @ `eae17bf` | ✅ | Dùng nguyên văn |
| MegaLoc | ✅ | Code `gmberton/MegaLoc@5fe0dd6`; trọng số HF `gberton/MegaLoc` `model.safetensors` (914.577.436 B, SHA-256 `d4f9f2bc…986ffb4f8`) | ✅ | **Đã kiểm chứng bằng cách tải về so sánh: 189/189 tensor giống hệt từng bit** với bản HF tháng 02/2025 (\|Δ\| = 0). Khác biệt duy nhất là bỏ `mask_token` toàn số 0 (không dùng khi suy luận) và đổi tên khóa (§6) |
| BoQ (DINOv2, 12288-d) | ✅ | Code `amaralibey/bag-of-queries@1a4965e` (không đổi kể từ 16/03/2025); trọng số `releases/v1.0/dinov2_12288.pth` (363 MB, upload 04/07/2024, chưa từng bị thay); DINOv2 `facebookresearch/dinov2@7764ea0` | ✅ | `load_state_dict(strict=True)` ghi đè toàn bộ trọng số backbone |
| SALAD | ⚠️ | `serizba/salad@6aede13` + `dino_salad.ckpt` (336 MB, upload 30/11/2023) | 🟡 | Không rõ độ phân giải (322×322 theo bài SALAD, hay resize về bội số của 14) |
| AnyLoc global / unstructured | ⚠️ | `AnyLoc/DINO@eea7d4c`: ViT-G/14, layer 31, facet value, 32 cụm; vocabulary `global` và `unstructured` trong release v1 (upload 28/10/2023) | 🟡 | Không rõ độ phân giải và tiền xử lý. Cần ViT-G/14 (~4,4 GB) |
| MixVPR | ⚠️ | `amaralibey/MixVPR@4043915`, trọng số trên Google Drive (4096/512/128-d, 320×320) | 🟡 | Không rõ biến thể |
| CosPlace | ⚠️ | `gmberton/CosPlace@52b56e9`, release v1.0 (nhiều backbone và số chiều) | 🟡 | Không rõ backbone, số chiều, độ phân giải |
| NetVLAD | ⚠️ | VGG16 Pitts30K hoặc TokyoTM (MATLAB → PyTorch, của hloc) | 🟡 | Không rõ cách cài đặt và PCA |
| DenseVLAD | ⚠️ | Chỉ có code MATLAB gốc (Torii và cs., 2015). Không có bản Python chính thức kèm vocabulary | 🟡/❌ | Khó khớp số |
| DINOv3 (CLS) | 🔒 | `facebookresearch/dinov3@6876159`; trọng số qua form của Meta hoặc HF gated (DINOv3 License) | 🟡 | Không rõ biến thể và độ phân giải. **Không được phát tán công khai** |
| Môi trường | ✅ một phần | `requirements.txt` | ✅ cho BoQ, MegaLoc | Thiếu gói của các phương pháp khác. GPU tác giả dùng không rõ |
| Kết quả chi tiết của AnyLoc-unstr., MixVPR, CosPlace, DINOv3, NetVLAD | ❌ | "will be made available on the benchmark website" | ❌ | Chỉ biết là "không đứng đầu ở cặp nào" |

---

## 4. Chi tiết 9 phương pháp (10 cấu hình)

| # | Phương pháp | Số trong bài | Cấu hình tác giả | Cấu hình "chuẩn" tôi đề xuất cho Bước 2 (khi tác giả không công bố) |
|---|---|---|---|---|
| 1 | **MegaLoc** | Bảng II, III, Hình 5 | **Có** (`methods.py`) | Giữ nguyên: 518×518, fp32 |
| 2 | **BoQ** DINOv2-12288 | Bảng II, III, Hình 1, 6 | **Có** | Giữ nguyên: 322×322, fp32 |
| 3 | SALAD | Bảng II, III | Không | 322×322 (như bài SALAD và quy ước của `methods.py`), có quét thêm 224 và resize theo bội 14 |
| 4 | AnyLoc-global | Bảng II, III | Không | Code hub chính thức, ViT-G/14 L31 value, vocab `global`; quét 224, 322, 518 và "giữ tỉ lệ, cạnh ≤ 1024" như demo AnyLoc |
| 5 | AnyLoc-unstructured | Không | Không | Như trên với vocab `unstructured` |
| 6 | MixVPR | Không | Không | ResNet50, 4096-d, 320×320 (biến thể mạnh nhất, mặc định của README) |
| 7 | CosPlace | Không | Không | ResNet50, 2048-d (mặc định của `hubconf`) |
| 8 | NetVLAD | Không | Không | VGG16 Pitts30K, PCA 4096 (mặc định của `VPR-methods-evaluation`) |
| 9 | DenseVLAD | Bảng II, III | Không | Tự cài lại theo bài gốc (dense RootSIFT 4 tỉ lệ, 128 từ, VLAD, intra-norm, PCA-whitening 4096). Vocabulary học từ ảnh database. Đánh dấu là **"tái cài đặt"** |
| 10 | DINOv3 CLS | Không | Không | ViT-B/16 LVD-1689M, CLS token; quét 224 và 518. **Chỉ chạy nếu bạn xin được quyền tải** |

Bộ [`gmberton/VPR-methods-evaluation`](https://github.com/gmberton/VPR-methods-evaluation) (`9c8dfd4`, 13/08/2026) hỗ trợ sẵn 8/10 cấu hình (trừ DenseVLAD và DINOv3). Đây là chuẩn cộng đồng hợp lý để lấy "cấu hình mặc định". Ghi chú: tác giả cho biết đã dùng Claude Code để sinh code truy cập các phương pháp, nên khả năng cao họ dùng `torch.hub` với tham số mặc định, giống mẫu trong `methods.py`.

---

## 5. Kiểm chứng dữ liệu release (đã chạy thực tế trên Kaggle)

_(Phần này được điền từ kết quả notebook CPU `vprspace-release-inspect`. Notebook tải trực tiếp cả 11 phần từ GitHub, tính SHA-256, liệt kê toàn bộ file, đọc header PNG và tính lại Bảng IV từ ma trận GT.)_

<!-- SECTION5_PLACEHOLDER -->

---

## 6. Các điểm tinh tế ảnh hưởng tới độ "chính xác tuyệt đối"

1. **`torch.hub` không ghim phiên bản.** Code của tác giả tải nhánh `main` tại thời điểm chạy lần đầu (sau đó dùng cache). Tôi đã kiểm tra lịch sử các repo phía trên:
   - **MegaLoc** có thay đổi ngày 27/07/2026 ("make model faster and lighter"): phép attention viết tay được thay bằng `F.scaled_dot_product_attention`, và `repeat/sum` được thay bằng `einsum`. Hai cách **tương đương toán học**, chỉ khác sai số làm tròn float (~1e-6).
   - **File trọng số MegaLoc trên HF không đổi** (cùng SHA-256 `d4f9f2bc…ffb4f8`) ở cả 5 revision kể từ 21/01/2026. File cũ hơn (02/2025, SHA-256 `7f843577…`) chỉ khác ở tên khóa và tensor `mask_token` toàn số 0. **Toàn bộ 189 tensor có giá trị giống hệt từng bit** (tôi đã tải cả hai bản về so sánh). Vì vậy dù tác giả tải trọng số lúc nào, trọng số cũng giống nhau.
   - **BoQ**: code không đổi kể từ 16/03/2025, file trọng số không đổi kể từ 04/07/2024. Các file trọng số của SALAD, AnyLoc và CosPlace cũng chưa từng bị thay kể từ 2023. Trang GitHub không hiển thị SHA-256 cho các file này, nên Bước 2 sẽ tự tính và ghim lại.
   - **DINOv2** hiện dùng SDPA; khác biệt với các bản cũ cũng chỉ ở mức làm tròn.
   - Trong Bước 2 tôi sẽ ghim commit và SHA-256 của mọi thứ. Tùy chọn: chạy thêm bản MegaLoc cũ để chứng minh kết quả không đổi.
2. **Số học GPU**:
   - Code tác giả để mặc định của PyTorch: TF32 bật cho cuDNN (conv), tắt cho matmul. L4 (Ada) hỗ trợ TF32 giống GPU Ampere/Ada mà tác giả nhiều khả năng đã dùng. T4 không có TF32.
   - Mixed precision (fp16/bf16) **sẽ làm lệch số**. Vì vậy các lần chạy dùng để đối chiếu phải giữ **fp32**.
   - Batch size không đổi ngữ nghĩa (chế độ eval), nhưng có thể đổi thuật toán GEMM, kéo theo sai khác cỡ 1e-6. Tôi sẽ kiểm chứng bằng thực nghiệm (§8.3).
3. **Resize trên CPU** bằng `torchvision 0.21` (bicubic + antialias trên tensor float). Resize trên GPU sẽ khác ở mức làm tròn, nên cần giữ resize trên CPU cho các phương pháp phải khớp chính xác.
4. **Đọc ảnh**:
   - Ảnh được đọc bằng `PIL ... .convert("RGB")`. Nếu PNG là 16-bit, PIL sẽ cắt giá trị khi chuyển sang RGB. Phải giữ nguyên file gốc, không nén lại.
   - Thứ tự ảnh là `sorted(glob("*.png"))` và phải khớp thứ tự hàng/cột của ma trận GT.
5. **Phiên bản thư viện**: `np.trapezoid` cần numpy ≥ 2.0; các gói ghim cần Python ≥ 3.11. Bước 2 sẽ cài đúng `torch 2.6.0+cu124` và các gói ghim dạng offline (wheel đóng gói sẵn), vì phiên L4 chạy **không có Internet**.

---

## 7. Tài nguyên Kaggle (đã xác minh lúc ~11:35 UTC, 06/11/2026)

Token được lưu riêng ngoài repo (`/root/.kaggle/accounts.json`, quyền 600) và **không** xuất hiện trong code, báo cáo hay commit. Đã kiểm tra kết nối bằng `kaggle competitions list` (CLI 2.2.5) cho cả 10 tài khoản.

| Username | GPU | Quota GPU còn lại | Ghi chú |
|---|---|---|---|
| nguyenmanh0404 | L4 | **29,92 h** | |
| hieuho669 ("Hiếu") | L4 | **22,06 h** | |
| duckzuykobip ("DuckZuy") | L4 | 10,80 h | |
| dinhthinh ("Dinh Thinh") | L4 | 9,98 h | |
| namhunhh | L4 | 9,63 h | |
| quynhtran35 | L4 | 0,00 h (đã dùng 34,21/30) | Chỉ còn dùng được CPU |
| yuan1103 | L4 | 0,00 h (43,23/30) | Chỉ còn dùng được CPU |
| phatle1578 | L4 | 0,00 h (44,70/30) | Chỉ còn dùng được CPU |
| mxuanquy | T4 | **26,44 h** | |
| yeshenhi | T4 | **26,53 h** | |

- Tổng quota còn lại: **≈ 135 GPU-giờ**. Quota của mọi tài khoản **được làm mới vào 2026-11-07 00:00 UTC** (mỗi tài khoản về 30 h).
- Cả 10 tài khoản đã tham gia cuộc thi **`gemma-4-developer-agent`** (id **149921**, hạn 02/12/2026). Đây là nguồn của `competition_sources` / `dataSources` trong metadata L4 ở mục 4.1–4.2 của tài liệu vận hành.
- Lúc kiểm tra, không tài khoản nào có phiên đang chạy.
- Tài liệu vận hành nhắc tới tài khoản **`nhatlong1103`** làm nơi chứa Kaggle Dataset công khai, nhưng **chưa có token của tài khoản này** (xem quyết định D4).
- Đã có sẵn công cụ tự động:
  - `kg <tài_khoản> …` gọi CLI với token tương ứng, không in token ra.
  - `kwatch` theo dõi phiên và tự báo khi xong. Nếu phiên **chưa chạy sau 8 phút (L4, CPU) hoặc 10 phút (T4)** thì coi là kẹt hàng đợi, sẽ hủy và tạo lại tối đa 4 lần, đúng như Mục 2 của tài liệu vận hành. Có thêm giới hạn thời gian chạy tối đa để phát hiện treo.
  - Lưu ý: trong thử nghiệm, Kaggle API không trả log khi phiên đang chạy (kể cả `logs -f`), nên tiêu chí "không có đầu ra" được áp dụng theo trạng thái hàng đợi.

<!-- KAGGLE_ENV_PLACEHOLDER -->

---

## 8. Kế hoạch đề xuất cho Bước 2 (bám sát tài liệu vận hành)

### 8.1 Phạm vi và tiêu chí "đạt"

| Nhóm | Phương pháp | Tiêu chí đạt |
|---|---|---|
| **A – chính xác** | MegaLoc, BoQ | 32 giá trị AUPRC ở Bảng II và các dòng Bảng III **trùng khớp sau khi làm tròn 2 chữ số**. 6 similarity ở Hình 6 trùng khớp 3 chữ số. Bảng IV trùng khớp hoàn toàn |
| **B – gần đúng, có đối chiếu** | SALAD, AnyLoc-global, DenseVLAD | Báo cáo độ lệch \|Δ\| theo từng cặp và trung bình; mục tiêu \|Δ\| ≤ 0,02 cho SALAD và AnyLoc. DenseVLAD chỉ so sánh xu hướng |
| **C – gần đúng, không đối chiếu** | AnyLoc-unstr., MixVPR, CosPlace, NetVLAD, DINOv3 | Báo cáo số. Kiểm tra ràng buộc của bài: "không đứng đầu ở cặp nào" |

### 8.2 Luồng chạy và phân bổ tài khoản

Khối lượng tính toán thực tế **nhỏ**: khoảng 8.784 ảnh × 10 cấu hình. Ước lượng phần GPU chỉ cỡ **vài chục phút** trên 1 phiên 4×L4. Phần nặng nhất là AnyLoc ViT-G/14 ở độ phân giải cao, cùng DenseVLAD trên CPU. Vì vậy, theo đúng tinh thần của tài liệu (*"khai thác tối đa 2 phiên GPU + 5 phiên CPU trên mỗi tài khoản, thay vì rải mỗi tài khoản 1 phiên"*), tôi đề xuất **tập trung vào ít tài khoản** và giữ các tài khoản còn lại làm dự phòng:

| Pha | Việc | Tài khoản, loại phiên |
|---|---|---|
| P0 | Lưu token, kiểm tra kết nối, quota và cuộc thi | ✅ **đã xong** |
| P1 | Kiểm chứng release (SHA-256, manifest, Bảng IV) | 🔄 **đang chạy** (quynhtran35, CPU; kết quả ở §5) |
| P2 | Đóng gói offline: (a) dữ liệu đóng thành vài archive theo họ dataset, để mỗi output < 20 GB và ít file (Kaggle giới hạn `/kaggle/working` 20 GB, và có thể giới hạn khoảng 500 file output); (b) wheel đúng phiên bản ghim, repo hub theo commit đã ghim, toàn bộ trọng số kèm SHA-256 | CPU + Internet (≤ 5 phiên song song) |
| P3 | Viết code, rà soát, chạy thử khô (dry-run) trên CPU với 2 cặp × vài ảnh; kiểm thử `metrics.py` | Chạy cục bộ và CPU Kaggle |
| P4-G1 | **Nhóm A** (cấu hình đúng như tác giả) + Nhóm B/C với cấu hình chuẩn | **nguyenmanh0404** – phiên L4×4 thứ 1 |
| P4-G2 | Quét cấu hình cho Nhóm B (SALAD, AnyLoc) để đo độ nhạy | **nguyenmanh0404** – phiên L4×4 thứ 2 |
| P4-G3 | Bản sao Nhóm A trên máy khác để kiểm tra tính tất định | **hieuho669** – L4×4 |
| P4-T | Bản sao Nhóm A trên T4 (không có TF32) để đo độ nhạy phần cứng | **mxuanquy** – T4×2 |
| P4-C | DenseVLAD (CPU, chia shard theo trajectory) + tính metric | quynhtran35, yuan1103, phatle1578 (hết GPU, còn CPU), ≤ 5 phiên mỗi tài khoản |
| Dự phòng | Chạy lại khi kẹt, lỗi hoặc hết quota | duckzuykobip, dinhthinh, namhunhh, yeshenhi |
| P5 | Tổng hợp, so sánh với `paper_reference.json`, viết báo cáo kết quả | Phiên làm việc này |
| P6 | Dọn nhánh, viết 3 PDF bàn giao (Scientific Findings, Engineering & Ops Lessons, Handoff), `environment_summary.pdf`, README đọc repo | Mục 9 của tài liệu vận hành |

Phiên L4 dùng đúng metadata ở mục 4.1–4.2: `"machine_shape": "NvidiaL4"`, `"competition_sources": ["gemma-4-developer-agent"]`, **Internet tắt**. Dữ liệu, wheel và trọng số được gắn vào qua `kernel_sources` hoặc `dataset_sources` (xem D4).

### 8.3 Tối ưu 4×L4 / 96 GB VRAM mà không làm lệch kết quả

- **Song song theo công việc, không dùng DDP.** DDP chỉ có ý nghĩa khi huấn luyện; ở đây chỉ có suy luận. Mỗi GPU chạy 1 tiến trình worker. Đơn vị công việc là (phương pháp, trajectory). Phân công theo thuật toán LPT dựa trên chi phí ước lượng (số ảnh × FLOPs), sao cho 4 GPU xong gần như cùng lúc.
- **Batch lớn nhất có thể**: tự dò bằng cách bắt lỗi OOM rồi giảm dần, nhắm tới ~90% VRAM mỗi GPU. Có thể nạp đồng thời nhiều mô hình trên một GPU. Với Nhóm A, sẽ chạy **cả batch của tác giả (48/24) lẫn batch tối đa**, rồi chứng minh descriptor lệch ≤ 1e-5 và metric trùng nhau. Chỉ sau đó mới dùng batch tối đa cho mọi lần chạy khác.
- **Giải mã PNG một lần cho nhiều phương pháp**: worker `DataLoader` đọc ảnh một lần, tạo mọi độ phân giải cần thiết bằng đúng phép biến đổi CPU của tác giả, dùng `pin_memory` và prefetch, với số worker tối đa theo số nhân CPU.
- **Giữ fp32** cho mọi lần chạy dùng để đối chiếu. Tùy chọn chạy thêm một lần AMP chỉ để đo tốc độ; kết quả AMP không dùng để so sánh.
- **Lưu cache descriptor** (`.npy` fp32) để tính lại metric trên CPU mà không cần GPU, và để các tài khoản dùng chung, không phải chạy trùng.

### 8.4 Kiểm tra trước khi chạy (bắt buộc, theo Mục 6)

1. Kiểm thử đơn vị cho `metrics.py`: so với kết quả tính tay trên ma trận nhỏ.
2. Kiểm tra dữ liệu: SHA-256 từng phần và từng file; số ảnh và thống kê GT so với Bảng IV (đã làm ở §5).
3. Chạy thử trên CPU (2 cặp × 5 ảnh) để kiểm tra I/O, thứ tự file và hình dạng ma trận.
4. Rà soát code lần cuối: công thức, cấu hình, khả năng chạy offline.
5. Sau đó mới chạy trên L4.

### 8.5 Giám sát

- Mỗi phiên có một `kwatch` chạy nền, tự báo khi xong hoặc lỗi, nên không cần kiểm tra thủ công.
- Kẹt hàng đợi (8/10 phút): hủy và tạo lại, tối đa 4 lần. Nếu vẫn lỗi thì xem log, quota và cấu hình.
- Nếu phiên treo mà không tự hủy được, tôi sẽ báo bạn để hủy thủ công.

---

## 9. Các quyết định cần bạn chốt trước Bước 2

| # | Câu hỏi | Đề xuất của tôi |
|---|---|---|
| **D1** | Phạm vi phương pháp: (a) chỉ Nhóm A; (b) A + B; (c) cả 10 cấu hình | **(c)**. Nhóm A là phần "tái lập chính xác". B và C được báo cáo rõ là "tái cài đặt gần đúng" |
| **D2** | Với nhóm B: chỉ dùng cấu hình chuẩn đã định trước, hay cho phép quét cấu hình để tìm cái gần với bài báo nhất (báo cáo minh bạch cả hai)? | Cấu hình chuẩn làm kết quả chính; quét cấu hình là phân tích phụ |
| **D3** | DINOv3: bạn có muốn xin quyền tải trọng số (HF gated hoặc form của Meta) không? Nếu không, bỏ cấu hình này | Bỏ, trừ khi bạn cần đủ 10/10 |
| **D4** | Kaggle Dataset **công khai** (tài liệu yêu cầu dùng `nhatlong1103`, nhưng chưa có token) hay **riêng tư** cho từng tài khoản? Lưu ý: dữ liệu phái sinh từ MADMAX và S3LI chưa rõ giấy phép phân phối lại | **Riêng tư** cho dữ liệu (mỗi tài khoản GPU tự chạy notebook CPU chuẩn bị dữ liệu, ~10 phút, không tốn quota GPU). **Công khai** cho wheel và trọng số có giấy phép mở (nếu bạn gửi token `nhatlong1103`) |
| **D5** | Môi trường: cài đúng gói ghim của tác giả (`torch 2.6.0+cu124`…, offline) hay dùng môi trường mặc định của Kaggle? | **Gói ghim** của tác giả, ghi lại cả phiên bản môi trường Kaggle |
| **D6** | Có cố sinh lại GT bằng tổ hợp RPE (xấp xỉ, rất tốn GPU, không thể khớp chính xác) không? | **Không**. Dùng GT đã phát hành |
| **D7** | Bắt đầu Bước 2 ngay (dùng ~82 h quota L4 còn lại) hay đợi quota làm mới lúc 00:00 UTC 07/11? | Bắt đầu ngay. Khối lượng nhỏ, quota hiện có là đủ |

---

## 10. Nguồn

- Bài báo: <https://arxiv.org/abs/2610.35642> (bản v1, toàn văn đã đọc qua alphaXiv).
- Repo và release: <https://github.com/rfechner/VPR-Space>, <https://github.com/rfechner/VPR-Space/releases/tag/v0.1.0>, issue #1 <https://github.com/rfechner/VPR-Space/issues/1>.
- Phương pháp: [MegaLoc (GitHub)](https://github.com/gmberton/MegaLoc), [MegaLoc (HF)](https://huggingface.co/gberton/MegaLoc), [Bag-of-Queries](https://github.com/amaralibey/Bag-of-Queries), [SALAD](https://github.com/serizba/salad), [AnyLoc](https://github.com/AnyLoc/DINO), [MixVPR](https://github.com/amaralibey/MixVPR), [CosPlace](https://github.com/gmberton/CosPlace), [DINOv2](https://github.com/facebookresearch/dinov2), [DINOv3](https://github.com/facebookresearch/dinov3), [DINOv3 trên HF (gated)](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation).
- Dữ liệu gốc: [MADMAX (DLR)](https://www.dlr.de/en/rm/research/publications-and-downloads/datasets/marokko-navigations-tests), [S3LI (DLR)](https://www.dlr.de/en/rm/research/publications-and-downloads/datasets/s3li-dataset), [LunarLoc (arXiv 2506.16940)](https://arxiv.org/abs/2506.16940), [SPICE-HL³ (Zenodo)](https://zenodo.org/records/13970078).
