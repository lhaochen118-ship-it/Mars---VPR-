# Kiểm chứng lại: phần không được công bố, phương án thay thế và độ tin cậy của nguồn

*Ngày 06/11/2026. Trả lời ba câu hỏi:*

1. *Những thứ tôi liệt kê có thật sự không được cung cấp không?*
2. *Phương án thay thế đã chính xác và hợp lý nhất chưa?*
3. *Nguồn tải code và trọng số có uy tín, chính xác không?*

## Phạm vi tìm kiếm lại

Tôi đã rà soát lại ở **tất cả** các nơi sau:

- Repo `rfechner/VPR-Space`: toàn bộ 10 commit, tag `v0.1.0`, issues và PR.
- **Toàn bộ 8.961 file** trong release v0.1.0, kể cả `README.md` và thư mục `misc/` (thư mục này không được nhắc trong README của repo).
- Toàn văn bài báo, kể cả phụ lục.
- Trang arXiv: chỉ có bản v1, không có file đính kèm.
- **9 repo công khai khác của tác giả** (`mini-vpr`, `on-the-fly-vpr`, `devon-island-vpr-challenge`, `hdc-delf-reproduction`…).
- Tìm kiếm web và Hugging Face.
- Repo chính thức của từng phương pháp.

---

## 1. Có thật sự không được cung cấp không?

| Mục | Kết luận sau khi kiểm tra lại | Bằng chứng |
|---|---|---|
| Cấu hình của 7 phương pháp (SALAD, AnyLoc ×2, MixVPR, CosPlace, NetVLAD, DINOv3, DenseVLAD) | **Đúng là không có** | `methods.py` chỉ có `boq` và `megaloc`. Repo `mini-vpr` (09/2026) của tác giả cũng **chỉ có BoQ/MegaLoc**, với cấu hình khác hẳn (MegaLoc 322 thay vì 518, resize PIL trước `ToTensor`). Các repo còn lại không liên quan |
| Môi trường cho 7 phương pháp | **Đúng là không có** | `requirements.txt` thiếu `pytorch_lightning` và `pytorch_metric_learning` (SALAD cần ngay khi khởi tạo model), `fast_pytorch_kmeans` (AnyLoc), `timm`/`transformers` (DINOv3), `kornia`/`vlfeat` (DenseVLAD) |
| Implementation của DenseVLAD | **Đúng là không có** | Cả 3 địa chỉ trang dự án gốc của Torii đều trả **HTTP 503** (không truy cập được). Không có bản Python chính thức |
| **Nhãn làm sạch dữ liệu** | ⚠️ **Sửa lại: CÓ được công bố** | `misc/flagged_errors/`: 117 file CSV, kèm README. Lưu ý từ tác giả: một số chuỗi chỉ rà soát một phần; LunarLoc không được gán nhãn; lỗi trục z ở S3LI được sửa chứ không bỏ frame |
| **Pipeline sinh GT** | ⚠️ **Sửa lại: CÓ THỂ tái lập một phần, và chính xác** | `misc/rpe_estimates/` chứa **kết quả của cả 5 RPE** (DUSt3R, MUSt3R, MapAnything, MADPose, SP+LG/"matchdepth") cho mọi cặp ảnh, cả 2 thứ tự, kèm mặt nạ `evaluated` và `gt_dist_m`. Tôi đã dựng lại bước bỏ phiếu với ngưỡng Bảng V (xem §1.1): **khớp 100%** |
| Code chạy 5 RPE (checkpoint, độ phân giải, RANSAC…) | **Đúng là không có** | Nhưng không cần cho việc đánh giá, vì kết quả RPE đã có sẵn |
| GT thay thế ở Hình 5 (pose thresholding, frustum overlap) | **Đúng là không có** | Release chỉ có `vpr_gt/VPR-GT-RPE-0` |
| Kết quả chi tiết của 5 cấu hình không có trong bảng; website benchmark | **Đúng là không có** | Bài ghi "sẽ công bố khi xuất bản"; chưa có website |

### 1.1 Kết quả dựng lại bước bỏ phiếu của GT

Quy ước tìm được, và là quy ước **duy nhất** cho khớp hoàn toàn trên tất cả 16 cặp:

- Tư thế tương đối GT cho thứ tự (db, query) là `inv(P_query) · P_db`.
- Ước lượng của RPE giữ nguyên.
- Hệ số tỉ lệ s ≥ 0 trong Δt = min_s ‖s·t̂ − t_GT‖.
- Một cặp là dương khi đủ n RPE vượt ngưỡng ở **cả hai** thứ tự, và khoảng cách giữa hai camera ≤ d_max.

| Cặp | Hard: thực / dựng lại | Soft: thực / dựng lại | Sai lệch |
|---|---|---|---|
| MM A0–A2 | 2336 / 2336 | 5797 / 5797 | 0 |
| MM C0–C1 | 3974 / 3974 | 26181 / 26181 | 0 |
| S3E crater – crater inout | 2887 / 2887 | 13128 / 13128 | 0 |
| … (đủ 16 cặp) | | | **0 ở cả 16 cặp, cả hard lẫn soft** |

Chi tiết nằm trong [`results/verification/gt_rederivation.json`](../results/verification/gt_rederivation.json).

**Phát hiện mới, một điểm không nhất quán *bên trong bài báo*:**

- Bảng IV, cột "% of pos. matches", thấp hơn GT phát hành ở 14/16 cặp. Ví dụ A0–A2: bài ghi 8,55%, GT phát hành cho 7,59%.
- Không mẫu số nào giải thích được chênh lệch: đã thử toàn bộ cặp, chỉ các cặp được đánh giá, hard ∪ soft, query có match.
- Không ngưỡng n nào giải thích được: với A0–A2, n=4 cho 7,59% và n=3 cho 10,0%.
- Vì GT phát hành **khớp tuyệt đối** với Bảng V và các RPE estimate, khả năng cao cột này được tính từ một phiên bản GT trước đó.
- Phép thử quyết định là AUPRC của BoQ/MegaLoc so với Bảng II, đang chạy.

---

## 2. Phương án thay thế đã hợp lý nhất chưa?

Tôi đối chiếu lại với **code chính thức** của từng phương pháp:

| Phương pháp | Đối chiếu với nguồn chính thức | Điều chỉnh sau kiểm tra |
|---|---|---|
| **AnyLoc** | `AnyLoc/AnyLoc` `demo/anyloc_vlad_generate.py`: `max_img_size = 1024`, resize **BICUBIC**, `CenterCrop` về bội số 14. **Khớp chính xác** với `anyloc_demo1024` của tôi. Script đánh giá trong bài AnyLoc (`scripts/dino_v2_vlad_global_vocab.py`) chỉ crop về bội số 14 sau bước resize của bộ dataset | Giữ nguyên. Vẫn quét 224/322/518 |
| **SALAD** | README: `eval.py --image_size 322 322`. `eval.py` resize ảnh PIL bằng **BILINEAR** rồi mới `ToTensor` | **Thêm `salad_322_official`**, dùng đúng phép biến đổi của `eval.py`. Bản chính vẫn là pipeline thống nhất của tác giả (bicubic, `Images`). Lý do: chính `methods.py` cho thấy tác giả *không* dùng độ phân giải chính thức của từng phương pháp (MegaLoc chạy 518, trong khi `mini-vpr` của họ dùng 322) |
| **CosPlace** | `parser.py`: `--resize_test_imgs` mặc định **False**, tức test ở độ phân giải gốc. `hubconf` mặc định ResNet50-2048 | Giữ nguyên (native, R50-2048) |
| **MixVPR** | README: *"images must be resized to 320x320"*. 4096-d là model chính | Giữ nguyên |
| **NetVLAD** | Trang chính thức (ENS/Inria) ghi *"The best model (VGG-16 + NetVLAD + whitening, trained on Pittsburgh)"*, tức `vd16_pitts30k_conv5_3_vlad_preL2_intra_white.mat`. File `Pitts30K_struct.mat` của hloc chính là model này, được xuất bằng `netvlad_tf_open/matlab/net_class2struct.m` | Giữ nguyên (4096-d whitened = model tốt nhất của tác giả gốc) |
| **DINOv3** | Model card của timm nêu hai khác biệt so với bản gốc: (a) bỏ bias QKV vốn toàn số 0, không ảnh hưởng gì; (b) **chu kỳ RoPE ở bản gốc lưu bfloat16, timm dùng float32 nên lệch số học nhẹ**. Card đưa sẵn cách sửa để khớp output gốc | **Đã áp dụng cách sửa đó** (cắt chu kỳ RoPE về bfloat16). Biến thể và độ phân giải không rõ nên quét S/B/L × 224/512 |
| **DenseVLAD** | Trang gốc không truy cập được (HTTP 503). Theo bài gốc và quy ước `vl_phow` của VLFeat: bin 4/6/8/10 px, bước 2 px, làm mịn σ = √((bin/6)² − 0,25), RootSIFT, 128 từ, intra-norm, PCA-whitening 4096 | Đổi sang **dense SIFT trên GPU** (kornia), vì SIFT của OpenCV mất **50 s/ảnh** (~120 CPU-giờ). Vocabulary và PCA học **chỉ từ ảnh database**, không dùng query |
| **GT** | Đã chứng minh GT phát hành khớp tuyệt đối với Bảng V | Dùng GT phát hành |
| **Môi trường** | Đã cài thành công offline trong venv sạch trên Kaggle: `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124`, `numpy 2.4.4`, `pillow 12.2.0`, `scipy 1.17.1`, `opencv 5.0.0` (đúng các bản tác giả ghim) | Giữ nguyên |

---

## 3. Nguồn tải có uy tín, chính xác không?

| Thành phần | Nguồn | Chủ sở hữu | Kiểm tra tính toàn vẹn |
|---|---|---|---|
| Dữ liệu VPR-Space | GitHub release `rfechner/VPR-Space` v0.1.0 | Tác giả thứ nhất | **11/11 phần khớp SHA-256** trang release công bố. README của release ghi dữ liệu được tái phát hành "with the consent of the original authors" |
| BoQ (code, trọng số) | `amaralibey/Bag-of-Queries`, release v1.0 | Tác giả thứ nhất BoQ (Ali-bey), link code chính thức của bài | File trọng số upload 04/07/2024, chưa từng bị thay. SHA-256 ghi lại khi tải |
| MegaLoc | `gmberton/MegaLoc` + HF `gberton/MegaLoc` | Tác giả thứ nhất (Berton) | **189/189 tensor giống hệt từng bit** giữa bản HF 02/2025 và bản hiện tại (đã tải cả hai về so). SHA-256 `d4f9f2bc…ffb4f8` |
| SALAD | `serizba/salad`, release v1.0.0 | Tác giả thứ nhất (Izquierdo) | Upload 30/11/2023 |
| AnyLoc | Tổ chức `AnyLoc` (repo `DINO` cho hub, `AnyLoc` cho code bài) | Nhóm tác giả AnyLoc | Vocabulary upload 28/10/2023. ViT-G/14 lấy từ máy chủ chính thức của Meta |
| DINOv2 | `facebookresearch/dinov2` + `dl.fbaipublicfiles.com` | Meta | Chính thức |
| MixVPR | `amaralibey/MixVPR`; link Google Drive lấy từ README chính thức | Tác giả thứ nhất | — |
| CosPlace | `gmberton/CosPlace`, release v1.0 | Tác giả thứ nhất | Upload 04/01/2023 |
| NetVLAD | Trọng số MATLAB gốc của Arandjelović và cs., bản struct do ETH CVG (tác giả hloc) host | Chuyển đổi bởi bên thứ ba có uy tín, được hloc và `VPR-methods-evaluation` dùng | Tương đương theo cấu trúc chuyển đổi |
| DINOv3 | `timm/vit_*_patch16_dinov3.lvd1689m` | timm (Hugging Face) chuyển từ trọng số chính thức; repo `facebook/*` thì bị khóa | Khác biệt số học đã biết và đã được sửa (RoPE bf16) |
| Wheel Python | `download.pytorch.org`, PyPI | Chính thức | Kiểm tra bằng cài đặt offline + import trong venv |

**Kết luận.** Mọi trọng số đều đến từ repo chính thức của tác giả từng phương pháp, hoặc từ bản chuyển đổi được cộng đồng dùng rộng rãi (NetVLAD, DINOv3), và các bản này đã được kiểm chứng hoặc hiệu chỉnh. SHA-256 của mọi file tải về được ghi trong `deps_manifest_*.json` (output của các notebook deps trên Kaggle) và sẽ được đưa vào repo.
