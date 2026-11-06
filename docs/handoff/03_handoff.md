# PDF 3 — Handoff (VPR-Space reproduction)

*Cập nhật 06/11/2026. Dành cho agent hoặc khung chat tiếp theo; không lặp lại nội dung của PDF 1 và PDF 2.*

## Mục tiêu

Tái lập bài **VPR-Space** (Fechner & Neubert, arXiv:2610.35642) trên **16 cặp DB/Q**:

- Khớp **chính xác** cho MegaLoc và BoQ: Bảng II, Bảng III, Bảng IV, Hình 6.
- Tái cài đặt **gần đúng**, có ghi chú, cho SALAD, AnyLoc (global, unstructured), MixVPR, CosPlace, NetVLAD, DINOv3 và DenseVLAD.

## Các quyết định đã thống nhất với người dùng

- Dùng **mọi tài nguyên**: 10 tài khoản Kaggle, không giữ dự phòng.
- Làm đủ 10 cấu hình. Nhóm A dùng đúng code tác giả; các nhóm khác dùng cấu hình chuẩn chọn trước, kèm quét biến thể.
- Môi trường: các gói ghim của tác giả, cài offline trong venv. Riêng DenseVLAD và bản kiểm tra môi trường dùng môi trường mặc định Kaggle.
- Dữ liệu để **riêng tư** trên từng tài khoản: chưa có token `nhatlong1103`, và giấy phép phân phối lại của MADMAX/S3LI chưa rõ.
- DINOv3 dùng bản mirror của timm (không bị khóa), có sửa RoPE về bf16.
- Không sinh lại GT bằng RPE. GT phát hành đã được chứng minh khớp 100% bằng cách dựng lại bước bỏ phiếu.

## Tiến độ

| Hạng mục | Trạng thái |
|---|---|
| Báo cáo Bước 1, kiểm chứng release, kiểm chứng nguồn | ✅ |
| Dựng lại bước bỏ phiếu GT | ✅ 0 sai lệch, 16/16 cặp |
| Notebook dữ liệu và deps trên 7 tài khoản GPU | ✅ (deps-core cần 5 lần sửa; xem PDF 2) |
| Wave 2: 12 phiên GPU (`kaggle/wave2_plan.json`) | 🔄 đang chạy |
| So sánh với bài báo, báo cáo kết quả | ⏳ chờ kết quả |

## Vị trí của mọi thứ

| Thứ | Ở đâu |
|---|---|
| Token | `/root/.kaggle/accounts.json` (máy phiên này; **không** có trong repo) |
| Dữ liệu (riêng tư) | Output của `<user>/vprspace-data-a` và `<user>/vprspace-data-b` trên 7 tài khoản GPU |
| Deps | Output của `<user>/vprspace-deps-{core,extra,anyloc}` |
| Descriptor | Output của `<user>/vprspace-run-<session>` ở `desc/<config>/<trajectory>.npy` (float32) |
| Kết quả metric | Output của phiên, ở `results/<config>__<session>.json`; trong repo là `results/runs/` |
| Manifest dữ liệu (SHA-256 từng file) | Output của `quynhtran35/vprspace-release-inspect` |

## Bước tiếp theo

1. Thu kết quả:
   ```bash
   python3 kaggle/collect.py kaggle/wave2_plan.json results/runs
   ```
2. AnyLoc demo1024 chạy tách 2 phiên, nên cần tải descriptor của cả hai rồi chạy `metrics_all.py` cục bộ. Thư mục GT và pose cục bộ được giải nén từ output của `release-inspect`.
3. So sánh:
   ```bash
   python3 vprspace_repro/compare.py --results results/runs --out reports/KET_QUA_so_sanh.md
   ```
4. Nếu một phương pháp gần đúng lệch nhiều so với bài báo: quét thêm biến thể (`configs.py`) trong wave 3.
5. Viết PDF 1 (Scientific Findings), cập nhật 3 PDF và `environment_summary`, rồi dọn nhánh.
