# PDF 3 — Handoff (VPR-Space reproduction)

*Cập nhật 07/11/2026. Dành cho agent hoặc khung chat tiếp theo. Không lặp lại PDF 1 (phát hiện khoa học) và PDF 2 (kinh nghiệm vận hành).*

## Mục tiêu

Tái lập bài **VPR-Space** (Fechner & Neubert, arXiv:2610.35642) trên 16 cặp DB/Q:

- **Chính xác** cho MegaLoc và BoQ (code tác giả công bố).
- **Gần đúng, có ghi chú** cho SALAD, AnyLoc (global, unstructured), DenseVLAD (bài có số để đối chiếu).
- Kết quả tham khảo cho MixVPR, CosPlace, NetVLAD, DINOv3 (bài không công bố số).

## Quyết định đã thống nhất với người dùng

- Dùng **mọi tài nguyên**, không giữ dự phòng; kết quả nhanh nhất với độ chính xác cao nhất.
- **Kiểm tra code, toán, quy ước trước khi chạy**; tự kiểm tra mỗi bước.
- Theo tài liệu "Reproduce VPR - Formatted Headings.docx" (nhánh main). Token không được đưa vào prompt, mã nguồn hay báo cáo.
- Môi trường: gói ghim của tác giả (torch 2.6.0+cu124), cài offline. GT dùng bản phát hành, đã được chứng minh khớp. Dữ liệu để riêng tư theo từng tài khoản.
- DINOv3: dùng mirror timm, sửa RoPE về bf16. DenseVLAD: chuyển từ kornia sang **VLFeat gốc**.

## Tiến độ

| Hạng mục | Trạng thái |
|---|---|
| Bước 1: báo cáo khả năng tái lập, kiểm chứng nguồn | ✅ |
| Dựng lại GT | ✅ 16/16, 0 sai lệch |
| Phiên GPU wave 2–5 (Nhóm A, SALAD, AnyLoc ×4, CosPlace ×8, MixVPR ×3, NetVLAD ×6, DINOv3 ×8, DenseVLAD kornia) | ✅ đã thu đủ vào `results/runs/` |
| Vá AnyLoc MadMax-E0 và ghép 16/16 | ✅ `results/runs/merged-anyloc/` |
| DenseVLAD VLFeat: 640 px (`g-dvlad-vlfeat-640`) | ✅ |
| DenseVLAD VLFeat: vocabulary và whitening từ GSV-Cities (`h-dvlad-vlfeat-urban`, dinhthinh) | ✅ gần bài nhất (r = 0,983) |
| DenseVLAD VLFeat: độ phân giải gốc (`g-dvlad-vlfeat-native`, namhunhh) | xem `reports/KET_QUA_tai_lap.md` §6 |
| Báo cáo kết quả `reports/KET_QUA_tai_lap.md`, bảng đầy đủ `reports/KET_QUA_so_sanh.md` | ✅ |
| PDF 1–3, `environment_summary`, `research_workflow`, `readme.md` | ✅ |

## Vị trí của mọi thứ

| Thứ | Ở đâu |
|---|---|
| Token | `/root/.kaggle/accounts.json` trên máy phiên (quyền 600); **không** có trong repo. Wrapper: `kg <label>` |
| Tài khoản | `kaggle/launch.py` → `USERNAMES` (nhãn → username). L4: 12 tài khoản; T4: mxuanquy, yeshenhi |
| Dữ liệu (riêng tư) | Output `<user>/vprspace-data-a` và `-b` trên: hieuho669, duckzuykobip, nguyenmanh0404, namhunhh, dinhthinh, yeshenhi, mxuanquy |
| Deps | `<user>/vprspace-deps-{core,extra,anyloc,hf}`; VLFeat: `<user>/vprspace-vlfeat-build` trên dinhthinh, namhunhh, hieuho669 |
| Descriptor | Output phiên `<user>/vprspace-run-<session>` → `desc/<set>/<trajectory>.npy`. Tài khoản của từng phiên ghi trong `kaggle/wave*_plan.json` |
| Metric | `results/runs/<session>/results/<set>__<session>.json` (+ `logs/`) |
| Phân tích kiểm chứng | `results/verification/` (GT re-derivation, G0–G2 GT sensitivity) |
| Hình | `results/figures/` |
| GT và pose cục bộ cho tính metric | Giải nén từ output `quynhtran35/vprspace-release-inspect` (small files) |

## Bước tiếp theo (nếu tiếp tục)

1. **DenseVLAD**: nếu tìm được `247code.zip` của Torii (vocabulary `dnscnt_RDSIFT_K128.mat` và phép chiếu PCA), chạy:
   ```bash
   python3 vprspace_repro/densevlad_vlfeat.py --root <root> --desc-dir <desc> --res 640 --vocab-mat <mat>
   ```
   Code đã hỗ trợ đọc `.mat`. Cần thêm bước đọc phép chiếu PCA.
2. **G0–G2**: hỏi tác giả (issue GitHub) về phiên bản ảnh/GT của G0–G2 và định nghĩa cột "% of pos. matches" của Bảng IV.
3. Khi tác giả công bố "fully detailed results" trên website benchmark: đối chiếu MixVPR, CosPlace, NetVLAD, DINOv3 và AnyLoc-unstructured với `reports/KET_QUA_so_sanh.md`.
4. Mọi lần chạy mới:
   - viết plan `kaggle/wave<N>_plan.json`;
   - dry-run với `VPRS_DRY_RUN=1`;
   - chạy smoke test cục bộ (`kaggle/smoke_test.py`);
   - chạy `kaggle/push_pending.py`;
   - theo dõi bằng `kwatch`;
   - thu bằng `kaggle/collect.py`;
   - so sánh bằng `compare.py` và `report_tables.py`.
