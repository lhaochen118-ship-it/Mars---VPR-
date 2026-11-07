# Research workflow — VPR-Space reproduction

*Cập nhật 07/11/2026. Quy trình thực tế đã dùng, theo tài liệu "Reproduce VPR" (nhánh main). Mỗi bước nêu đầu vào, công cụ và tiêu chí "xong".*

**Luồng:** (1) đọc bài và release → (2) kiểm kê thành phần → (3) lập plan → (4) kiểm chứng dữ liệu và GT → (5) code và smoke test → (6) chạy Kaggle → (7) thu, ghép, so sánh → (8) phân tích sai lệch. Nếu cần biến thể có căn cứ (độ phân giải, VLFeat, whitening…), quay lại bước (5). Cuối cùng là (9) báo cáo và bàn giao.

| Bước | Việc làm | Công cụ | Xong khi |
|---|---|---|---|
| 1 | Đọc bài (alphaXiv); đọc release và code tác giả (`upstream/`, giữ nguyên văn) | `paper_reference.json` chứa mọi số trong bài | Mọi bảng và hình đều có số tham chiếu |
| 2 | Phân loại từng thành phần: code/trọng số có công bố không, cấu hình có công bố không | `reports/BUOC1_*`, `reports/KIEM_CHUNG_*` | Có nguồn uy tín cho mọi trọng số; nêu rõ thứ không có |
| 3 | Lập plan phiên: cấu hình → tài khoản; tối đa 2 phiên GPU/tài khoản; cân bằng tải LPT | `kaggle/wave*_plan.json` | Dry-run (`VPRS_DRY_RUN=1`) in metadata đúng |
| 4 | Kiểm chứng SHA-256 dữ liệu; dựng lại GT từ ước lượng RPE | `release_inspect`, `rederive_gt.py` | 11/11 SHA khớp; GT khớp 16/16 |
| 5 | Viết code; kiểm thử cục bộ bằng ảnh thật, deps thật và cây `/kaggle/input` giả | `kaggle/smoke_test.py`; với binary ngoài thì dùng test double cho `phow` | Mọi cấu hình PASS (shape, finite, không FAILED) |
| 6 | Chạy phiên: notebook dữ liệu/deps (CPU) → phiên L4 Internet tắt → `kwatch` | `push_pending.py`, `kwatch` | Exit 0; không treo hàng đợi quá 8 phút |
| 7 | Thu `results/`, `logs/`; ghép phiên tách hoặc phiên vá; so sánh với bài | `collect.py`, `merge_results.py`, `merge_split.py`, `compare.py`, `report_tables.py` | 16/16 cặp cho mỗi bộ descriptor |
| 8 | Với mỗi sai lệch: xét giả thuyết **trước** khi chạy thêm (code, môi trường, độ phân giải, GT, triển khai) | `g0g2_gt_sensitivity.py`, `densevlad_pca.py`, quét cấu hình | Mỗi sai lệch được giải thích, hoặc được đánh dấu "không xác định được" kèm bằng chứng |
| 9 | Viết báo cáo; bàn giao; dọn nhánh; quét token | `docs/handoff/md2pdf.py` | 3 PDF + `environment_summary` + `research_workflow` + `readme.md` |

## Nguyên tắc đã áp dụng

- **Kiểm tra trước khi chạy**: code, công thức và quy ước (MATLAB/VLFeat, chỉ số từ 0 hay 1, ddof) phải được rà và kiểm thử cục bộ trước mỗi phiên GPU.
- **Một nguồn sự thật cho số liệu**: JSON metric của phiên. Bảng báo cáo sinh tự động, không chép tay.
- **Biến thể phải chọn trước** (độ phân giải 224/322/518/native…). Biến thể chọn sau khi xem số của bài được ghi rõ là "phân tích độ nhạy".
- **Không lộ token**: token chỉ nằm trong `/root/.kaggle/accounts.json`; quét `KGAT_` trước mỗi commit.
