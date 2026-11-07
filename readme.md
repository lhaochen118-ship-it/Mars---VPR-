# Mars—VPR: tái lập bài báo VPR-Space (Fechner & Neubert, arXiv:2610.35642)

## Đọc gì trước

| Thứ tự | File | Nội dung |
|---|---|---|
| 1 | [`reports/KET_QUA_tai_lap.md`](reports/KET_QUA_tai_lap.md) | **Kết quả Bước 2**: Bảng II–III, Hình 6, DenseVLAD, các điểm lệch và phân tích |
| 2 | [`docs/handoff/`](docs/handoff/) | PDF 1 Scientific Findings, PDF 2 Engineering & Ops Lessons, PDF 3 Handoff, `environment_summary`, `research_workflow` |
| 3 | [`reports/BUOC1_bao_cao_kha_nang_tai_lap.md`](reports/BUOC1_bao_cao_kha_nang_tai_lap.md) | Bước 1: phần nào tái lập chính xác, phần nào gần đúng, phần nào không công bố |
| 4 | [`reports/KIEM_CHUNG_phan_thay_the_va_nguon.md`](reports/KIEM_CHUNG_phan_thay_the_va_nguon.md) | Kiểm chứng phương án thay thế và nguồn trọng số |
| 5 | [`reports/KET_QUA_so_sanh.md`](reports/KET_QUA_so_sanh.md) | Bảng so sánh đầy đủ mọi bộ descriptor (sinh tự động) |

## Kết quả chính (một dòng)

- **BoQ**: khớp 16/16 cặp ở Bảng II.
- **MegaLoc**: khớp 14/16; lệch thật duy nhất ở MM G0–G2.
- **SALAD** (322 px): khớp 15/16.
- **AnyLoc-global** (322 px): khớp 14/16.
- **DenseVLAD**: không tái lập chính xác được, vì vocabulary và PCA gốc của Torii đã mất. Xem báo cáo §6.

## Cấu trúc repo

```
vprspace_repro/
  upstream/              code tác giả, giữ NGUYÊN VĂN (commit eae17bf): dataset, evaluate, methods, metrics
  configs.py             43 cấu hình descriptor (Nhóm A chính xác + tái cài đặt có ghi chú)
  worker.py              trích descriptor trên 1 GPU (torch.hub/HF offline, batch tự giảm khi OOM)
  densevlad_vlfeat.py    DenseVLAD trên VLFeat 0.9.21 gốc (vl_phow); tuỳ chọn vocab + whitening từ GSV-Cities
  vlfeat_phow/           shim C (phow_shim.c) gọi vl_imsmooth/vl_dsift giống MEX của MATLAB, wrapper ctypes phow.py
  densevlad_pca.py       quét PCA-whitening (phân tích độ nhạy)
  densevlad_gpu.py       DenseVLAD bản kornia cũ (đã được thay, giữ để đối chiếu)
  metrics_all.py         metric bằng chính evaluate.py/metrics.py của tác giả
  compare.py             so sánh với paper_reference.json (Bảng II–V, Hình 6)
  report_tables.py       bảng Markdown cho báo cáo, sinh từ results/runs
  plot_table2.py, figures.py   hình
  rederive_gt.py, gt_param_search.py, g0g2_gt_sensitivity.py   kiểm chứng GT
  paper_reference.json   mọi con số chép từ bài báo
kaggle/
  kernels/               notebook: release_inspect, data_prep, deps_prep, vlfeat_build, torii_fetch, gpu_run/bootstrap
  launch.py              metadata + push (USERNAMES)
  launch_session.py      nhúng code vào bootstrap
  push_pending.py        đẩy phiên theo plan (tự đợi khi đủ 2 slot GPU)
  collect.py, merge_results.py, merge_split.py, smoke_test.py
  wave2..7_plan.json     các plan phiên đã chạy
  tools/                 kg (CLI theo tài khoản, không in token), kwatch, kwatch_many
results/
  runs/<session>/        results/*.json (metric), logs/ (worker, pip freeze), log phiên
  runs/merged-anyloc/    AnyLoc 16/16 sau khi vá MadMax-E0
  verification/          GT re-derivation, độ nhạy GT ở G0–G2
  figures/               hình
reports/                 báo cáo tiếng Việt
docs/handoff/            tài liệu bàn giao (.md + .pdf), md2pdf.py
```

## Chạy lại

1. Đặt token Kaggle vào `/root/.kaggle/accounts.json` (quyền 600, **không commit**), rồi liên kết các công cụ:
   ```bash
   ln -sf $PWD/kaggle/tools/* /usr/local/bin/
   ```
2. Với mỗi tài khoản GPU, chạy notebook CPU (Internet bật):
   - `vprspace-data-a`/`-b`: dữ liệu, kiểm SHA-256;
   - `vprspace-deps-core`/`-extra`/`-anyloc`/`-hf`: wheel ghim, cache hub/HF;
   - `vprspace-vlfeat-build`: VLFeat cho DenseVLAD.

   Ví dụ:
   ```bash
   python3 kaggle/launch.py push <acct> vprspace-deps-core kaggle/kernels/deps_prep/vprspace_deps_prep.py --internet --sub "__PROFILE__=core"
   ```
3. Chạy phiên GPU L4×4 (Internet tắt) theo plan. Kiểm tra trước bằng dry-run:
   ```bash
   VPRS_DRY_RUN=1 python3 kaggle/push_pending.py kaggle/wave6_plan.json <session>
   ```
   ```bash
   python3 kaggle/push_pending.py kaggle/wave6_plan.json <session...>
   ```
4. Thu kết quả và so sánh:
   ```bash
   python3 kaggle/collect.py kaggle/wave6_plan.json results/runs
   ```
   ```bash
   python3 vprspace_repro/compare.py --results results/runs --out reports/KET_QUA_so_sanh.md
   ```
   ```bash
   python3 vprspace_repro/report_tables.py --results results/runs
   ```

## Việc tiếp theo

Xem mục "Bước tiếp theo" trong [`docs/handoff/03_handoff.md`](docs/handoff/03_handoff.md).
