# Mars—VPR: tái lập bài báo VPR-Space (Fechner & Neubert, arXiv:2610.35642)

## Đọc gì trước

| Thứ tự | File | Nội dung |
|---|---|---|
| 1 | [`reports/BUOC1_bao_cao_kha_nang_tai_lap.md`](reports/BUOC1_bao_cao_kha_nang_tai_lap.md) | Bước 1: phần nào tái lập chính xác, phần nào gần đúng, phần nào không công bố. Kiểm chứng release |
| 2 | [`reports/KIEM_CHUNG_phan_thay_the_va_nguon.md`](reports/KIEM_CHUNG_phan_thay_the_va_nguon.md) | Kiểm tra lại phương án thay thế và nguồn trọng số. Dựng lại GT khớp 100% |
| 3 | `reports/KET_QUA_*.md` | Kết quả Bước 2: so sánh với Bảng II–IV và Hình 6 *(được tạo khi các phiên GPU chạy xong)* |
| 4 | [`docs/handoff/`](docs/handoff/) | PDF bàn giao (Scientific Findings, Engineering & Ops Lessons, Handoff), `environment_summary` |

## Cấu trúc repo

```
vprspace_repro/
  upstream/            code của tác giả, giữ NGUYÊN VĂN (commit eae17bf): dataset, evaluate, methods, metrics
  configs.py           ~35 cấu hình descriptor: Nhóm A (chính xác) + các bản tái cài đặt có ghi chú
  worker.py            trích descriptor trên 1 GPU (torch.hub và HF offline, batch tự giảm khi OOM)
  densevlad_gpu.py     DenseVLAD tái cài đặt (dense RootSIFT bằng kornia)
  metrics_all.py       metric tính bằng chính evaluate.py/metrics.py của tác giả
  compare.py           so sánh với paper_reference.json (Bảng II–V, Hình 6)
  rederive_gt.py       dựng lại GT từ misc/rpe_estimates + poses + Bảng V
  figures.py           Hình 1 (phải) và Hình 5
  paper_reference.json các con số chép từ bài báo
kaggle/
  kernels/             notebook Kaggle: release_inspect, data_prep, deps_prep, gpu_run/bootstrap, l4_probe
  launch.py, launch_session.py, push_pending.py, auto_launch.py, collect.py, wave2_plan.json
  tools/               kg (CLI theo tài khoản, không in token), kwatch, kwatch_many
results/               kết quả (verification/, runs/<session>/...)
reports/               báo cáo tiếng Việt
docs/handoff/          tài liệu bàn giao
```

## Chạy lại

1. Đặt token Kaggle vào `/root/.kaggle/accounts.json` (quyền 600, **không commit**), rồi chạy `ln -sf $PWD/kaggle/tools/* /usr/local/bin/`.
2. Notebook CPU (Internet bật) cho mỗi tài khoản GPU: `vprspace-data-a`/`-b` (tải 11 phần, kiểm SHA-256) và `vprspace-deps-core`/`-extra`/`-anyloc` (wheel ghim, cache hub/HF).
   ```bash
   python3 kaggle/launch.py push <acct> vprspace-deps-core kaggle/kernels/deps_prep/vprspace_deps_prep.py --internet --sub "__PROFILE__=core"
   ```
3. Phiên GPU L4×4 (Internet tắt) theo `kaggle/wave2_plan.json`:
   ```bash
   python3 kaggle/push_pending.py kaggle/wave2_plan.json <session...>
   ```
4. Thu kết quả và so sánh:
   ```bash
   python3 kaggle/collect.py kaggle/wave2_plan.json results/runs
   ```
   ```bash
   python3 vprspace_repro/compare.py --results results/runs --out reports/KET_QUA_so_sanh.md
   ```
