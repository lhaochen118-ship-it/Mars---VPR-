# PDF 1 — Scientific Findings (VPR-Space reproduction)

*Cập nhật 07/11/2026. Chỉ nêu phát hiện khoa học và bằng chứng; kinh nghiệm vận hành ở PDF 2, trạng thái và bước tiếp theo ở PDF 3. Số liệu chi tiết: `reports/KET_QUA_tai_lap.md`.*

## 1. Đóng góp của đợt tái lập

1. **Tái lập chính xác** kết quả của hai phương pháp mà tác giả công bố code (`methods.py`).
   - Dùng đúng dữ liệu phát hành, GT, code đánh giá và môi trường ghim của tác giả.
   - Dựng lại hoàn toàn bước sinh GT từ các ước lượng RPE phát hành; khớp 100%.
2. **Xác định cấu hình chưa công bố** của SALAD và AnyLoc bằng quét độ phân giải chọn trước: cả hai dùng **322×322**.
   - SALAD khớp 15/16 cặp; AnyLoc-global khớp 14/16 cặp.
3. **Tìm ra nguyên nhân** khiến DenseVLAD tái cài đặt lệch xa bài báo, và định lượng phần không thể tái lập (§3.2).
4. Phát hiện **4 điểm không nhất quán** trong bài hoặc bản phát hành (§3).

## 2. Kết quả tích cực (có bằng chứng)

| # | Phát hiện | Bằng chứng | Điều kiện kiểm chứng |
|---|---|---|---|
| P1 | BoQ: Bảng II 16/16 cặp và 3 hàng tổng hợp, Bảng III 12/12 ô, Hình 6a khớp hoàn toàn | `results/runs/a-exact/results/boq__a-exact.json` | `compare.py`; khớp sau khi làm tròn 2 chữ số |
| P2 | MegaLoc: Bảng II 14/16, Bảng III 11/12; trung bình 0,469 so với 0,47 | `megaloc__a-exact.json` | như P1 |
| P3 | Kết quả rất ổn định: batch, môi trường và GPU chỉ làm \|ΔAUPRC\| ≤ 8,2·10⁻⁴ | `a-exact-kaggleenv`, `a-exact-t4`, `a-exact-t4-bs` | Số học GPU khác nhau không đổi kết luận nào |
| P4 | GT phát hành = bỏ phiếu theo Bảng V trên `misc/rpe_estimates`: (1) tư thế tương đối `inv(P_q)·P_db`; (2) lỗi tịnh tiến 5-DoF min₍ₛ≥₀₎‖s·t̂ − t‖; (3) điều kiện đúng ở **cả hai** thứ tự ảnh; (4) ≥ n RPE; (5) khoảng cách ≤ d_max | `results/verification/gt_rederivation.*` | 0 sai lệch trên 16/16 cặp, cả hard lẫn soft |
| P5 | Cách đọc bảng của bài: std là **ddof=1**; "Worst case" là **min theo từng metric**; chỉ số truy vấn ở Hình 6 **bắt đầu từ 0** | So sánh hai cách đọc trong `compare.py` | Chỉ cách đọc này khớp số của bài |
| P6 | SALAD và AnyLoc-global của bài dùng ảnh 322×322 (ToTensor → Resize bicubic antialias → Normalize, như BoQ của tác giả) | `b-salad`, `merged-anyloc` | Các độ phân giải 224, 518, native, demo đều lệch rõ hơn (1–4/16) |
| P7 | Nhận định chính của bài được xác nhận: R@100p gần 0 ở trường hợp xấu nhất với mọi phương pháp; trung bình AUPRC tốt nhất ≈ 0,47 | Bảng III tái lập | — |
| P8 | Độ phân giải quan trọng: ở 518×518, AnyLoc-global tăng +0,029, SALAD +0,017, BoQ +0,005 so với cấu hình của bài | `KET_QUA_tai_lap.md` §9 | Thứ hạng phương pháp phụ thuộc độ phân giải |

## 3. Kết quả tiêu cực, hạn chế, điểm không nhất quán

### 3.1 MegaLoc và SALAD ở MADMAX G0–G2

- **Lệch**: MegaLoc 0,553 so với 0,57; SALAD 0,471 so với 0,48. BoQ và AnyLoc ở cùng cặp thì khớp.
- **Không do** code, môi trường, batch hay độ phân giải.
- **Có thể do GT, nhưng không xác định được**:
  - Thay đổi GT kéo các phương pháp đi cùng chiều, và không có thay đổi một tham số nào của Bảng V làm MegaLoc và BoQ cùng khớp.
  - Có 55/2.200 tổ hợp đổi nhiều tham số làm cả hai khớp.
  - Bằng chứng: `results/verification/g0g2_gt_sensitivity.json`.
- **Kết luận**: khả năng cao bài dùng ảnh hoặc GT của G0–G2 khác bản phát hành, nhưng không chứng minh được.

### 3.2 DenseVLAD

*(xem số liệu cuối cùng ở `KET_QUA_tai_lap.md` §6)*

- Bản kornia ban đầu cho AUPRC trung bình 0,219, so với 0,35 của bài. Nguyên nhân:
  1. **Thiếu ngưỡng tương phản của `vl_phow`** (0,005). Ở ảnh SPICE và LunarLoc, 21–57% descriptor bị VLFeat đưa về 0 (MADMAX 0,4–14%); kornia lại chuẩn hoá nhiễu thành descriptor đơn vị.
  2. **Khác hình học bin**: VLFeat nội suy song tuyến và có cửa sổ Gauss 1,5 bin; σ làm mịn = size/6.
  3. **PCA-whitening suy biến**: whitening học trên 3.989 vector DB với 3.989 thành phần làm mọi vector DB có cùng chuẩn và cos đôi một = −1/(n−1), nên retrieval sụp (0,037).
- Pipeline **VLFeat gốc** (bản 0.9.21, gọi đúng như MEX của MATLAB) đạt 0,296. Nó có **tương quan theo cặp r = 0,974** với cột DenseVLAD của bài: cùng hình dạng nhưng thấp hơn.
- Phần còn thiếu là **vocabulary và phép chiếu PCA-whitening 4096-D của Torii** (`247code.zip`), học trên ảnh phố đô thị. Gói này không còn: host gốc trả 403, Internet Archive không lưu.
- Whitening học trên DB ở k = 128–512 đưa trung bình lên 0,36–0,38, nhưng làm hỏng hình dạng theo cặp (r = 0,86–0,93).

### 3.3 Bảng IV, cột "% of pos. matches"

Không khớp với GT phát hành theo bất kỳ định nghĩa nào; cột "% queries w/ match" thì khớp. Có thể là lỗi biên soạn.

### 3.4 Nhận định "AnyLoc-unstructured không cao nhất ở cặp nào"

Không đúng với vocabulary unstructured chính thức ở cấu hình 322 (cấu hình làm AnyLoc-global khớp bài):

- MM A0–A2: 0,733, so với BoQ 0,70 trong bài;
- MM E0–E1: 0,335, so với MegaLoc 0,33.

### 3.5 Lỗi nhỏ khác

- **Ảnh hỏng trong bản phát hành**: `MadMax-E0/0293.png` thiếu IEND, chỉ giải mã được 1.387/1.544 hàng.
  - Bài cho thấy tác giả đã đọc phần giải mã được: chỉ khi đọc như vậy mới ra đúng số E0–E1.
  - Bộ đọc nghiêm ngặt (mặc định của PIL) sẽ lỗi.
- **Hình 6b**: false positive 0,353 không tương ứng với top-1 (0,374). Giá trị best correct 0,078 thì khớp.
- **Hình 6c**: lệch 0,001 ở cả hai giá trị.

## 4. Ý nghĩa khoa học

- Benchmark **tái lập được** ở mức số liệu đối với các phương pháp có code. Phần chưa tái lập được tập trung ở các phương pháp mà bài không công bố cấu hình.
- Kết luận "không có phương pháp nào mạnh ở miền space analog" là **vững**: bất kể cấu hình, không phương pháp nào vượt 0,48 AUPRC trung bình, và R@100p xấu nhất ≈ 0.
- So sánh giữa các phương pháp **nhạy với độ phân giải**: chênh lệch tới 0,03 AUPRC, ngang khoảng cách giữa các phương pháp. Benchmark nên cố định và công bố độ phân giải.
- DenseVLAD trên ảnh không gian tối hoặc đồng nhất chịu ảnh hưởng mạnh của **ngưỡng tương phản** và **whitening**. Đây là một chi tiết triển khai làm đổi hơn 0,1 AUPRC.

## 5. Điều kiện để kiểm chứng lại

- Dữ liệu `vpr-space-v0` v0.1.0 với SHA-256 khớp, GT `VPR-GT-RPE-0`, code tác giả commit eae17bf.
- Môi trường ghim: torch 2.6.0+cu124 …; phần cứng L4 hoặc T4.
- Đọc ảnh cắt cụt bằng `LOAD_TRUNCATED_IMAGES=True`.
- Lệnh tái tạo mọi bảng:
  ```bash
  python3 vprspace_repro/compare.py --results results/runs --out reports/KET_QUA_so_sanh.md
  ```
  ```bash
  python3 vprspace_repro/report_tables.py --results results/runs
  ```
