# VPR-Space reproduction: comparison with the paper

63 descriptor sets with results.

## Table II (AUPRC): agreement with the paper

| descriptor set | paper column | setups exact (2 dp) | mean abs diff | max abs diff | mean AUPRC (ours / paper) |
|---|---|---|---|---|---|
| `megaloc [a-exact-t4]` | megaloc | 14/16 | 0.0036 | 0.0168 | 0.469 / 0.47 |
| `megaloc_bs128 [a-exact-t4-bs]` | megaloc | 14/16 | 0.0036 | 0.0168 | 0.469 / 0.47 |
| `megaloc [a-exact-kaggleenv]` | megaloc | 14/16 | 0.0036 | 0.0169 | 0.469 / 0.47 |
| `megaloc [a-exact]` | megaloc | 14/16 | 0.0037 | 0.0168 | 0.469 / 0.47 |
| `megaloc_bs128 [a-exact]` | megaloc | 14/16 | 0.0037 | 0.0168 | 0.469 / 0.47 |
| `megaloc_322 [d-groupa-resolution-t4]` | megaloc | 1/16 | 0.0245 | 0.0803 | 0.467 / 0.47 |
| `megaloc_224 [d-groupa-resolution-t4]` | megaloc | 1/16 | 0.0359 | 0.0640 | 0.449 / 0.47 |
| `boq_bs256 [a-exact-t4-bs]` | boq | 15/16 | 0.0024 | 0.0051 | 0.469 / 0.47 |
| `boq [a-exact-t4]` | boq | 15/16 | 0.0024 | 0.0051 | 0.469 / 0.47 |
| `boq [a-exact-kaggleenv]` | boq | 15/16 | 0.0024 | 0.0051 | 0.469 / 0.47 |
| `boq [a-exact]` | boq | 16/16 | 0.0025 | 0.0050 | 0.469 / 0.47 |
| `boq_bs256 [a-exact]` | boq | 16/16 | 0.0025 | 0.0050 | 0.469 / 0.47 |
| `boq_518 [d-groupa-resolution-t4]` | boq | 3/16 | 0.0144 | 0.0599 | 0.474 / 0.47 |
| `boq_224 [d-groupa-resolution-t4]` | boq | 1/16 | 0.0274 | 0.0775 | 0.442 / 0.47 |
| `salad_322 [b-salad]` | salad | 15/16 | 0.0033 | 0.0086 | 0.394 / 0.39 |
| `salad_322_official [b-salad]` | salad | 10/16 | 0.0047 | 0.0141 | 0.393 / 0.39 |
| `salad_518 [b-salad]` | salad | 1/16 | 0.0263 | 0.0840 | 0.411 / 0.39 |
| `salad_224 [b-salad]` | salad | 1/16 | 0.0284 | 0.1163 | 0.367 / 0.39 |
| `salad_native14 [b-salad]` | salad | 2/16 | 0.0446 | 0.0963 | 0.384 / 0.39 |
| `anyloc_global_322 [merged]` | anyloc_global | 14/16 | 0.0031 | 0.0064 | 0.395 / 0.40 |
| `anyloc_global_518 [merged]` | anyloc_global | 4/16 | 0.0369 | 0.0914 | 0.424 / 0.40 |
| `anyloc_global_demo1024 [merged]` | anyloc_global | 3/16 | 0.0402 | 0.1141 | 0.423 / 0.40 |
| `anyloc_global_224 [merged]` | anyloc_global | 2/16 | 0.0431 | 0.0882 | 0.353 / 0.40 |
| `densevlad_vlfeat_640_drop0 [g-dvlad-vlfeat-640]` | densevlad | 2/16 | 0.0581 | 0.1294 | 0.296 / 0.35 |
| `densevlad_vlfeat_640_keep0 [g-dvlad-vlfeat-640]` | densevlad | 2/16 | 0.0582 | 0.1298 | 0.295 / 0.35 |
| `densevlad_vlfeat_640_drop0_pcaw128 [local-pca]` | densevlad | 2/16 | 0.0682 | 0.1808 | 0.357 / 0.35 |
| `densevlad_vlfeat_640_drop0_pcaw256 [local-pca]` | densevlad | 2/16 | 0.0793 | 0.2151 | 0.383 / 0.35 |
| `densevlad_vlfeat_640_drop0_pcaw64 [local-pca]` | densevlad | 1/16 | 0.0815 | 0.2034 | 0.317 / 0.35 |
| `densevlad_vlfeat_640_drop0_pcaw512 [local-pca]` | densevlad | 2/16 | 0.0927 | 0.2122 | 0.361 / 0.35 |
| `densevlad_vlfeat_640_drop0_pcaw1024 [local-pca]` | densevlad | 1/16 | 0.1166 | 0.3889 | 0.305 / 0.35 |
| `densevlad_16384 [b-densevlad]` | densevlad | 1/16 | 0.1310 | 0.2628 | 0.219 / 0.35 |
| `densevlad_vlfeat_640_drop0_pcaw2048 [local-pca]` | densevlad | 1/16 | 0.1487 | 0.5791 | 0.227 / 0.35 |
| `densevlad_vlfeat_640_drop0_pcaw3980 [local-pca]` | densevlad | 1/16 | 0.2242 | 0.7508 | 0.126 / 0.35 |
| `densevlad_pca4096 [b-densevlad]` | densevlad | 0/16 | 0.3131 | 0.7060 | 0.037 / 0.35 |

### Per setup: `megaloc [a-exact-t4]` vs paper column megaloc

| setup | ours | paper | diff | exact |
|---|---|---|---|---|
| MM A0 - A2 | 0.6677 | 0.67 | -0.0023 | ✅ |
| MM A2 - A4 | 0.3648 | 0.36 | +0.0048 | ✅ |
| MM C0 - C1 | 0.5137 | 0.51 | +0.0037 | ✅ |
| MM E0 - E1 | 0.3315 | 0.33 | +0.0015 | ✅ |
| MM F0 - F3 | 0.7883 | 0.79 | -0.0017 | ✅ |
| MM G0 - G2 | 0.5532 | 0.57 | -0.0168 | ❌ |
| S3E crater - crater inout | 0.5187 | 0.52 | -0.0013 | ✅ |
| S3E traverse 1 - loops | 0.1966 | 0.20 | -0.0034 | ✅ |
| LL 9 - 10 | 0.6165 | 0.62 | -0.0035 | ✅ |
| LL 15 - 17 | 0.4801 | 0.48 | +0.0001 | ✅ |
| HL3 SPAD dawn - night | 0.1776 | 0.18 | -0.0024 | ✅ |
| HL3 SPAD dawn - noon | 0.7578 | 0.76 | -0.0022 | ✅ |
| HL3 SPAD noon - night | 0.2155 | 0.21 | +0.0055 | ❌ |
| HL3 FLIR dawn - night | 0.3310 | 0.33 | +0.0010 | ✅ |
| HL3 FLIR dawn - noon | 0.7035 | 0.70 | +0.0035 | ✅ |
| HL3 FLIR noon - night | 0.2845 | 0.28 | +0.0045 | ✅ |

| Table III | ours worst (min) | ours worst (hardest setup) | paper worst | ours mean ± std (ddof0/ddof1) | paper mean ± std |
|---|---|---|---|---|---|
| R@1 | 0.242 | 0.242 | 0.240 | 0.593 ± 0.241/0.249 | 0.59 ± 0.25 |
| R@5 | 0.484 | 0.484 | 0.460 | 0.747 ± 0.186/0.192 | 0.75 ± 0.19 |
| R@10 | 0.526 | 0.526 | 0.530 | 0.812 ± 0.146/0.151 | 0.81 ± 0.15 |
| R@100p | 0.000 | 0.026 | 0.000 | 0.072 ± 0.071/0.073 | 0.07 ± 0.07 |

### Per setup: `boq_bs256 [a-exact-t4-bs]` vs paper column boq

| setup | ours | paper | diff | exact |
|---|---|---|---|---|
| MM A0 - A2 | 0.7002 | 0.70 | +0.0002 | ✅ |
| MM A2 - A4 | 0.4092 | 0.41 | -0.0008 | ✅ |
| MM C0 - C1 | 0.5242 | 0.52 | +0.0042 | ✅ |
| MM E0 - E1 | 0.3006 | 0.30 | +0.0006 | ✅ |
| MM F0 - F3 | 0.7127 | 0.71 | +0.0027 | ✅ |
| MM G0 - G2 | 0.4863 | 0.49 | -0.0037 | ✅ |
| S3E crater - crater inout | 0.3337 | 0.33 | +0.0037 | ✅ |
| S3E traverse 1 - loops | 0.1649 | 0.17 | -0.0051 | ❌ |
| LL 9 - 10 | 0.5646 | 0.56 | +0.0046 | ✅ |
| LL 15 - 17 | 0.4880 | 0.49 | -0.0020 | ✅ |
| HL3 SPAD dawn - night | 0.2100 | 0.21 | -0.0000 | ✅ |
| HL3 SPAD dawn - noon | 0.7852 | 0.79 | -0.0048 | ✅ |
| HL3 SPAD noon - night | 0.3496 | 0.35 | -0.0004 | ✅ |
| HL3 FLIR dawn - night | 0.3525 | 0.35 | +0.0025 | ✅ |
| HL3 FLIR dawn - noon | 0.7208 | 0.72 | +0.0008 | ✅ |
| HL3 FLIR noon - night | 0.4077 | 0.41 | -0.0023 | ✅ |

| Table III | ours worst (min) | ours worst (hardest setup) | paper worst | ours mean ± std (ddof0/ddof1) | paper mean ± std |
|---|---|---|---|---|---|
| R@1 | 0.375 | 0.419 | 0.380 | 0.634 ± 0.191/0.198 | 0.64 ± 0.20 |
| R@5 | 0.594 | 0.655 | 0.590 | 0.778 ± 0.133/0.137 | 0.78 ± 0.14 |
| R@10 | 0.688 | 0.735 | 0.690 | 0.843 ± 0.108/0.112 | 0.84 ± 0.11 |
| R@100p | 0.000 | 0.000 | 0.000 | 0.076 ± 0.073/0.075 | 0.08 ± 0.08 |

### Per setup: `salad_322 [b-salad]` vs paper column salad

| setup | ours | paper | diff | exact |
|---|---|---|---|---|
| MM A0 - A2 | 0.6549 | 0.65 | +0.0049 | ✅ |
| MM A2 - A4 | 0.3856 | 0.39 | -0.0044 | ✅ |
| MM C0 - C1 | 0.4112 | 0.41 | +0.0012 | ✅ |
| MM E0 - E1 | 0.2323 | 0.23 | +0.0023 | ✅ |
| MM F0 - F3 | 0.6785 | 0.68 | -0.0015 | ✅ |
| MM G0 - G2 | 0.4714 | 0.48 | -0.0086 | ❌ |
| S3E crater - crater inout | 0.3229 | 0.32 | +0.0029 | ✅ |
| S3E traverse 1 - loops | 0.0975 | 0.10 | -0.0025 | ✅ |
| LL 9 - 10 | 0.3623 | 0.36 | +0.0023 | ✅ |
| LL 15 - 17 | 0.5053 | 0.51 | -0.0047 | ✅ |
| HL3 SPAD dawn - night | 0.1325 | 0.13 | +0.0025 | ✅ |
| HL3 SPAD dawn - noon | 0.6863 | 0.69 | -0.0037 | ✅ |
| HL3 SPAD noon - night | 0.1875 | 0.19 | -0.0025 | ✅ |
| HL3 FLIR dawn - night | 0.2734 | 0.27 | +0.0034 | ✅ |
| HL3 FLIR dawn - noon | 0.5953 | 0.60 | -0.0047 | ✅ |
| HL3 FLIR noon - night | 0.3086 | 0.31 | -0.0014 | ✅ |

| Table III | ours worst (min) | ours worst (hardest setup) | paper worst | ours mean ± std (ddof0/ddof1) | paper mean ± std |
|---|---|---|---|---|---|
| R@1 | 0.253 | 0.413 | 0.250 | 0.593 ± 0.203/0.210 | 0.59 ± 0.21 |
| R@5 | 0.558 | 0.625 | 0.560 | 0.768 ± 0.139/0.144 | 0.77 ± 0.14 |
| R@10 | 0.663 | 0.708 | 0.660 | 0.837 ± 0.117/0.121 | 0.84 ± 0.12 |
| R@100p | 0.000 | 0.000 | 0.000 | 0.053 ± 0.071/0.073 | 0.05 ± 0.07 |

### Per setup: `anyloc_global_322 [merged]` vs paper column anyloc_global

| setup | ours | paper | diff | exact |
|---|---|---|---|---|
| MM A0 - A2 | 0.6793 | 0.68 | -0.0007 | ✅ |
| MM A2 - A4 | 0.2868 | 0.29 | -0.0032 | ✅ |
| MM C0 - C1 | 0.3311 | 0.33 | +0.0011 | ✅ |
| MM E0 - E1 | 0.2610 | 0.26 | +0.0010 | ✅ |
| MM F0 - F3 | 0.6337 | 0.63 | +0.0037 | ✅ |
| MM G0 - G2 | 0.4951 | 0.50 | -0.0049 | ✅ |
| S3E crater - crater inout | 0.1792 | 0.18 | -0.0008 | ✅ |
| S3E traverse 1 - loops | 0.0846 | 0.08 | +0.0046 | ✅ |
| LL 9 - 10 | 0.4368 | 0.44 | -0.0032 | ✅ |
| LL 15 - 17 | 0.4013 | 0.40 | +0.0013 | ✅ |
| HL3 SPAD dawn - night | 0.2243 | 0.23 | -0.0057 | ❌ |
| HL3 SPAD dawn - noon | 0.6604 | 0.66 | +0.0004 | ✅ |
| HL3 SPAD noon - night | 0.3436 | 0.35 | -0.0064 | ❌ |
| HL3 FLIR dawn - night | 0.3157 | 0.32 | -0.0043 | ✅ |
| HL3 FLIR dawn - noon | 0.6457 | 0.65 | -0.0043 | ✅ |
| HL3 FLIR noon - night | 0.3345 | 0.33 | +0.0045 | ✅ |

| Table III | ours worst (min) | ours worst (hardest setup) | paper worst | ours mean ± std (ddof0/ddof1) | paper mean ± std |
|---|---|---|---|---|---|
| R@1 | 0.312 | 0.386 | 0.310 | 0.557 ± 0.198/0.204 | 0.56 ± 0.21 |
| R@5 | 0.537 | 0.626 | 0.540 | 0.733 ± 0.155/0.160 | 0.73 ± 0.16 |
| R@10 | 0.642 | 0.709 | 0.640 | 0.808 ± 0.126/0.130 | 0.81 ± 0.13 |
| R@100p | 0.001 | 0.001 | 0.001 | 0.042 ± 0.043/0.045 | 0.04 ± 0.04 |

### Per setup: `densevlad_vlfeat_640_drop0 [g-dvlad-vlfeat-640]` vs paper column densevlad

| setup | ours | paper | diff | exact |
|---|---|---|---|---|
| MM A0 - A2 | 0.6540 | 0.69 | -0.0360 | ❌ |
| MM A2 - A4 | 0.3256 | 0.32 | +0.0056 | ❌ |
| MM C0 - C1 | 0.2752 | 0.31 | -0.0348 | ❌ |
| MM E0 - E1 | 0.2015 | 0.30 | -0.0985 | ❌ |
| MM F0 - F3 | 0.5936 | 0.57 | +0.0236 | ❌ |
| MM G0 - G2 | 0.3921 | 0.41 | -0.0179 | ❌ |
| S3E crater - crater inout | 0.3213 | 0.35 | -0.0287 | ❌ |
| S3E traverse 1 - loops | 0.0743 | 0.07 | +0.0043 | ✅ |
| LL 9 - 10 | 0.3167 | 0.41 | -0.0933 | ❌ |
| LL 15 - 17 | 0.2148 | 0.34 | -0.1252 | ❌ |
| HL3 SPAD dawn - night | 0.0169 | 0.02 | -0.0031 | ✅ |
| HL3 SPAD dawn - noon | 0.6730 | 0.80 | -0.1270 | ❌ |
| HL3 SPAD noon - night | 0.0314 | 0.05 | -0.0186 | ❌ |
| HL3 FLIR dawn - night | 0.0406 | 0.17 | -0.1294 | ❌ |
| HL3 FLIR dawn - noon | 0.5561 | 0.67 | -0.1139 | ❌ |
| HL3 FLIR noon - night | 0.0505 | 0.12 | -0.0695 | ❌ |

| Table III | ours worst (min) | ours worst (hardest setup) | paper worst | ours mean ± std (ddof0/ddof1) | paper mean ± std |
|---|---|---|---|---|---|
| R@1 | 0.063 | 0.063 | 0.060 | 0.468 ± 0.259/0.268 | 0.52 ± 0.27 |
| R@5 | 0.129 | 0.189 | 0.230 | 0.601 ± 0.270/0.279 | 0.68 ± 0.24 |
| R@10 | 0.183 | 0.242 | 0.330 | 0.679 ± 0.257/0.266 | 0.76 ± 0.20 |
| R@100p | 0.000 | 0.000 | 0.000 | 0.033 ± 0.046/0.047 | 0.05 ± 0.05 |

## Fig. 6 (BoQ similarities, 3 decimals)

| run | setup | query index | best correct (ours / paper) | false positive = top-1 (ours / paper) |
|---|---|---|---|---|
| `boq [a-exact-kaggleenv]` | MM A0 - A2 | 56 (0-based) | 0.246 / 0.246 | 0.296 / 0.296 |
| `boq [a-exact-kaggleenv]` | MM A0 - A2 | 56 (1-based) | 0.309 / 0.246 | 0.309 / 0.296 |
| `boq [a-exact-kaggleenv]` | S3E traverse 1 - loops | 200 (0-based) | 0.078 / 0.078 | 0.374 / 0.353 |
| `boq [a-exact-kaggleenv]` | S3E traverse 1 - loops | 200 (1-based) | -0.007 / 0.078 | 0.369 / 0.353 |
| `boq [a-exact-kaggleenv]` | HL3 SPAD dawn - night | 93 (0-based) | 0.018 / 0.019 | 0.181 / 0.182 |
| `boq [a-exact-kaggleenv]` | HL3 SPAD dawn - night | 93 (1-based) | – (no hard positive) / 0.019 | 0.175 / 0.182 |
| `boq_bs256 [a-exact-t4-bs]` | MM A0 - A2 | 56 (0-based) | 0.246 / 0.246 | 0.296 / 0.296 |
| `boq_bs256 [a-exact-t4-bs]` | MM A0 - A2 | 56 (1-based) | 0.309 / 0.246 | 0.309 / 0.296 |
| `boq_bs256 [a-exact-t4-bs]` | S3E traverse 1 - loops | 200 (0-based) | 0.078 / 0.078 | 0.374 / 0.353 |
| `boq_bs256 [a-exact-t4-bs]` | S3E traverse 1 - loops | 200 (1-based) | -0.007 / 0.078 | 0.369 / 0.353 |
| `boq_bs256 [a-exact-t4-bs]` | HL3 SPAD dawn - night | 93 (0-based) | 0.018 / 0.019 | 0.181 / 0.182 |
| `boq_bs256 [a-exact-t4-bs]` | HL3 SPAD dawn - night | 93 (1-based) | – (no hard positive) / 0.019 | 0.175 / 0.182 |
| `boq [a-exact-t4]` | MM A0 - A2 | 56 (0-based) | 0.246 / 0.246 | 0.296 / 0.296 |
| `boq [a-exact-t4]` | MM A0 - A2 | 56 (1-based) | 0.309 / 0.246 | 0.309 / 0.296 |
| `boq [a-exact-t4]` | S3E traverse 1 - loops | 200 (0-based) | 0.078 / 0.078 | 0.374 / 0.353 |
| `boq [a-exact-t4]` | S3E traverse 1 - loops | 200 (1-based) | -0.007 / 0.078 | 0.369 / 0.353 |
| `boq [a-exact-t4]` | HL3 SPAD dawn - night | 93 (0-based) | 0.018 / 0.019 | 0.181 / 0.182 |
| `boq [a-exact-t4]` | HL3 SPAD dawn - night | 93 (1-based) | – (no hard positive) / 0.019 | 0.175 / 0.182 |
| `boq [a-exact]` | MM A0 - A2 | 56 (0-based) | 0.246 / 0.246 | 0.296 / 0.296 |
| `boq [a-exact]` | MM A0 - A2 | 56 (1-based) | 0.309 / 0.246 | 0.309 / 0.296 |
| `boq [a-exact]` | S3E traverse 1 - loops | 200 (0-based) | 0.078 / 0.078 | 0.374 / 0.353 |
| `boq [a-exact]` | S3E traverse 1 - loops | 200 (1-based) | -0.006 / 0.078 | 0.369 / 0.353 |
| `boq [a-exact]` | HL3 SPAD dawn - night | 93 (0-based) | 0.018 / 0.019 | 0.181 / 0.182 |
| `boq [a-exact]` | HL3 SPAD dawn - night | 93 (1-based) | – (no hard positive) / 0.019 | 0.175 / 0.182 |
| `boq_bs256 [a-exact]` | MM A0 - A2 | 56 (0-based) | 0.246 / 0.246 | 0.296 / 0.296 |
| `boq_bs256 [a-exact]` | MM A0 - A2 | 56 (1-based) | 0.309 / 0.246 | 0.309 / 0.296 |
| `boq_bs256 [a-exact]` | S3E traverse 1 - loops | 200 (0-based) | 0.078 / 0.078 | 0.374 / 0.353 |
| `boq_bs256 [a-exact]` | S3E traverse 1 - loops | 200 (1-based) | -0.006 / 0.078 | 0.369 / 0.353 |
| `boq_bs256 [a-exact]` | HL3 SPAD dawn - night | 93 (0-based) | 0.018 / 0.019 | 0.181 / 0.182 |
| `boq_bs256 [a-exact]` | HL3 SPAD dawn - night | 93 (1-based) | – (no hard positive) / 0.019 | 0.175 / 0.182 |
| `boq_224 [d-groupa-resolution-t4]` | MM A0 - A2 | 56 (0-based) | 0.168 / 0.246 | 0.246 / 0.296 |
| `boq_224 [d-groupa-resolution-t4]` | MM A0 - A2 | 56 (1-based) | 0.287 / 0.246 | 0.299 / 0.296 |
| `boq_224 [d-groupa-resolution-t4]` | S3E traverse 1 - loops | 200 (0-based) | -0.035 / 0.078 | 0.361 / 0.353 |
| `boq_224 [d-groupa-resolution-t4]` | S3E traverse 1 - loops | 200 (1-based) | -0.066 / 0.078 | 0.349 / 0.353 |
| `boq_224 [d-groupa-resolution-t4]` | HL3 SPAD dawn - night | 93 (0-based) | 0.040 / 0.019 | 0.163 / 0.182 |
| `boq_224 [d-groupa-resolution-t4]` | HL3 SPAD dawn - night | 93 (1-based) | – (no hard positive) / 0.019 | 0.128 / 0.182 |
| `boq_518 [d-groupa-resolution-t4]` | MM A0 - A2 | 56 (0-based) | 0.317 / 0.246 | 0.339 / 0.296 |
| `boq_518 [d-groupa-resolution-t4]` | MM A0 - A2 | 56 (1-based) | 0.395 / 0.246 | 0.395 / 0.296 |
| `boq_518 [d-groupa-resolution-t4]` | S3E traverse 1 - loops | 200 (0-based) | 0.075 / 0.078 | 0.428 / 0.353 |
| `boq_518 [d-groupa-resolution-t4]` | S3E traverse 1 - loops | 200 (1-based) | 0.022 / 0.078 | 0.431 / 0.353 |
| `boq_518 [d-groupa-resolution-t4]` | HL3 SPAD dawn - night | 93 (0-based) | 0.007 / 0.019 | 0.158 / 0.182 |
| `boq_518 [d-groupa-resolution-t4]` | HL3 SPAD dawn - night | 93 (1-based) | – (no hard positive) / 0.019 | 0.162 / 0.182 |

## All descriptor sets (mean over 16 setups)

| descriptor set | AUPRC | R@1 | R@5 | R@10 | R@100p | worst AUPRC |
|---|---|---|---|---|---|---|
| `boq_518 [d-groupa-resolution-t4]` | 0.474 | 0.631 | 0.790 | 0.845 | 0.082 | 0.149 |
| `boq [a-exact]` | 0.469 | 0.635 | 0.778 | 0.843 | 0.077 | 0.165 |
| `boq_bs256 [a-exact]` | 0.469 | 0.635 | 0.778 | 0.843 | 0.077 | 0.165 |
| `boq [a-exact-kaggleenv]` | 0.469 | 0.634 | 0.778 | 0.843 | 0.076 | 0.165 |
| `boq [a-exact-t4]` | 0.469 | 0.634 | 0.778 | 0.843 | 0.076 | 0.165 |
| `boq_bs256 [a-exact-t4-bs]` | 0.469 | 0.634 | 0.778 | 0.843 | 0.076 | 0.165 |
| `megaloc_bs128 [a-exact]` | 0.469 | 0.592 | 0.748 | 0.812 | 0.072 | 0.178 |
| `megaloc [a-exact]` | 0.469 | 0.592 | 0.748 | 0.812 | 0.072 | 0.178 |
| `megaloc [a-exact-t4]` | 0.469 | 0.593 | 0.747 | 0.812 | 0.072 | 0.178 |
| `megaloc_bs128 [a-exact-t4-bs]` | 0.469 | 0.593 | 0.747 | 0.812 | 0.072 | 0.178 |
| `megaloc [a-exact-kaggleenv]` | 0.469 | 0.593 | 0.747 | 0.812 | 0.072 | 0.178 |
| `megaloc_322 [d-groupa-resolution-t4]` | 0.467 | 0.605 | 0.760 | 0.830 | 0.072 | 0.172 |
| `megaloc_224 [d-groupa-resolution-t4]` | 0.449 | 0.594 | 0.757 | 0.816 | 0.070 | 0.160 |
| `boq_224 [d-groupa-resolution-t4]` | 0.442 | 0.613 | 0.768 | 0.826 | 0.072 | 0.131 |
| `anyloc_global_518 [merged]` | 0.424 | 0.549 | 0.740 | 0.807 | 0.066 | 0.098 |
| `anyloc_global_demo1024 [merged]` | 0.423 | 0.575 | 0.737 | 0.808 | 0.065 | 0.110 |
| `salad_518 [b-salad]` | 0.411 | 0.591 | 0.770 | 0.832 | 0.064 | 0.089 |
| `anyloc_unstructured_demo1024 [merged]` | 0.395 | 0.550 | 0.726 | 0.790 | 0.051 | 0.080 |
| `anyloc_global_322 [merged]` | 0.395 | 0.557 | 0.733 | 0.808 | 0.042 | 0.085 |
| `salad_322 [b-salad]` | 0.394 | 0.593 | 0.768 | 0.837 | 0.053 | 0.098 |
| `anyloc_unstructured_518 [merged]` | 0.393 | 0.551 | 0.726 | 0.789 | 0.048 | 0.065 |
| `salad_322_official [b-salad]` | 0.393 | 0.589 | 0.762 | 0.833 | 0.054 | 0.095 |
| `salad_native14 [b-salad]` | 0.384 | 0.579 | 0.754 | 0.821 | 0.059 | 0.097 |
| `densevlad_vlfeat_640_drop0_pcaw256 [local-pca]` | 0.383 | 0.508 | 0.659 | 0.733 | 0.053 | 0.023 |
| `anyloc_unstructured_322 [merged]` | 0.369 | 0.536 | 0.724 | 0.798 | 0.038 | 0.062 |
| `salad_224 [b-salad]` | 0.367 | 0.563 | 0.750 | 0.807 | 0.053 | 0.077 |
| `densevlad_vlfeat_640_drop0_pcaw512 [local-pca]` | 0.361 | 0.516 | 0.674 | 0.744 | 0.060 | 0.022 |
| `densevlad_vlfeat_640_drop0_pcaw128 [local-pca]` | 0.357 | 0.480 | 0.633 | 0.712 | 0.039 | 0.020 |
| `anyloc_global_224 [merged]` | 0.353 | 0.530 | 0.728 | 0.800 | 0.038 | 0.052 |
| `mixvpr_4096 [c-cosplace-mixvpr]` | 0.352 | 0.517 | 0.695 | 0.781 | 0.063 | 0.068 |
| `anyloc_unstructured_224 [merged]` | 0.339 | 0.520 | 0.709 | 0.791 | 0.024 | 0.060 |
| `densevlad_vlfeat_640_drop0_pcaw64 [local-pca]` | 0.317 | 0.464 | 0.610 | 0.682 | 0.029 | 0.023 |
| `dinov3_l16_320 [f-dinov3-320]` | 0.316 | 0.489 | 0.664 | 0.742 | 0.038 | 0.034 |
| `dinov3_b16_224 [c-netvlad-dinov3]` | 0.312 | 0.505 | 0.682 | 0.751 | 0.039 | 0.034 |
| `dinov3_s16_320 [f-dinov3-320]` | 0.311 | 0.501 | 0.669 | 0.732 | 0.039 | 0.055 |
| `dinov3_l16_224 [c-netvlad-dinov3]` | 0.309 | 0.494 | 0.666 | 0.744 | 0.042 | 0.033 |
| `dinov3_b16_320 [f-dinov3-320]` | 0.308 | 0.500 | 0.668 | 0.747 | 0.040 | 0.040 |
| `densevlad_vlfeat_640_drop0_pcaw1024 [local-pca]` | 0.305 | 0.510 | 0.682 | 0.749 | 0.062 | 0.020 |
| `dinov3_l16_512 [c-netvlad-dinov3]` | 0.302 | 0.477 | 0.642 | 0.722 | 0.033 | 0.037 |
| `dinov3_s16_224 [c-netvlad-dinov3]` | 0.298 | 0.501 | 0.664 | 0.736 | 0.034 | 0.049 |
| `mixvpr_512 [c-cosplace-mixvpr]` | 0.296 | 0.504 | 0.671 | 0.757 | 0.041 | 0.033 |
| `densevlad_vlfeat_640_drop0 [g-dvlad-vlfeat-640]` | 0.296 | 0.468 | 0.601 | 0.679 | 0.033 | 0.017 |
| `densevlad_vlfeat_640_keep0 [g-dvlad-vlfeat-640]` | 0.295 | 0.471 | 0.603 | 0.681 | 0.032 | 0.017 |
| `dinov3_b16_512 [c-netvlad-dinov3]` | 0.293 | 0.497 | 0.666 | 0.743 | 0.031 | 0.035 |
| `cosplace_r50_2048_518 [d-cosplace-netvlad-square-t4]` | 0.271 | 0.477 | 0.632 | 0.713 | 0.029 | 0.038 |
| `cosplace_r50_2048_native [c-cosplace-mixvpr]` | 0.269 | 0.475 | 0.635 | 0.698 | 0.030 | 0.040 |
| `cosplace_r50_2048_512 [c-cosplace-mixvpr]` | 0.269 | 0.490 | 0.640 | 0.716 | 0.027 | 0.040 |
| `cosplace_r50_512_native [c-cosplace-mixvpr]` | 0.265 | 0.474 | 0.621 | 0.688 | 0.026 | 0.036 |
| `cosplace_r101_2048_native [c-cosplace-mixvpr]` | 0.259 | 0.467 | 0.623 | 0.710 | 0.030 | 0.032 |
| `cosplace_vgg16_512_native [c-cosplace-mixvpr]` | 0.257 | 0.465 | 0.621 | 0.687 | 0.028 | 0.019 |
| `cosplace_r152_2048_native [c-cosplace-mixvpr]` | 0.256 | 0.463 | 0.625 | 0.700 | 0.027 | 0.047 |
| `netvlad_32768_native [c-netvlad-dinov3]` | 0.253 | 0.459 | 0.621 | 0.706 | 0.030 | 0.020 |
| `mixvpr_128 [c-cosplace-mixvpr]` | 0.250 | 0.444 | 0.639 | 0.720 | 0.037 | 0.031 |
| `netvlad_4096_native [c-netvlad-dinov3]` | 0.250 | 0.462 | 0.637 | 0.715 | 0.029 | 0.035 |
| `netvlad_32768_518 [d-cosplace-netvlad-square-t4]` | 0.242 | 0.427 | 0.608 | 0.714 | 0.028 | 0.022 |
| `cosplace_r50_2048_322 [d-cosplace-netvlad-square-t4]` | 0.242 | 0.459 | 0.631 | 0.699 | 0.023 | 0.037 |
| `netvlad_4096_518 [d-cosplace-netvlad-square-t4]` | 0.233 | 0.419 | 0.609 | 0.715 | 0.026 | 0.031 |
| `netvlad_4096_max640 [c-netvlad-dinov3]` | 0.231 | 0.436 | 0.622 | 0.712 | 0.022 | 0.035 |
| `densevlad_vlfeat_640_drop0_pcaw2048 [local-pca]` | 0.227 | 0.473 | 0.673 | 0.749 | 0.050 | 0.021 |
| `netvlad_4096_322 [d-cosplace-netvlad-square-t4]` | 0.222 | 0.422 | 0.622 | 0.711 | 0.027 | 0.017 |
| `densevlad_16384 [b-densevlad]` | 0.219 | 0.419 | 0.587 | 0.667 | 0.028 | 0.017 |
| `densevlad_vlfeat_640_drop0_pcaw3980 [local-pca]` | 0.126 | 0.338 | 0.559 | 0.666 | 0.022 | 0.025 |
| `densevlad_pca4096 [b-densevlad]` | 0.037 | 0.110 | 0.321 | 0.467 | 0.004 | 0.002 |
