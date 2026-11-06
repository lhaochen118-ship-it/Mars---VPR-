"""Descriptor configurations. Group A ("authors") reproduces the released methods.py exactly; all other
configurations are re-implementations with documented, a-priori choices (the paper does not publish them)."""

CONFIGS = {
    # ---- Group A: exactly as in the authors' methods.py (torch.hub, fp32, their batch sizes) ----
    "boq": dict(kind="authors", method="boq"),                       # 322x322, batch 48
    "megaloc": dict(kind="authors", method="megaloc"),               # 518x518, batch 24
    "boq_bs256": dict(kind="authors", method="boq", batch_size=256),  # same, larger batch (numerics check)
    "megaloc_bs128": dict(kind="authors", method="megaloc", batch_size=128),
    # ---- SALAD (DINOv2-B, GSV-Cities weights) ----
    "salad_322": dict(kind="salad", size=(322, 322)),
    "salad_224": dict(kind="salad", size=(224, 224)),
    "salad_518": dict(kind="salad", size=(518, 518)),
    "salad_native14": dict(kind="salad", size="native14"),
    "salad_322_official": dict(kind="salad", size="pilbilinear322"),  # SALAD eval.py transform
    # ---- AnyLoc (DINOv2 ViT-G/14 L31 value, 32 clusters); one pass yields global + unstructured ----
    "anyloc_224": dict(kind="anyloc", size=(224, 224)),
    "anyloc_322": dict(kind="anyloc", size=(322, 322)),
    "anyloc_518": dict(kind="anyloc", size=(518, 518)),
    "anyloc_demo1024": dict(kind="anyloc", size="anyloc_demo"),
    # ---- CosPlace ----
    "cosplace_r50_2048_native": dict(kind="cosplace", backbone="ResNet50", dim=2048, size="native"),
    "cosplace_r50_2048_512": dict(kind="cosplace", backbone="ResNet50", dim=2048, size=(512, 512)),
    "cosplace_r101_2048_native": dict(kind="cosplace", backbone="ResNet101", dim=2048, size="native"),
    "cosplace_r152_2048_native": dict(kind="cosplace", backbone="ResNet152", dim=2048, size="native"),
    "cosplace_r50_512_native": dict(kind="cosplace", backbone="ResNet50", dim=512, size="native"),
    "cosplace_vgg16_512_native": dict(kind="cosplace", backbone="VGG16", dim=512, size="native"),
    # ---- MixVPR (ResNet50, fixed 320x320 input) ----
    "mixvpr_4096": dict(kind="mixvpr", dim=4096, size=(320, 320)),
    "mixvpr_512": dict(kind="mixvpr", dim=512, size=(320, 320)),
    "mixvpr_128": dict(kind="mixvpr", dim=128, size=(320, 320)),
    # ---- NetVLAD (VGG16, Pitts30K MATLAB weights via hloc) ----
    "netvlad_4096_native": dict(kind="netvlad", dim=4096, size="native"),
    "netvlad_32768_native": dict(kind="netvlad", dim=32768, size="native"),
    "netvlad_4096_max640": dict(kind="netvlad", dim=4096, size="max640"),
    # ---- DINOv3 CLS token (timm mirrors of the official LVD-1689M weights) ----
    "dinov3_s16_224": dict(kind="dinov3", arch="vit_small_patch16_dinov3.lvd1689m", size=(224, 224)),
    "dinov3_b16_224": dict(kind="dinov3", arch="vit_base_patch16_dinov3.lvd1689m", size=(224, 224)),
    "dinov3_b16_512": dict(kind="dinov3", arch="vit_base_patch16_dinov3.lvd1689m", size=(512, 512)),
    "dinov3_l16_224": dict(kind="dinov3", arch="vit_large_patch16_dinov3.lvd1689m", size=(224, 224)),
    "dinov3_l16_512": dict(kind="dinov3", arch="vit_large_patch16_dinov3.lvd1689m", size=(512, 512)),
    # ---- DenseVLAD (re-implementation, GPU dense RootSIFT; PCA-4096 variant written by the post step) ----
    "densevlad_16384": dict(kind="densevlad", size="gray640"),
}

# relative per-image GPU cost (for load balancing only)
COST = {"authors": 1.0, "salad": 1.0, "anyloc": 12.0, "cosplace": 0.5, "mixvpr": 0.2, "netvlad": 1.5, "dinov3": 0.6, "densevlad": 2.0}


def outputs_of(name):
    """Descriptor set names produced by a configuration."""
    c = CONFIGS[name]
    if c["kind"] == "anyloc":
        res = name.split("_", 1)[1]
        return [f"anyloc_global_{res}", f"anyloc_unstructured_{res}"]
    return [name]
