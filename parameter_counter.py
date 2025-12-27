import torch
from models.Posmer_7cls import pyramid_mamba_expr2


def count_parameters(model):
    table = []
    total_params = 0
    trainable_params = 0

    print(f"\n{'=' * 60}")
    print(f"{'Module':<30} | {'Total Params':<12} | {'Trainable':<10}")
    print(f"{'-' * 60}")

    # Iterate through the main submodules of the specific architecture
    # Based on pyramid_trans_expr2 in Posmer_7cls.py
    modules_to_check = [
        ('Face Backbone (MobileFaceNet)', model.face_backbone),
        ('Visual Backbone (IR50)', model.visual_backbone),
        ('Fusion Stage 1', model.fusion_stage1),
        ('Fusion Stage 2', model.fusion_stage2),
        ('Fusion Stage 3', model.fusion_stage3),
        ('ViM Backend (Classifier)', model.vim_backend),
        ('Projections & Downsamples', torch.nn.Sequential(
            model.downsample1, model.downsample2, model.downsample3,
            model.face_proj, model.proj_stage1, model.proj_stage2, model.proj_stage3
        ))
    ]

    for name, submodule in modules_to_check:
        params = sum(p.numel() for p in submodule.parameters())
        trainable = sum(p.numel() for p in submodule.parameters() if p.requires_grad)

        print(f"{name:<30} | {params:<12,} | {trainable:<10,}")

        total_params += params
        trainable_params += trainable

    print(f"{'=' * 60}")

    # Calculate global totals (double check against the sum above)
    global_total = sum(p.numel() for p in model.parameters())
    global_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'Overall Summary':^30}")
    print(f"{'-' * 30}")
    print(f"Total Parameters:     {global_total:,}")
    print(f"Trainable Parameters: {global_trainable:,}")
    print(f"Non-Trainable Params: {global_total - global_trainable:,}")
    print(f"{'=' * 60}\n")

    return global_total


if __name__ == '__main__':
    # Initialize the model exactly as in main.py
    print("Initializing Model...")
    model = pyramid_mamba_expr2(
        img_size=224,
        num_classes=7,
        ir50_path=None,  # No need to load weights for counting
        facenet_path=None  # No need to load weights for counting
    )

    count_parameters(model)