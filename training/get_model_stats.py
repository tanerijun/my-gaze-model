import argparse

import timm
from mmengine.analysis import get_model_complexity_info

# Import local backbones
from src.models.backbone_lowformer import LowFormerBackbone
from src.models.backbone_mobileone import MobileOneBackbone
from src.models.backbone_resnet import ResNetBackbone


def get_stats(model_name, input_shape=(3, 224, 224)):
    """
    Prints the FLOPs and parameter count for a given model.
    """
    model = None
    try:
        print(f"Attempting to build model: {model_name}")
        if model_name.startswith("lowformer"):
            model = LowFormerBackbone(arch=model_name, pretrained=False)
        elif model_name.startswith("resnet"):
            model = ResNetBackbone(arch=model_name, pretrained=False)
        elif model_name.startswith("mobileone"):
            model = MobileOneBackbone(arch=model_name, pretrained=False)
        else:
            # Assume it's a timm model
            print("Assuming it's a timm model...")
            model = timm.create_model(model_name, pretrained=False)

        model.eval()

        # Get complexity information
        analysis_results = get_model_complexity_info(model, input_shape)

        flops = analysis_results["flops_str"]
        params = analysis_results["params_str"]

        print(f"--- Stats for: {model_name} ---")
        print(f"  Input Shape: {input_shape}")
        print(f"  FLOPs: {flops}")
        print(f"  Parameters: {params}")
        print("--------------------" + "-" * len(model_name))

    except Exception as e:
        print(f"Could not get stats for {model_name}.")
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get FLOPs and parameter count for a model."
    )
    parser.add_argument(
        "model_name", type=str, help="The name of the timm or local model to analyze."
    )
    args = parser.parse_args()
    get_stats(args.model_name)
