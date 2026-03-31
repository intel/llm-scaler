import argparse
import os

import torch
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser(description="Extract UNet and VAE weights from a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="/llm/models/comfyui/comfyui_models/checkpoints/WAN/wan2.2-t2v-rapid-aio-v8.1.safetensors",
        help="Path to source checkpoint (.safetensors or .ckpt).",
    )
    parser.add_argument(
        "--output-unet",
        default="./wan2.2-t2v-rapid-aio-v8.1-unet.safetensors",
        help="Path to output UNet safetensors file.",
    )
    parser.add_argument(
        "--output-vae",
        default="./wan2.2-t2v-rapid-aio-v8.1-vae.safetensors",
        help="Path to output VAE safetensors file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    full_checkpoint_path = args.checkpoint
    output_unet_path = args.output_unet
    output_vae_path = args.output_vae

    print(f"Loading checkpoint from: {full_checkpoint_path}")

    try:
        if full_checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(full_checkpoint_path)
        elif full_checkpoint_path.endswith(".ckpt"):
            state_dict = torch.load(full_checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        else:
            raise ValueError("Unsupported file extension. Must be .safetensors or .ckpt")

        unet_state_dict = {}
        unet_prefix = "model.diffusion_model."

        vae_state_dict = {}
        vae_prefix_sd1 = "first_stage_model."
        vae_prefix_sdxl = "vae."

        found_unet_components = False
        for k, v in state_dict.items():
            if k.startswith(unet_prefix):
                clean_k = k[len(unet_prefix):]
                unet_state_dict[clean_k] = v
                found_unet_components = True

        if not found_unet_components:
            print(f"Warning: No UNet keys found in {full_checkpoint_path} with prefix '{unet_prefix}'.")
            print("This might not be a standard Stable Diffusion checkpoint or the prefix is wrong.")
            print("No UNet file will be saved.")
        else:
            os.makedirs(os.path.dirname(output_unet_path) or ".", exist_ok=True)
            save_file(unet_state_dict, output_unet_path)
            print(f"Successfully extracted UNet and saved to: {output_unet_path}")
            print(f"Number of UNet parameters extracted: {len(unet_state_dict)}.")

        found_vae_components = False
        for k, v in state_dict.items():
            if k.startswith(vae_prefix_sd1):
                clean_k = k[len(vae_prefix_sd1):]
                vae_state_dict[clean_k] = v
                found_vae_components = True
            elif k.startswith(vae_prefix_sdxl):
                clean_k = k[len(vae_prefix_sdxl):]
                vae_state_dict[clean_k] = v
                found_vae_components = True

        if not found_vae_components:
            print(f"Warning: No VAE keys found in {full_checkpoint_path} with prefixes '{vae_prefix_sd1}' or '{vae_prefix_sdxl}'.")
            print("This might not be a standard Stable Diffusion checkpoint or the prefixes are wrong.")
            print("No VAE file will be saved.")
        else:
            os.makedirs(os.path.dirname(output_vae_path) or ".", exist_ok=True)
            save_file(vae_state_dict, output_vae_path)
            print(f"Successfully extracted VAE and saved to: {output_vae_path}")
            print(f"Number of VAE parameters extracted: {len(vae_state_dict)}.")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {full_checkpoint_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the checkpoint path is correct and the file is accessible.")


if __name__ == "__main__":
    main()
