import os
import huggingface_hub
from huggingface_hub.utils import HfHubHTTPError


def get_or_download_model_path():
    env_var_name = "_LLM_SCALER_OMNI_HY3D_PATH"
    model_sub_path_in_repo = "hunyuan3d-paintpbr-v2-1"

    base_download_dir = os.getenv(env_var_name, "/llm/models/Hunyuan3D-2.1")

    if base_download_dir is None:
        raise ValueError(
            f"Environment variable '{env_var_name}' is not set. "
            f"Please set it to the desired base directory for model downloads and storage."
        )

    os.makedirs(base_download_dir, exist_ok=True)

    expected_full_model_path = os.path.join(
        base_download_dir, model_sub_path_in_repo
    )

    if os.path.exists(expected_full_model_path) and os.path.isdir(
        expected_full_model_path
    ):
        print(f"Model found at designated path: {expected_full_model_path}")
        model_path = expected_full_model_path
    else:
        print(
            f"Model not found at {expected_full_model_path}. Attempting to download..."
        )
        try:
            downloaded_repo_root_path = huggingface_hub.snapshot_download(
                repo_id="tencent/Hunyuan3D-2.1",
                allow_patterns=[
                    f"{model_sub_path_in_repo}/*",
                ],
                local_dir=base_download_dir,
                local_dir_use_symlinks=False,
            )

            model_path = os.path.join(
                downloaded_repo_root_path, model_sub_path_in_repo
            )

            if not (os.path.exists(model_path) and os.path.isdir(model_path)):
                raise RuntimeError(
                    f"Download completed, but expected model directory not found at {model_path}. "
                    f"This might indicate an issue with allow_patterns or the remote repository structure."
                )

            print(f"Model downloaded successfully to: {model_path}")

        except HfHubHTTPError as e:
            print(f"Error downloading model from Hugging Face Hub: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during model download: {e}")
            raise

    return model_path


if __name__ == "__main__":

    try:
        final_model_dir = get_or_download_model_path()
        print(f"\nFinal model directory to use: {final_model_dir}")

    except ValueError as e:
        print(f"Configuration error: {e}")
    except RuntimeError as e:
        print(f"Model download/verification error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
