import requests
import websocket  # pip install websocket-client
import json
import time
import uuid
import statistics
import sys
import copy  # For deep copying workflow_json

# ==============================================================================
#                 ComfyUI Benchmark Configuration
# ==============================================================================

# --- ComfyUI Server Details ---
COMFYUI_HOST = "127.0.0.1"  # Your ComfyUI server IP address
COMFYUI_PORT = "8188"  # Your ComfyUI server port
SERVER_ADDRESS = f"{COMFYUI_HOST}:{COMFYUI_PORT}"
HTTP_PROMPT_URL = f"http://{SERVER_ADDRESS}/prompt"
WS_BASE_URL = f"ws://{SERVER_ADDRESS}/ws"

# --- Benchmark Parameters ---
NUM_WARMUP = 1  # Number of warm-up requests
NUM_TRIALS = 3  # Number of benchmark trials for timed results

# --- Workflow Specific Parameters (can be overridden or modified dynamically within the function) ---
DEFAULT_SAMPLER_SEED = 898471028164125
DEFAULT_STEPS = 20
DEFAULT_CFG = 5
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 704
DEFAULT_LENGTH = 121  # Number of frames for video

# --- ComfyUI Workflow JSON ---
wan22_ti2v_5b_workflow_json = {
    "3": {
        "inputs": {
            "seed": DEFAULT_SAMPLER_SEED,
            "steps": DEFAULT_STEPS,
            "cfg": DEFAULT_CFG,
            "sampler_name": "uni_pc",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["48", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["55", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "6": {
        "inputs": {
            "text": "Low contrast. In a retro 1970s-style subway station, a street musician plays in dim colors and rough textures. He wears an old jacket, playing guitar with focus. Commuters hurry by, and a small crowd gathers to listen. The camera slowly moves right, capturing the blend of music and city noise, with old subway signs and mottled walls in the background.",
            "clip": ["38", 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
    },
    "7": {
        "inputs": {
            "text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "clip": ["38", 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
    },
    "8": {
        "inputs": {"samples": ["3", 0], "vae": ["39", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "37": {
        "inputs": {
            "unet_name": "wan2.2_ti2v_5B_fp16.safetensors",
            "weight_dtype": "default",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"},
    },
    "38": {
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default",
        },
        "class_type": "CLIPLoader",
        "_meta": {"title": "Load CLIP"},
    },
    "39": {
        "inputs": {"vae_name": "wan2.2_vae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "48": {
        "inputs": {"shift": 8, "model": ["37", 0]},
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "ModelSamplingSD3"},
    },
    "55": {
        "inputs": {
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
            "length": DEFAULT_LENGTH,
            "batch_size": 1,
            "vae": ["39", 0],
        },
        "class_type": "Wan22ImageToVideoLatent",
        "_meta": {"title": "Wan22ImageToVideoLatent"},
    },
    "57": {
        "inputs": {"fps": 24, "images": ["8", 0]},
        "class_type": "CreateVideo",
        "_meta": {"title": "Create Video"},
    },
    "58": {
        "inputs": {
            "filename_prefix": "video/ComfyUI",
            "format": "auto",
            "codec": "auto",
            "video": ["57", 0],
        },
        "class_type": "SaveVideo",
        "_meta": {"title": "Save Video"},
    },
}

# ==============================================================================
#                 ComfyUI API Helper Functions
# ==============================================================================


def queue_prompt(workflow, client_id):
    """
    Submits a ComfyUI workflow to the API.
    """
    p = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    try:
        response = requests.post(HTTP_PROMPT_URL, data=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        prompt_info = response.json()
        print(f"  Prompt submitted. Prompt ID: {prompt_info['prompt_id']}")
        return prompt_info
    except requests.exceptions.ConnectionError:
        print(
            f"Error: Could not connect to ComfyUI server at {HTTP_PROMPT_URL}. Is it running?"
        )
        return None
    except requests.exceptions.HTTPError as e:
        print(f"Error submitting prompt: {e}")
        print(f"Response: {response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during prompt submission: {e}")
        return None


def wait_for_completion(ws_url, client_id, prompt_id):
    """
    Listens via WebSocket for ComfyUI task completion.
    """
    ws = None
    try:
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        # print(f"  Connected to WebSocket: {ws_url}") # Too verbose for each run

        queue_empty_for_client = False
        prompt_finished_executing = False

        # Keep track of active prompt_ids seen on this client
        active_prompts = {prompt_id}

        while not (queue_empty_for_client and prompt_finished_executing):
            message = ws.recv()
            if not isinstance(message, str):
                continue  # Skip non-string messages (e.g., binary)

            message_obj = json.loads(message)

            if message_obj.get("type") == "status":
                data = message_obj["data"]
                if "sid" in data and data["sid"] == client_id:
                    if data["status"]["exec_info"]["queue_remaining"] == 0:
                        queue_empty_for_client = True
                        # print(f"  [WS Status] Queue empty for client {client_id}.")
                    else:
                        queue_empty_for_client = False

            elif message_obj.get("type") == "executing":
                data = message_obj["data"]

                # If this message is for our prompt_id
                if data.get("prompt_id") == prompt_id:
                    if data.get("node") is None:
                        # Our prompt has finished execution (node is null)
                        prompt_finished_executing = True
                        # print(f"  [WS Executing] Prompt {prompt_id} finished execution.")
                    else:
                        pass  # Still executing a node
                # For other prompts on this client_id, just track completion
                elif data.get("client_id") == client_id:
                    if (
                        data.get("node") is None
                        and data.get("prompt_id") in active_prompts
                    ):
                        active_prompts.remove(data.get("prompt_id"))  # Mark as done
                    elif data.get("node") is not None:
                        active_prompts.add(data.get("prompt_id"))

            # Exit condition: if our specific prompt_id is no longer active AND queue is empty
            if prompt_id not in active_prompts and queue_empty_for_client:
                prompt_finished_executing = True
                break

        return True

    except websocket._exceptions.WebSocketConnectionClosedException:
        print("  Websocket connection closed unexpectedly.")
        return False
    except Exception as e:
        print(f"  Error during WebSocket communication: {e}")
        return False
    finally:
        if ws:
            ws.close()
            # print("  WebSocket connection closed.") # Too verbose for each run


# ==============================================================================
#                 Benchmark Function for Text-to-Video Workflow
# ==============================================================================


def benchmark_t2v_model(workflow_json_template, model_name="ComfyUI Workflow"):
    """
    Benchmarks a ComfyUI Text-to-Video workflow by submitting prompts via API
    and listening for completion via WebSocket.

    Args:
        workflow_json_template (dict): The base ComfyUI workflow JSON.
        model_name (str): A descriptive name for the model/workflow being benchmarked.
    """

    print(f"\n--- Starting Benchmark for ComfyUI Workflow: {model_name} ---")
    print(f"ComfyUI Server: {SERVER_ADDRESS}")
    print(f"Warm-up requests: {NUM_WARMUP}, Trial requests: {NUM_TRIALS}")

    # --- Warm-up requests ---
    print(f"\n--- Starting Warm-up ({NUM_WARMUP} requests) ---")
    print(
        "These requests are to let the service initialize and are not timed for results."
    )
    for i in range(NUM_WARMUP):
        print(f"  Warm-up request {i+1}/{NUM_WARMUP}...")

        # Deep copy the workflow template to avoid modifying the original
        workflow = copy.deepcopy(workflow_json_template)

        # Optionally, modify seed for each warm-up run
        # For a truly consistent benchmark, you might fix the seed,
        # but varying it ensures all parts of the graph are exercised if dependencies exist.
        workflow["3"]["inputs"]["seed"] = DEFAULT_SAMPLER_SEED + i

        client_id = str(uuid.uuid4())
        ws_url = f"{WS_BASE_URL}?clientId={client_id}"

        start_time = time.perf_counter()
        prompt_response = queue_prompt(workflow, client_id)
        if not prompt_response:
            print(
                "  Warm-up request failed during prompt submission. Exiting benchmark."
            )
            sys.exit(1)

        if not wait_for_completion(ws_url, client_id, prompt_response["prompt_id"]):
            print("  Warm-up request failed during WebSocket wait. Exiting benchmark.")
            sys.exit(1)

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"  Warm-up request successful. Latency: {duration:.4f} seconds.")

    # --- Benchmark Trials ---
    print(f"\n--- Starting Benchmark Trials ({NUM_TRIALS} requests) ---")
    latencies = []
    for i in range(NUM_TRIALS):
        print(f"  Trial request {i+1}/{NUM_TRIALS}...")

        # Deep copy for each trial
        workflow = copy.deepcopy(workflow_json_template)

        # Best practice for benchmarking: ensure new unique seeds for each generative run
        # if the output needs to be different, but for raw speed, it often doesn't matter.
        workflow["3"]["inputs"]["seed"] = DEFAULT_SAMPLER_SEED + NUM_WARMUP + i

        client_id = str(uuid.uuid4())
        ws_url = f"{WS_BASE_URL}?clientId={client_id}"

        start_time = time.perf_counter()
        prompt_response = queue_prompt(workflow, client_id)
        if not prompt_response:
            print(
                "  Trial request failed during prompt submission. Stopping benchmark."
            )
            break

        if not wait_for_completion(ws_url, client_id, prompt_response["prompt_id"]):
            print("  Trial request failed during WebSocket wait. Stopping benchmark.")
            break

        end_time = time.perf_counter()
        duration = end_time - start_time
        latencies.append(duration)
        print(f"  Trial successful. Latency: {duration:.4f} seconds.")

    # --- Results ---
    print("\n--- Benchmark Results ---")
    if not latencies:
        print("No successful trials completed to report statistics.")
    else:
        print(f"Total successful trials: {len(latencies)}")
        print(f"Configuration:")
        print(f"  ComfyUI Server: {SERVER_ADDRESS}")
        print(f"  Workflow Name: {model_name}")
        print(f"  Prompt (Positive): '{workflow_json_template['6']['inputs']['text']}'")
        print(
            f"  Image/Video Resolution: {workflow_json_template['55']['inputs']['width']}x{workflow_json_template['55']['inputs']['height']}"
        )
        print(
            f"  Video Length (frames): {workflow_json_template['55']['inputs']['length']}"
        )
        print(f"  Sampler Steps: {workflow_json_template['3']['inputs']['steps']}")
        print(f"  Sampler CFG: {workflow_json_template['3']['inputs']['cfg']}")
        print(f"  Warm-up requests: {NUM_WARMUP}")
        print(f"  Trial requests: {NUM_TRIALS}")

        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_latency = statistics.mean(latencies)

        if len(latencies) > 1:
            std_dev_latency = statistics.stdev(latencies)
            print(f"Average Latency:   {avg_latency:.4f} seconds")
            print(f"Std Deviation:     {std_dev_latency:.4f} seconds")
        else:
            print(
                f"Latency:           {avg_latency:.4f} seconds (only one trial completed)"
            )

    print(f"\nBenchmark for ComfyUI Workflow: '{model_name}' finished.")

    return avg_latency


# ==============================================================================
#                              Main Execution
# ==============================================================================

if __name__ == "__main__":

    # Run benchmark for the ComfyUI workflow
    benchmark_t2v_model(
        wan22_ti2v_5b_workflow_json, model_name="Wan2.2_ti2v_5B Workflow"
    )

