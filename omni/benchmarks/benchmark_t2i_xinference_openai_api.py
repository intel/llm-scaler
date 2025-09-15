import openai
import time
import statistics
import sys

BASE_URL = "http://localhost:9997"
from xinference.client import Client

x_client = Client(BASE_URL)


# MODEL_NAME = "sd3.5-medium"

PROMPT = "A Shiba Inu chasing butterflies on a sunny grassy field, cartoon style, with vibrant colors."
IMAGE_SIZE = "1024x1024"
NUM_IMAGES = 1

NUM_WARMUP = 3
NUM_TRIALS = 5

# --- Initialize OpenAI Client ---
API_KEY = "not-really-a-secret-key"
try:
    OPENAI_URL = BASE_URL + "/v1"
    client = openai.Client(api_key=API_KEY, base_url=OPENAI_URL)
    print(f"Initialized OpenAI client for base URL: {OPENAI_URL}")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure your 'openai' package is up to date and base_url is correct.")
    sys.exit(1)


def benchmark_t2i_model(model_uid, model_path=None, relaunch=True):
    if relaunch:
        model_uid = x_client.launch_model(
            model_uid=model_uid,
            model_name=model_uid,
            model_engine="transformers",
            model_type="image",
            model_path=model_path,
        )

    # --- Warm-up requests ---
    print(f"\n--- Starting Warm-up ({NUM_WARMUP} requests) ---")
    print("These requests are to let the service initialize and are not timed for results.")
    for i in range(NUM_WARMUP):
        print(f"Warm-up request {i+1}/{NUM_WARMUP}...")
        start_time = time.perf_counter()
        try:
            response = client.images.generate(
                model=model_uid,
                prompt=PROMPT,
                size=IMAGE_SIZE,
                n=NUM_IMAGES,
            )
            end_time = time.perf_counter()
            duration = end_time - start_time
            print(f"  Warm-up request successful. Latency: {duration:.4f} seconds.")
            # Optionally, you can print parts of the response, e.g., print(response.data[0].url)
        except openai.APIError as e:
            print(f"  Warm-up request failed with API error: {e}")
            print(
                "  Please check if the local service is running and configured correctly."
            )
            print(
                f"  Model '{model_uid}' might not be available or parameters like '{IMAGE_SIZE}'/'steps' are unsupported."
            )
            sys.exit(1)
        except Exception as e:
            print(f"  Warm-up request failed with a general error: {e}")
            print(
                "  Perhaps the local service is not reachable or there's a network issue."
            )
            sys.exit(1)

    # --- Benchmark Trials ---
    print(f"\n--- Starting Benchmark Trials ({NUM_TRIALS} requests) ---")
    latencies = []
    for i in range(NUM_TRIALS):
        print(f"Trial request {i+1}/{NUM_TRIALS}...")
        start_time = time.perf_counter()
        try:
            response = client.images.generate(
                model=model_uid,
                prompt=PROMPT,
                size=IMAGE_SIZE,
                n=NUM_IMAGES,
            )
            end_time = time.perf_counter()
            duration = end_time - start_time
            latencies.append(duration)
            print(f"  Trial successful. Latency: {duration:.4f} seconds.")
            # Optionally, you can print parts of the response, e.g., print(response.data[0].url)
        except openai.APIError as e:
            print(f"  Trial request failed with API error: {e}")
            print("  Encountered an error during trials. Stopping benchmark.")
            break  # Stop if an error occurs to avoid skewing results
        except Exception as e:
            print(f"  Trial request failed with a general error: {e}")
            print("  Encountered an error during trials. Stopping benchmark.")
            break

    # --- Results ---
    print("\n--- Benchmark Results ---")
    if not latencies:
        print("No successful trials completed to report statistics.")
    else:
        print(f"Total successful trials: {len(latencies)}")
        print(f"Configuration:")
        print(f"  Model: {model_uid}")
        print(f"  Prompt: '{PROMPT}'")
        print(f"  Image Size: {IMAGE_SIZE}")
        print(f"  Number of Images per request: {NUM_IMAGES}")

        print(f"  Warm-up requests: {NUM_WARMUP}")
        print(f"  Trial requests: {NUM_TRIALS}")

        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_latency = statistics.mean(latencies)

        # Standard deviation requires at least 2 data points
        if len(latencies) > 1:
            std_dev_latency = statistics.stdev(latencies)
            # print(f"Minimum Latency:   {min_latency:.4f} seconds")
            # print(f"Maximum Latency:   {max_latency:.4f} seconds")
            print(f"Average Latency:   {avg_latency:.4f} seconds")
            print(f"Std Deviation:     {std_dev_latency:.4f} seconds")
        else:
            print(
                f"Latency:           {avg_latency:.4f} seconds (only one trial completed)"
            )

    if relaunch:
        x_client.terminate_model(model_uid=model_uid)
    print(f"\nBenchmark for {model_uid} finished.")

    return avg_latency

benchmark_t2i_model(
    model_uid="sd3.5-medium",
    model_path="/llm/models/stable-diffusion-3.5-medium/",
)
benchmark_t2i_model(
    model_uid="FLUX.1-dev",
    model_path="/llm/models/FLUX.1-dev/",
)
benchmark_t2i_model(
    model_uid="HunyuanDiT-v1.2",
    model_path="/llm/models/HunyuanDiT-v1.2-Diffusers/",
)
