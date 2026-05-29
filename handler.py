import runpod
import base64
import io
from inference import load_models, run_inference

# Load models once at startup (stays warm)
models = load_models()

def handler(job):
    job_input = job["input"]

    # Expect base64-encoded image
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided"}

    image_bytes = base64.b64decode(image_b64)
    
    results = run_inference(models, image_bytes)
    return results

runpod.serverless.start({"handler": handler})