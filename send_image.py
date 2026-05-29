import base64
import requests
import time

ENDPOINT_ID = ""
API_KEY = ""
IMAGE_PATH = "test.jpeg"   # change to your actual image filename

with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

print("Submitting job...")
res = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={"input": {"image": image_b64}}
)
print("Response:", res.json())
job_id = res.json()["id"]
print(f"Job ID: {job_id}")

print("Waiting for result...")
while True:
    status = requests.get(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}",
        headers={"Authorization": f"Bearer {API_KEY}"}
    ).json()

    print(f"Status: {status['status']}")

    if status["status"] == "COMPLETED":
        output = status["output"]

        with open("output.png", "wb") as f:
            f.write(base64.b64decode(output["image_b64"]))
        print("Saved output.png")

        with open("output.wav", "wb") as f:
            f.write(base64.b64decode(output["audio_b64"]))
        print("Saved output.wav")

        with open("output_poem.txt", "w") as f:
            f.write(output["literature"])
        print("Saved output_poem.txt")

        print("\n--- Poem ---")
        print(output["literature"])
        print("\nDone! Check output.png, output.wav, and output_poem.txt")
        break

    elif status["status"] == "FAILED":
        print("Failed:", status)
        break

    time.sleep(10)