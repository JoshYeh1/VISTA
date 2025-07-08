#Tests AI model on local device by captioning imported image
import base64
from PIL import Image
import io
from ollama import Client
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument()



args = parser.parse_args()
start_time = time.time()
image_load_start = time.time()
# Load image
image_path = "image.png"
image = Image.open(image_path).convert("RGB")
resized_image = image.resize((256, 256))

# Convert to base64-encoded PNG
buffered = io.BytesIO()
resized_image.save(buffered, format="PNG")
image_bytes = buffered.getvalue()
image_b64 = base64.b64encode(image_bytes).decode('utf-8')

image_load_time = time.time() - image_load_start
print(f"Image load Duration: {image_load_time:.2f} seconds")
caption_load_start = time.time()

# Initialize Ollama client
client = Client()

# Generate caption
response = client.generate(
    model="qwen2.5vl:latest",  # use llava or llava-phi3 or qwen2.5vl:latest
    prompt="Briefly describe this image for someone who is visually impaired.",
    images=[image_b64],
    options={"num_predict": 50}  
)

caption_load_time = time.time() - caption_load_start

# Output result
print("Caption:", response["response"])
print(f"Duration of Caption: {caption_load_time:.2f} seconds")
duration = time.time() - start_time
print(f"Duration of Program: {duration:.2f} seconds")