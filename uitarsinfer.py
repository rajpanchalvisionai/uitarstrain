import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForImageTextToText
import os

# --- Configuration ---
MODEL_ID = "ByteDance-Seed/UI-TARS-2B-SFT"
# --- Choose ONE image source ---
IMAGE_SOURCE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/im-button.png"
# IMAGE_SOURCE = "./path/to/your/screenshot.png" # Replace with your local path

PROMPT = "What is shown in this image?"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Model and Processor ---
print(f"Loading model and processor for {MODEL_ID}...")
try:
    # Add trust_remote_code=True for both processor and model loading
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True  # Also add here for the model architecture
    ).to(device)
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model/processor: {e}")
    print("Please ensure you have an internet connection, the model ID is correct, and try updating 'transformers'.")
    exit()

# --- Load and Prepare Image ---
print(f"Loading image from: {IMAGE_SOURCE}")
try:
    if IMAGE_SOURCE.startswith("http://") or IMAGE_SOURCE.startswith("https://"):
        image = Image.open(requests.get(IMAGE_SOURCE, stream=True).raw).convert("RGB")
    elif os.path.exists(IMAGE_SOURCE):
        image = Image.open(IMAGE_SOURCE).convert("RGB")
    else:
        raise FileNotFoundError(f"Image source not found at: {IMAGE_SOURCE}")
    print("Image loaded successfully.")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# --- Prepare Text Input (Messages Format) ---
messages = [
    {"role": "user", "content": PROMPT},
]
print(f"Using prompt: \"{PROMPT}\"")

# --- Perform Inference ---
print("Preparing inputs for the model...")
try:
    inputs = processor(text=messages, images=image, return_tensors="pt").to(device)
    # No need to manually cast to float16 here if model is already loaded in float16
    # The processor should handle input types appropriately based on model config
except Exception as e:
    print(f"Error during input processing: {e}")
    exit()

print("Generating response...")
try:
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
        )
    print("Generation complete.")

    print("Decoding response...")
    decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)
    raw_output = decoded_output[0].strip()

    print("\n--- Raw Decoded Output ---")
    print(raw_output)

    # Attempt to extract just the assistant's response
    # This heuristic might need adjustment.
    # For chat models, the response is usually what comes after the last user prompt.
    # The model's internal prompting might include special tokens or structures.
    # A common way chat models are prompted:
    # "<|user|>\n{PROMPT}<|end|>\n<|assistant|>\n{RESPONSE}<|end|>"
    # Or for this model, it might be closer to the messages structure directly.
    # Let's assume the model's output might include the prompt.
    
    # Find the content of the last message in the input `messages`
    last_user_content = ""
    if messages and isinstance(messages, list) and messages[-1].get("role") == "user":
        last_user_content = messages[-1].get("content", "")

    response_part = raw_output
    if last_user_content and last_user_content in raw_output:
        # Take text after the last occurrence of the user's prompt content
        # This is a simple heuristic and might need refinement
        last_occurrence_index = raw_output.rfind(last_user_content)
        if last_occurrence_index != -1:
            potential_response = raw_output[last_occurrence_index + len(last_user_content):].strip()
            # Further clean-up if known assistant markers are present
            # e.g. if response starts with "assistant\n", remove it.
            # For now, this is a basic extraction.
            response_part = potential_response if potential_response else raw_output

    print("\n--- Extracted Response ---")
    print(response_part if response_part else "Could not reliably extract response part from raw output.")

except Exception as e:
    print(f"Error during generation or decoding: {e}")
    if "out of memory" in str(e).lower():
        print("CUDA Out of Memory error. Try using a GPU with more VRAM or using CPU (will be very slow).")

print("\n--- Script Finished ---")