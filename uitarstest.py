import torch
# use transformers 4.49.0
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import os

model_name = "ByteDance-Seed/UI-TARS-2B-SFT"

# Load processor and model
# Use AutoProcessor for multimodal components
# Use the specific model class as indicated in config.json
try:
    print("Attempting to load processor and model...")
    processor = AutoProcessor.from_pretrained(model_name)
    # Load the model using the specific class
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,  # Use float32 if you're on CPU or get NaNs
    )
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    print("-" * 50)
    print("TROUBLESHOOTING LOADING:")
    print("This error indicates an incompatibility between the model's configuration files")
    print("on Hugging Face Hub and your transformers library version.")
    print("We previously narrowed this down. If this error persists, double-check transformers version (should be 4.48.0) and cache.")
    print("-" * 50)
    exit()


# Load an image from a URL or local path
# Example using a placeholder URL, replace with your image path or URL
# image_url = "https://www.gstatic.com/webp/gallery2/1.jpg" # Replace with your image URL or local path
image_url = "" # Replace with your image URL or local path
image_path = "/home/administrator/flash/UI-tars/vncscreen.png" # Uncomment and use this for a local file

try:
    print(f"Loading image from: {image_url}")
    # Check if it's a URL or a local path
    if image_url.startswith("http://") or image_url.startswith("https://"):
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    elif os.path.exists(image_path): # Uncomment and use this for local path
        image = Image.open(image_path).convert("RGB")
    else:
         print(f"Error: Image path/URL seems invalid: {image_url}")
         exit()
    print("Image loaded successfully.")

except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Prepare input using the conversation format similar to Qwen2-VL
# The 'content' is a list containing dictionaries for each modality (image, text)
print("Preparing conversation input...")
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image # Pass the PIL Image object directly
            },
            {
                "type": "text",
                # Simplified prompt
                "text": "Describe this image."
                # Or try: "What is in this picture?"
            }
        ]
    }
]
print("Conversation input prepared.")

# Process the conversation using apply_chat_template
# This method formats the input correctly for the model's chat structure
try:
    print("Applying chat template...")
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True, # Add the prompt for the model to start generating
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    print("Chat template applied successfully.")
except Exception as e:
     print(f"Error applying chat template: {e}")
     print("This could happen if the processor didn't load correctly or if the conversation format is incorrect.")
     exit()


# Generate
print("Generating response...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        # Adjust generation parameters
        do_sample=False, # Disable sampling for less randomness
        # temperature=0.1, # Low temperature
        # top_p=0.7,       # Lower top_p
        pad_token_id=processor.tokenizer.eos_token_id
    )
print("Response generated.")

# Decode the generated output
# We need to slice the output_ids to remove the input part
print("Decoding output...")
# Find the length of the input tokens for slicing
input_length = inputs.input_ids[0].shape[0]
# Slice to get only the generated tokens
generated_ids = [output_ids[0, input_length:]]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print("Output decoded.")

# Print the decoded output
print("\n--- Generated Output ---")
print(output_text[0])
print("------------------------")