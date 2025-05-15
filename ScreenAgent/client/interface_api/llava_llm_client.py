import requests
import base64
from PIL import Image
from io import BytesIO
# import threading # Still removed

from interface_api.conversation import conv_templates

DEFAULT_IMAGE_TOKEN = "<image>"

def encode_image_to_base64_with_pil(image:Image):
    buffered = BytesIO()
    # Ensure image is in a format supported by BytesIO. PNG is a safe bet.
    image_format = image.format if image.format else 'PNG'
    try:
        image.save(buffered, format=image_format)
    except Exception as e:
        # Fallback to PNG if original format save fails
        print(f"Warning: Failed to save image in format {image_format}, falling back to PNG. Error: {e}")
        try:
            buffered = BytesIO() # Reset buffer
            image.save(buffered, format="PNG")
            image_format = "PNG"
        except Exception as e_png:
            print(f"Error: Failed to save image as PNG. Error: {e_png}")
            return None # Cannot encode image

    return base64.b64encode(buffered.getvalue()).decode()


class LanguageModelClient:
    # Constructor signature is (string, dict) based on our run_controller.py fix
    def __init__(self, server_name, llm_server_config):

        # Access the specific model's config using the server_name key
        model_specific_config = llm_server_config.get(server_name, {})
        if not model_specific_config:
             raise ValueError(f"Configuration for server '{server_name}' not found in llm_api config.")

        self.model_name = model_specific_config.get("model_name", server_name) # Use server_name as default model_name
        # Use the target_url from the specific model config
        self.target_url = model_specific_config.get("target_url", "http://localhost:40000/worker_generate")

        # Access common settings directly from the full llm_server_config dict
        self.temperature = llm_server_config.get("temperature", 1.0)
        self.top_p = llm_server_config.get("top_p", 0.9)
        self.max_tokens = llm_server_config.get("max_tokens", 500)

        print(f"Initialized {server_name} client:")
        print(f"  Model: {self.model_name}")
        print(f"  URL: {self.target_url}")
        print(f"  Temp: {self.temperature}, Top_p: {self.top_p}, Max Tokens: {self.max_tokens}")


    # --- FIX IS ON THE NEXT LINE ---
    # Update signature to accept all arguments passed by controller_core.py (4 after self)
    def send_request_to_server(self, prompt, image:Image, request_id=None, ask_llm_recall_func=None):
        # request_id and ask_llm_recall_func are accepted to match the caller,
        # but they are not used internally for the synchronous worker endpoint.

        conv = conv_templates["vicuna_v1"].copy() # Assuming "vicuna_v1" template is compatible or used as a base

        image_base64 = None
        if image is not None:
            # LLaVA-style prompt formatting
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            conv.append_message(conv.roles[0], inp)
            image_base64 = encode_image_to_base64_with_pil(image)
            if image_base64 is None:
                 print("Error encoding image to base64.")
                 # Depending on controller_core's error handling, returning None might suffice
                 # Or you might need to raise an exception
                 return None # Indicate failure to encode image
        else:
            conv.append_message(conv.roles[0], prompt)

        conv.append_message(conv.roles[1], None) # Append assistant's turn prompt
        prompt_with_image_token = conv.get_prompt() # This includes the <image> token if image was provided

        payload = {
            'model': self.model_name,
            'prompt': prompt_with_image_token, # Use the formatted prompt
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_new_tokens': self.max_tokens,
            "image": image_base64 if image_base64 else None # Pass the base64 string (or None if no image)
        }

        # --- Synchronous HTTP request logic remains ---
        text = None
        fail_message = None
        try:
            # Use a timeout for the request
            # Increase timeout further if needed for slow model inference
            response = requests.post(self.target_url, json=payload, timeout=180) # Increased timeout to 180s
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            response_json = response.json()
            text = response_json.get("text") # Use .get for safety
            error_code = response_json.get("error_code", 0) # Check for error code from worker

            if error_code != 0:
                 fail_message = f"Worker returned error code {error_code}: {text}"
                 text = None # Treat worker errors as failed requests in this sync function

        except requests.exceptions.RequestException as e:
            fail_message = f"HTTP request to {self.target_url} failed: {e}"
        except Exception as e:
            fail_message = f"An unexpected error occurred during request: {e}"

        if fail_message:
            print(f"Error sending request to LLM worker: {fail_message}")
            return None # Return None to indicate failure
        else:
            # Print the first part of the successful response for logging
            print(f"LLM response received (partial): {text[:100] if text else 'None'}...")
            return text # Return the generated text


# Example usage (for testing the client logic separately)
# if __name__ == '__main__':
#     # This part is for testing the client file independently
#     # You would need a dummy config and a running worker
#     dummy_config = {
#         "LLaVA": {
#             "model_name": "test-model",
#             "target_url": "http://localhost:40000/worker_generate"
#         },
#         "temperature": 0.7,
#         "top_p": 0.95,
#         "max_tokens": 100
#     }
#
#     client = LanguageModelClient("LLaVA", dummy_config)
#
#     # Load a dummy image (replace with actual image loading if needed)
#     try:
#         # Create a small dummy image
#         dummy_image = Image.new('RGB', (60, 30), color = 'red')
#         print("Dummy image created.")
#     except Exception as e:
#         print(f"Error creating dummy image: {e}")
#         dummy_image = None # Proceed without image if creation fails
#
#     # Test a prompt with image
#     print("Sending request with image...")
#     # We need to pass dummy values for request_id and ask_llm_recall_func to match the signature
#     response_text = client.send_request_to_server(prompt="Describe the image:", image=dummy_image, request_id="dummy_req_123", ask_llm_recall_func=None)
#     print(f"Response: {response_text}")
#
#     # Test a prompt without image
#     print("\nSending request without image...")
#     response_text_no_image = client.send_request_to_server(prompt="Tell me a joke.", image=None, request_id="dummy_req_456", ask_llm_recall_func=None)
#     print(f"Response: {response_text_no_image}")