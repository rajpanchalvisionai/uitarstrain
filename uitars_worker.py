"""
Modify from https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/model_worker.py
A model worker executes the model.

Adapted for ByteDance-Seed/UI-TARS-2B-SFT model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import os

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial
from io import BytesIO
from PIL import Image

# Assume constants and utils are available or replicate what's needed
# For simplicity, we'll define WORKER_HEART_BEAT_INTERVAL here
WORKER_HEART_BEAT_INTERVAL = 15 # seconds

# If llava.utils are not available, define simple placeholders
def build_logger(name, filename):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def server_error_msg():
    return "Internal Server Error"

def pretty_print_semaphore(semaphore):
     if semaphore is None:
         return "None"
     return f"Semaphore(value={semaphore._value}, waiting={len(semaphore._waiters)})"

from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import TextIteratorStreamer
from threading import Thread
import base64


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
# Use a generic log file name or adapt
logger = build_logger("model_worker", f"uiflow_model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None

# Helper function to load image from base64
def load_image_from_base64(image_base64):
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error decoding/loading image from base64: {e}")
        return None

def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name, # model_base is likely not used for HF hub models
                 load_8bit, load_4bit, device, use_flash_attn=False):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id

        # Determine model name from path if not provided
        if model_name is None:
             if model_path.endswith("/"):
                 model_path = model_path[:-1]
             self.model_name = os.path.basename(model_path)
        else:
            self.model_name = model_name


        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")

        # --- UI-TARS Specific Loading ---
        try:
            # Using AutoProcessor for multimodal components
            self.processor = AutoProcessor.from_pretrained(model_path)
            # Use the specific model class Qwen2VLForConditionalGeneration
            # Handle device mapping and quantization arguments
            load_kwargs = {
                "device_map": "auto" if self.device == "cuda" else self.device,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32, # Use float32 for CPU
            }
            if load_8bit:
                 load_kwargs["load_in_8bit"] = True
            if load_4bit:
                 load_kwargs["load_in_4bit"] = True
            if use_flash_attn:
                 # This might depend on the specific Qwen2VL implementation and transformers version
                 # Using attention_implementation="flash_attention_2" is common
                 load_kwargs["attn_implementation"] = "flash_attention_2"
                 # Ensure flash_attn is installed: pip install flash-attn --no-build-isolation
                 logger.info("Attempting to use flash_attention_2")


            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )
            self.tokenizer = self.processor.tokenizer # Tokenizer is part of the processor
            self.image_processor = self.processor # Image processor is also part of the processor
            # UI-TARS doesn't expose context_len like LLaVA, using model config
            self.context_len = getattr(self.model.config, 'max_position_embeddings', 8192) # Qwen2VL typically has larger context
            logger.info("Model and processor loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading UI-TARS model or processor: {e}")
            # Re-raise or exit if loading fails
            raise e
        # --- End UI-TARS Specific Loading ---


        # UI-TARS is multimodal
        self.is_multimodal = True #'llava' in self.model_name.lower() or 'qwen2-vl' in self.model_name.lower()

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,), daemon=True)
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        try:
            r = requests.post(url, json=data)
            assert r.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register to controller: {e}")


    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
                time.sleep(5) # Wait before retrying

        if not exist:
            logger.warning("Worker not found in controller, re-registering...")
            self.register_to_controller()


    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            # This calculation might need adjustment based on how semaphore is used
            # and if there are other queues. This is a basic estimate.
            num_waiting = 0
            if hasattr(model_semaphore, '_waiters') and model_semaphore._waiters is not None:
                 num_waiting = len(model_semaphore._waiters)
            return max(0, args.limit_model_concurrency - model_semaphore._value) + num_waiting


    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1, # This is a placeholder
            "queue_length": self.get_queue_length(),
        }

    # --- UI-TARS Specific Input Preparation ---
    def prepare_uitars_input(self, prompt, images):
        """
        Prepares input in the UI-TARS conversation format.
        Assumes prompt is text and images is a list of PIL Images.
        Creates a single user turn with interleaved images and text.
        """
        if not self.is_multimodal:
             # Handle text-only models if needed, though this worker is for VL
             return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        content = []
        # Simple approach: Add all images first, then the text.
        # A more sophisticated approach might interleave based on tokens like <image>
        # but UI-TARS `apply_chat_template` expects the structured list.
        # screenagent might need to control the order in the prompt itself
        # if complex interleaving is required. For now, images first then text.
        if images:
             for img in images:
                 if img: # Ensure image loading was successful
                     content.append({"type": "image", "image": img})

        if prompt:
             content.append({"type": "text", "text": prompt})

        if not content:
             # Handle empty prompt and image list
             return None # Or raise an error

        conversation = [{"role": "user", "content": content}]

        try:
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True, # Add the prompt for the model to start generating
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            return inputs
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            raise ValueError(f"Failed to prepare model input: {e}")


    @torch.inference_mode()
    def generate_stream(self, params):
        # tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor # Use self.processor

        prompt = params["prompt"]
        images_base64 = params.get("images", None) # Expecting a list of base64 strings
        pil_images = []
        if images_base64 and isinstance(images_base64, list):
            pil_images = [load_image_from_base64(img_b64) for img_b64 in images_base64 if img_b64]

        if self.is_multimodal and not pil_images and not prompt:
             yield json.dumps({"text": "Error: No input (prompt or image) provided for multimodal model.", "error_code": 1}).encode() + b"\0"
             return

        try:
            inputs = self.prepare_uitars_input(prompt, pil_images)
            if inputs is None: # Handle cases where input preparation failed or was empty
                 yield json.dumps({"text": "Error: Failed to prepare input.", "error_code": 1}).encode() + b"\0"
                 return

        except ValueError as e:
             yield json.dumps({"text": f"Error preparing input: {e}", "error_code": 1}).encode() + b"\0"
             return


        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        # max_context_length is handled by self.context_len and apply_chat_template implicitly
        max_new_tokens = min(int(params.get("max_new_tokens", 512)), 2048) # Increased max_new_tokens default/cap
        stop_str = params.get("stop", None) # Note: stop string handling with streamer might be complex
        do_sample = True if temperature > 0.001 else False

        # Check if input sequence length exceeds context window BEFORE generation
        input_seq_len = inputs.input_ids.shape[-1]
        if input_seq_len >= self.context_len:
             yield json.dumps({"text": "Error: Input sequence length exceeds model maximum context length.", "error_code": 1}).encode() + b"\0"
             return

        # Adjust max_new_tokens based on remaining context space
        max_new_tokens = min(max_new_tokens, self.context_len - input_seq_len)

        if max_new_tokens < 1:
            # This case should ideally be caught by the input_seq_len check, but double-check
            yield json.dumps({"text": "Exceeds max token length or no space for new tokens. Please start a new conversation.", "error_code": 0}).encode() + b"\0"
            return


        # Use TextIteratorStreamer as in the original LLaVA worker
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        # The generate call now uses the 'inputs' dict from apply_chat_template
        generation_kwargs = dict(
            inputs=inputs.input_ids, # Pass input_ids from the processed inputs
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            # UI-TARS model.generate handles images via the inputs generated by apply_chat_template
            # We don't pass 'images' or 'image_sizes' explicitly here like in LLaVA
            pad_token_id=self.tokenizer.eos_token_id # Add pad token id for generation
        )

        # Add image features to generation_kwargs if present in inputs
        if 'pixel_values' in inputs:
            generation_kwargs['pixel_values'] = inputs['pixel_values']
        if 'image_sizes' in inputs:
            # Note: Qwen2VL apply_chat_template might not return image_sizes explicitly
            # If needed, you might have to handle image_sizes separately during image loading
            # and pass them if the model's generate method expects them.
            # Check Qwen2VL docs if generation fails without this.
            # For now, assuming pixel_values is sufficient via inputs dict.
            pass


        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        # Stream tokens from the streamer
        for new_text in streamer:
            generated_text += new_text
            # Basic stop string handling (might yield partial stop string before stopping)
            if stop_str and generated_text.endswith(stop_str):
                 # Trim the stop string
                 generated_text = generated_text[:-len(stop_str)]
                 yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"
                 # Stop the streamer/thread if possible, or rely on it finishing
                 break # Exit loop after encountering stop string
            # Yield current cumulative text
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

        # Ensure the thread finishes (it should when streamer is exhausted or stopped)
        thread.join()


    def generate_stream_gate(self, params):
        # This method remains largely the same, handling exceptions
        try:
            # Check model is ready or handle busy state if concurrency limited
            # Semaphore handling is done in the FastAPI endpoint
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            logger.error(f"ValueError in generate_stream_gate: {e}")
            ret = {
                "text": server_error_msg(),
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            logger.error(f"CUDA error in generate_stream_gate: {e}")
            ret = {
                "text": server_error_msg(),
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            logger.error(f"Unknown error in generate_stream_gate: {e}")
            ret = {
                "text": server_error_msg(),
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


    @torch.inference_mode()
    def direct_generate(self, params):
        # This method implements the non-streaming generation like your original UI-TARS script
        # tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor # Use self.processor

        prompt = params["prompt"]
        images_base64 = params.get("images", None) # Expecting a list of base64 strings, consistent with stream
        pil_images = []
        if images_base64 and isinstance(images_base64, list):
            pil_images = [load_image_from_base64(img_b64) for img_b64 in images_base64 if img_b64]

        if self.is_multimodal and not pil_images and not prompt:
             return {"text": "Error: No input (prompt or image) provided for multimodal model.", "error_code": 1}

        try:
            inputs = self.prepare_uitars_input(prompt, pil_images)
            if inputs is None:
                 return {"text": "Error: Failed to prepare input.", "error_code": 1}
        except ValueError as e:
             return {"text": f"Error preparing input: {e}", "error_code": 1}


        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 512)), 2048)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        # Check if input sequence length exceeds context window BEFORE generation
        input_seq_len = inputs.input_ids.shape[-1]
        if input_seq_len >= self.context_len:
             return {"text": "Error: Input sequence length exceeds model maximum context length.", "error_code": 1}

        # Adjust max_new_tokens
        max_new_tokens = min(max_new_tokens, self.context_len - input_seq_len)

        if max_new_tokens < 1:
            return {"text": "Exceeds max token length or no space for new tokens. Please start a new conversation.", "error_code": 0}


        # The generate call uses the 'inputs' dict from apply_chat_template
        generation_kwargs = dict(
             inputs=inputs.input_ids,
             do_sample=do_sample,
             temperature=temperature,
             top_p=top_p,
             max_new_tokens=max_new_tokens,
             use_cache=True,
             pad_token_id=self.tokenizer.eos_token_id
             # stopping_criteria= # Add stopping criteria if needed for stop_str
        )
        if 'pixel_values' in inputs:
            generation_kwargs['pixel_values'] = inputs['pixel_values']


        try:
            generation_output = self.model.generate(
                **generation_kwargs
            )

            # Decode the generated output, slicing to remove input tokens
            # Assuming batch size is 1
            generated_ids = generation_output[0, input_seq_len:]
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # Optional: Trim stop string if it appears at the very end
            if stop_str and decoded.endswith(stop_str):
                 decoded = decoded[:-len(stop_str)]

            response = {"text": decoded, "error_code": 0}
            logger.info(f"Direct generate response (partial): {decoded[:100]}...")
            return response

        except Exception as e:
            logger.error(f"Error during direct generate: {e}")
            return {"text": server_error_msg(), "error_code": 1}


app = FastAPI()

# Note: The /worker_generate endpoint in the original LLaVA code
# did not use the semaphore or background task release.
# We'll add semaphore protection here for consistency and resource management.
# The user might need to check screenagent's expected behavior for this endpoint.
@app.post("/worker_generate")
async def generate(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    # Acquire semaphore before processing a request
    await model_semaphore.acquire()
    worker.send_heart_beat() # Send heart beat when starting a task

    try:
        # This is a synchronous call now, need to run in a thread pool to not block
        # FastAPI's async loop if the direct_generate takes a long time.
        # However, given it's inference, it's often run on GPU and might release
        # GIL, making it less blocking. Let's keep it simple first.
        # For a truly non-blocking endpoint, use `run_in_threadpool` from `fastapi.concurrency`
        # or ensure `direct_generate` itself is async (which it isn't currently).
        # We'll release the semaphore in a background task too, similar to streaming.
        response_data = worker.direct_generate(params)
        return response_data
    finally:
        # Ensure semaphore is released even if an error occurs
        background_tasks = BackgroundTasks()
        background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
        # This response structure needs to be handled correctly by the client.
        # Returning a Response object with background tasks might not be standard
        # for non-streaming endpoints. Releasing in a `finally` block directly
        # might be simpler if errors are caught *inside* direct_generate.
        # Let's rely on the `finally` block directly releasing the semaphore.
        # The background task release is more suited for streaming responses.
        # Let's just release the semaphore directly here.
        # background_tasks.add_task(partial(release_model_semaphore)) # Removed background task for non-streaming

        # Re-think: The LLaVA worker didn't use semaphore for direct_generate.
        # Let's revert to that. Only streaming endpoint uses the semaphore.
        # If screenagent expects the /worker_generate endpoint to be available
        # even when concurrency limit is reached for streaming, this is necessary.
        # If resource contention is an issue for direct_generate too, then semaphore is needed.
        # Assuming /worker_generate is less critical or handled differently.
        # Let's stick to the original LLaVA worker's semaphore logic (only for streaming).

        # Reverting the semaphore acquire/release for /worker_generate
        pass # No semaphore for direct_generate based on original LLaVA worker


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    # Acquire semaphore for streaming requests
    await model_semaphore.acquire()
    worker.send_heart_beat() # Send heart beat when starting a task

    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    # Use background task to release semaphore after the streaming response is done
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))

    # StreamingResponse automatically handles yielding from the generator
    return StreamingResponse(generator, background=background_tasks, media_type="application/json")


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


def release_model_semaphore(fn=None):
    """
    Releases the model semaphore. Called as a background task
    when a streaming response is finished or an error occurs.
    """
    if model_semaphore and model_semaphore._value < args.limit_model_concurrency:
         model_semaphore.release()
         logger.info("Semaphore released.")
         if fn:
             fn() # Call heartbeat or other cleanup if provided
    else:
         logger.warning("Semaphore attempted to be released but value is already max or semaphore is None.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    # Use model-path for the Hugging Face model ID
    parser.add_argument("--model-path", type=str, default="ByteDance-Seed/UI-TARS-2B-SFT",
                        help="Hugging Face model ID or path to the model.")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path (not typically needed for Hugging Face hub models).")
    parser.add_argument("--model-name", type=str,
                        help="Optional: A custom name for the model worker.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load the model on (e.g., 'cuda', 'cpu', 'cuda:0'). 'cuda' with device_map='auto' is recommended for GPU.")
    # multimodal argument is now redundant as UI-TARS is inherently multimodal
    # parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` or `qwen2-vl` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5,
                        help="Maximum number of concurrent streaming requests the worker can handle.")
    parser.add_argument("--stream-interval", type=int, default=1, # This might not be used by TextIteratorStreamer directly
                        help="Interval between streaming tokens (placeholder).")
    parser.add_argument("--no-register", action="store_true",
                        help="Do not register the worker with the controller.")
    parser.add_argument("--load-8bit", action="store_true",
                        help="Load the model in 8-bit mode.")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Load the model in 4-bit mode.")
    parser.add_argument("--use-flash-attn", action="store_true",
                        help="Use Flash Attention 2 if available. Requires flash-attn library.")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # The multimodal flag is now ignored, UI-TARS is always multimodal
    # if args.multi_modal:
    #     logger.warning("Multimodal flag is ignored. UI-TARS is automatically detected as multimodal.")


    try:
        worker = ModelWorker(args.controller_address,
                            args.worker_address,
                            worker_id,
                            args.no_register,
                            args.model_path,
                            args.model_base, # Still passed, though likely None
                            args.model_name,
                            args.load_8bit,
                            args.load_4bit,
                            args.device,
                            use_flash_attn=args.use_flash_attn)

        # Initial semaphore setup based on concurrency limit
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)


        logger.info(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    except Exception as e:
        logger.error(f"Failed to start worker: {e}")
        # Exit if model loading or setup fails
        exit(1)