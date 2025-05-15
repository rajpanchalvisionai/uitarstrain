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
import sys

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

# Corrected pretty_print_semaphore to handle _waiters being None
def pretty_print_semaphore(semaphore):
     if semaphore is None:
         return "None (not initialized)" # More informative
     # Check if _waiters is None before calling len()
     num_waiting = 0
     if hasattr(semaphore, '_waiters') and semaphore._waiters is not None:
          num_waiting = len(semaphore._waiters)
     # Use .value property instead of _value (internal attribute)
     # Also handle case where semaphore might be None
     sem_value = semaphore.value if semaphore is not None else 0
     # Access args safely in case it's not set yet
     limit = args.limit_model_concurrency if 'args' in globals() and args else 'N/A'
     return f"Semaphore(value={sem_value}/{limit}, waiting={num_waiting})"


from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import TextIteratorStreamer
from threading import Thread
import base64


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
# Use a generic log file name or adapt
logger = build_logger("model_worker", f"uiflow_model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None # Will be initialized at startup now
args = None # Define args globally to be accessible in pretty_print_semaphore

# Helper function to load image from base64
def load_image_from_base64(image_base64):
    try:
        # Ensure it's a proper base64 string (strip metadata like data:image/png;base64,)
        if "," in image_base64:
             image_base64 = image_base64.split(",", 1)[1] # Split only once
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error decoding/loading image from base64: {e}")
        return None

# Define a simple mock Controller class if not registering, to satisfy heart_beat_worker
class MockController:
     def send_heart_beat(self):
          logger.debug("Mock Controller Heartbeat: Received.")
          # Simulate a response that the worker checks
          class MockResponse:
               def json(self):
                    return {"exist": True}
               def raise_for_status(self):
                    pass # Assume success
               status_code = 200
          return MockResponse()


def heart_beat_worker(controller):
    # Use the controller object passed during initialization
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        try:
            controller.send_heart_beat() # Call the method on the controller instance
        except Exception as e:
            # Error during heartbeat is logged within send_heart_beat now
            pass # Continue looping


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
                "device_map": "auto" if self.device.startswith("cuda") else self.device, # Use startswith for safety
                "torch_dtype": torch.float16 if self.device.startswith("cuda") else torch.float32, # Use float32 for CPU
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
            logger.error(f"Error loading UI-TARS model or processor: {e}", exc_info=True) # Log traceback
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1) # Exit if model loading or setup fails
        # --- End UI-TARS Specific Loading ---


        # UI-TARS is multimodal
        self.is_multimodal = True

        # Determine the controller instance for heartbeat
        if not no_register:
            self.controller = self # Worker acts as controller for sending its own heartbeats
            self.register_to_controller() # Register before starting heart beat thread
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self.controller,), daemon=True) # Pass self as controller
            self.heart_beat_thread.start()
        else:
            # If no_register, create a mock controller to prevent heart_beat_worker errors
            self.controller = MockController()
            # Optionally skip the heart beat thread entirely if no_register is True
            # self.heart_beat_thread = None


    def register_to_controller(self):
        if args.no_register:
             logger.debug("Registration skipped (no_register flag is set).")
             return

        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        try:
            r = requests.post(url, json=data, timeout=5) # Add timeout for registration
            r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            # assert r.status_code == 200 # Explicitly check status code - raise_for_status does this
            logger.info("Successfully registered to controller.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register to controller at {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during registration: {e}", exc_info=True)


    def send_heart_beat(self):
        # This method is now called by the heart_beat_worker thread
        # It sends its own status to the controller_addr
        if args.no_register:
            # logger.debug("Heart beat skipped (no_register flag is set).") # Too noisy
            return

        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        # Add retry logic for sending heart beat
        for i in range(3): # Retry up to 3 times
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                ret.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                exist = ret.json()["exist"]
                if not exist:
                     logger.warning("Worker not found in controller on heart beat, attempting to re-register...")
                     self.register_to_controller() # Re-register here
                else:
                     logger.debug("Heart beat successful.")
                break # Success, exit retry loop
            except requests.exceptions.RequestException as e:
                logger.error(f"Heart beat error (attempt {i+1}/3): {e}")
                if i < 2: # Wait before retrying, but not after the last attempt
                    time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error during heart beat (attempt {i+1}/3): {e}", exc_info=True) # Log traceback
                if i < 2:
                    time.sleep(5)

        # If heart beat consistently fails, the controller should eventually detect the worker is down.
        # We don't want the worker to crash just because it can't reach the controller.


    def get_queue_length(self):
        if model_semaphore is None:
            return 0 # Semaphore not initialized, no queue to report
        else:
            # Use .value property instead of _value (internal attribute)
            num_waiting = 0
            if hasattr(model_semaphore, '_waiters') and model_semaphore._waiters is not None:
                 num_waiting = len(model_semaphore._waiters)
            # Queue length is max capacity minus current value, plus number waiting
            # Access args.limit_model_concurrency safely if args is None somehow
            limit = args.limit_model_concurrency if 'args' in globals() and args else 0
            return max(0, limit - model_semaphore.value) + num_waiting


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
        Returns a dictionary of tensors suitable for model.generate.
        """
        if not self.is_multimodal:
             # If somehow this runs for a non-multimodal model, log and return None
             logger.error("prepare_uitars_input called on a non-multimodal worker.")
             return None

        content = []
        # Add images first, then the text.
        if images:
             for img in images:
                 if img: # Ensure image loading was successful
                     content.append({"type": "image", "image": img})
                 else:
                      logger.warning("Skipping None image in prepare_uitars_input.")


        if prompt:
             content.append({"type": "text", "text": prompt})

        if not content:
             # Handle empty prompt and image list
             logger.warning("prepare_uitars_input called with no content (empty prompt and image list).")
             return None

        conversation = [{"role": "user", "content": content}]
        # logger.debug(f"Conversation structure for apply_chat_template: {conversation}") # Log conversation structure


        try:
            # apply_chat_template should return input_ids, attention_mask, and potentially pixel_values
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True, # Add the prompt for the model to start generating
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            # logger.debug(f"Inputs from apply_chat_template: {inputs.keys()}") # Log keys in the inputs dict
            # For debugging: print shapes of tensors in inputs
            # debug_inputs_info = {k:v.shape if isinstance(v, torch.Tensor) else type(v) for k,v in inputs.items()}
            # logger.debug(f"Inputs tensor shapes: {debug_inputs_info}")

            return inputs
        except Exception as e:
            logger.error(f"Error applying chat template: {e}", exc_info=True) # Log traceback
            # Include the conversation structure in the error for debugging
            logger.error(f"Conversation structure: {conversation}")
            # Propagate the error so the calling method can return a structured error
            raise ValueError(f"Failed to prepare model input: {e}")


    @torch.inference_mode()
    def generate_stream(self, params):

        prompt = params["prompt"]
        # Expecting a list of base64 strings for the 'images' key
        images_base64 = params.get("images", []) # Default to empty list if 'images' key is missing
        pil_images = [load_image_from_base64(img_b64) for img_b64 in images_base64 if img_b64] # Filter out None results


        # Allow text-only prompts even for multimodal, just without image
        if self.is_multimodal and not pil_images and not prompt:
             yield json.dumps({"text": "Error: No input (prompt or image) provided for multimodal model.", "error_code": 1}).encode() + b"\0"
             return
        # If not multimodal and no prompt, also error
        if not self.is_multimodal and not prompt:
             yield json.dumps({"text": "Error: No prompt provided for text-only model.", "error_code": 1}).encode() + b"\0"
             return


        try:
            inputs = self.prepare_uitars_input(prompt, pil_images)
            if inputs is None: # Handles cases where input preparation failed or was empty
                 yield json.dumps({"text": "Error: Failed to prepare input.", "error_code": 1}).encode() + b"\0"
                 return

        except ValueError as e: # Catch errors raised by prepare_uitars_input
             logger.error(f"Error preparing input in generate_stream: {e}")
             yield json.dumps({"text": f"Error preparing input: {e}", "error_code": 1}).encode() + b"\0"
             return


        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))

        # Calculate max_new_tokens based on remaining context space, ensuring at least 1
        input_seq_len = inputs.input_ids.shape[-1]
        # Subtract 1 for potential EOS token or other special tokens the model might add
        remaining_context = self.context_len - input_seq_len - 1

        if remaining_context < 1:
            logger.warning(f"Input sequence length ({input_seq_len}) exceeds or is too close to context length ({self.context_len}). No space for new tokens.")
            yield json.dumps({"text": "Error: Input sequence length exceeds model maximum context length.", "error_code": 1}).encode() + b"\0"
            return

        max_new_tokens = min(int(params.get("max_new_tokens", 512)), remaining_context)
        max_new_tokens = max(1, max_new_tokens) # Ensure minimum max_new_tokens is at least 1

        logger.info(f"Generating {max_new_tokens} new tokens. Input length: {input_seq_len}, Context length: {self.context_len}")

        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False


        # Use TextIteratorStreamer as in the original LLaVA worker
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        # --- FIX: Pass the entire inputs dictionary to model.generate ---
        # This ensures all necessary keys generated by apply_chat_template are passed
        generation_kwargs = dict(
            **inputs, # Unpack inputs: includes input_ids, attention_mask, pixel_values, etc.
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id # Add pad token id for generation
        )

        logger.info(f"Starting streaming generation with kwargs (shapes): { {k:v.shape if isinstance(v, torch.Tensor) else v for k,v in generation_kwargs.items()} }") # Log shapes of tensors
        try:
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
        except Exception as e:
            logger.error(f"Error starting generation thread: {e}", exc_info=True)
            yield json.dumps({"text": f"Generation error: {e}", "error_code": 1}).encode() + b"\0"
            return # Exit the generator

        generated_text = ""
        # Stream tokens from the streamer
        # Keep track of yielded text to handle stop string correctly at the end
        yielded_text = ""
        try:
            for new_text in streamer:
                generated_text += new_text
                # Yield the new token immediately as part of the stream
                yield json.dumps({"text": yielded_text + new_text, "error_code": 0}).encode() + b"\0"
                yielded_text += new_text

                # Check for stop string after yielding the current token
                if stop_str and yielded_text.endswith(stop_str):
                     break # Exit loop after encountering stop string


            # Ensure the thread finishes (it should when streamer is exhausted or stopped)
            thread.join()

            # Final processing after the stream ends
            final_text = yielded_text
            if stop_str and yielded_text.endswith(stop_str):
                 final_text = yielded_text[:-len(stop_str)]

            # Check if the streamer finished normally or timed out
            if not streamer.text_iterator: # Check if the underlying iterator is exhausted/closed
                 logger.info("Streaming finished normally.")
            else:
                 logger.warning("Streaming may have timed out or finished unexpectedly.")

            # The last yield in the loop should have sent the final token.
            # No need for an extra final yield unless specific client behavior requires it.


        except Exception as e:
             logger.error(f"Error during streaming: {e}", exc_info=True)
             yield json.dumps({"text": f"Streaming error: {e}", "error_code": 1}).encode() + b"\0"
        finally:
             # Ensure the thread is joined even if streaming failed
             if thread.is_alive():
                  logger.warning("Generation thread is still alive after streaming ended unexpectedly. Attempting to join.")
                  thread.join(timeout=1) # Try joining briefly
                  if thread.is_alive():
                       logger.error("Generation thread is still alive after timeout. Thread may be stuck.")


    def generate_stream_gate(self, params):
        # This method remains largely the same, handling exceptions
        try:
            # Check model is ready or handle busy state if concurrency limited
            # Semaphore handling is done in the FastAPI endpoint
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            logger.error(f"ValueError in generate_stream_gate: {e}")
            ret = {
                "text": server_error_msg(),
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            logger.error(f"CUDA error in generate_stream_gate: {e}")
            ret = {
                "text": server_error_msg(),
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            logger.error(f"Unknown error in generate_stream_gate: {e}", exc_info=True) # Log traceback
            ret = {
                "text": server_error_msg(),
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


    @torch.inference_mode()
    def direct_generate(self, params):
        # This method implements the non-streaming generation like your original UI-TARS script

        prompt = params["prompt"]
        # Expecting image(s) under the 'image' key (single base64 string) or 'images' key (list of base64 strings)
        image_base64_single = params.get("image", None)
        images_base64_list = params.get("images", [])

        pil_images = []
        if image_base64_single:
             img = load_image_from_base64(image_base64_single)
             if img:
                 pil_images = [img] # Wrap single image in a list
             else:
                 logger.warning("Skipping None image from 'image' key in direct_generate.")
        elif images_base64_list:
             pil_images = [load_image_from_base64(img_b64) for img_b64 in images_base64_list if img_b64] # Filter out None results


        # Allow text-only prompts even for multimodal
        if self.is_multimodal and not pil_images and not prompt:
             return {"text": "Error: No input (prompt or image) provided for multimodal model.", "error_code": 1}
        # If not multimodal and no prompt, also error
        if not self.is_multimodal and not prompt:
             return {"text": "Error: No prompt provided for text-only model.", "error_code": 1}


        try:
            inputs = self.prepare_uitars_input(prompt, pil_images)
            if inputs is None:
                 return {"text": "Error: Failed to prepare input.", "error_code": 1}
        except ValueError as e: # Catch errors from prepare_uitars_input
             logger.error(f"Error preparing input in direct_generate: {e}")
             return {"text": f"Error preparing input: {e}", "error_code": 1}


        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))

        # Calculate max_new_tokens based on remaining context space, ensuring at least 1
        input_seq_len = inputs.input_ids.shape[-1]
        remaining_context = self.context_len - input_seq_len - 1 # Subtract 1 for potential EOS token

        if remaining_context < 1:
             logger.warning(f"Input sequence length ({input_seq_len}) exceeds or is too close to context length ({self.context_len}). No space for new tokens.")
             return {"text": "Error: Input sequence length exceeds model maximum context length.", "error_code": 1}

        max_new_tokens = min(int(params.get("max_new_tokens", 512)), remaining_context)
        max_new_tokens = max(1, max_new_tokens) # Ensure minimum is 1

        logger.info(f"Generating {max_new_tokens} new tokens. Input length: {input_seq_len}, Context length: {self.context_len}")

        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False


        # The generate call uses the 'inputs' dict from apply_chat_template
        # --- FIX: Pass the entire inputs dictionary to model.generate ---
        # This ensures all necessary keys generated by apply_chat_template are passed
        generation_kwargs = dict(
             **inputs, # Unpack inputs: includes input_ids, attention_mask, pixel_values, etc.
             do_sample=do_sample,
             temperature=temperature,
             top_p=top_p,
             max_new_tokens=max_new_tokens,
             use_cache=True,
             pad_token_id=self.tokenizer.eos_token_id
             # stopping_criteria= # Add stopping criteria if needed for stop_str
        )

        logger.info(f"Starting direct generation with kwargs (shapes): { {k:v.shape if isinstance(v, torch.Tensor) else v for k,v in generation_kwargs.items()} }") # Log shapes
        try:
            generation_output = self.model.generate(
                **generation_kwargs
            )

            # Decode the generated output, slicing to remove input tokens
            # Assuming batch size is 1 and output is a tensor
            if not isinstance(generation_output, torch.Tensor) or generation_output.ndim == 0:
                 logger.error(f"Model.generate returned unexpected output type/shape: {type(generation_output)}, {generation_output.shape if isinstance(generation_output, torch.Tensor) else 'N/A'}")
                 return {"text": "Model generation failed: Unexpected output format.", "error_code": 1}

            # Assuming the output tensor is [batch_size, sequence_length]
            # and batch_size is 1.
            if generation_output.shape[0] != 1:
                 logger.warning(f"Model.generate returned batch size > 1 ({generation_output.shape[0]}). Using the first sequence.")

            generated_ids = generation_output[0, input_seq_len:] # Slice the first sequence

            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # Optional: Trim stop string if it appears at the very end
            if stop_str and decoded.endswith(stop_str):
                 decoded = decoded[:-len(stop_str)]

            response = {"text": decoded, "error_code": 0}
            logger.info(f"Direct generate success. Response (partial): {decoded[:100]}...")
            return response

        except Exception as e:
            logger.error(f"Error during direct generate: {e}", exc_info=True) # Log traceback
            # Return server error message in the expected format
            return {"text": server_error_msg(), "error_code": 1}


app = FastAPI()

# The /worker_generate endpoint will NOT use the semaphore
@app.post("/worker_generate")
async def generate(request: Request):
    global global_counter
    global_counter += 1
    params = await request.json()

    # No semaphore acquire/release for the non-streaming endpoint based on original LLaVA worker structure
    # This assumes ScreenAgent's primary planning calls don't require concurrency control at the worker level
    # or that they are handled differently by the client logic.

    try:
        # This is a synchronous call within an async endpoint. For long tasks,
        # consider running it in a thread pool using `from fastapi.concurrency import run_in_threadpool`
        # and `response_data = await run_in_threadpool(worker.direct_generate, params)`
        # For now, let's keep it simple as direct_generate includes CUDA operations which might release GIL.
        response_data = worker.direct_generate(params)
        return response_data
    except Exception as e:
        logger.error(f"Exception in /worker_generate endpoint: {e}", exc_info=True)
        # Return a structured error response
        return {"text": server_error_msg(), "error_code": 1}


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    # Acquire semaphore for streaming requests
    # Initialize semaphore if it's None (should be initialized at startup now, but defensive check)
    if model_semaphore is None:
         logger.warning("Semaphore was None in generate_stream, initializing now.")
         # Access args safely
         limit = args.limit_model_concurrency if 'args' in globals() and args else 5
         model_semaphore = asyncio.Semaphore(limit)

    # Acquire semaphore - this will pause here if concurrency limit is reached
    logger.info("Attempting to acquire semaphore for streaming request.")
    await model_semaphore.acquire()
    logger.info("Semaphore acquired for streaming request.")
    # worker.send_heart_beat() # Send heart beat when starting a task - Maybe not needed, semaphore release task sends heartbeat


    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    # Use background task to release semaphore after the streaming response is done
    # This needs to be released even if the generator raises an exception
    # Pass worker.controller to release_model_semaphore
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))

    # StreamingResponse automatically handles yielding from the generator
    # Ensure content type is correct for streaming JSON
    return StreamingResponse(generator, background=background_tasks, media_type="application/json")


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


def release_model_semaphore(fn=None):
    """
    Releases the model semaphore. Called as a background task
    when a streaming response is finished or an error occurs.
    """
    # Only release if the semaphore is not None and is currently acquired (value < limit)
    # Use .value property
    # Access args safely
    limit = args.limit_model_concurrency if 'args' in globals() and args else 0
    if model_semaphore is not None and model_semaphore.value < limit:
         model_semaphore.release()
         logger.info("Semaphore released.")
         if fn:
             try:
                 # Ensure the function (heartbeat) runs in its own context
                 # or is designed to be thread-safe/async-safe if called from a background task
                 # The heartbeat function is designed to use requests (blocking), so running it
                 # directly might block the event loop slightly, but it's common in these workers.
                 fn() # Call heartbeat or other cleanup if provided
             except Exception as e:
                 logger.error(f"Error calling post-release cleanup function: {e}", exc_info=True) # Log traceback
    else:
         # This might happen if release is called multiple times or semaphore state is unexpected
         logger.warning(f"Semaphore attempted to be released but value ({model_semaphore.value if model_semaphore else 'None'}) is already max ({limit}) or semaphore is None.")


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

    # Parse args and assign to global args variable early
    args = parser.parse_args()
    logger.info(f"args: {args}")


    try:
        worker = ModelWorker(args.controller_address,
                             args.worker_address,
                             worker_id,
                             args.no_register,
                             args.model_path,
                             args.model_base,
                             args.model_name,
                             args.load_8bit,
                             args.load_4bit,
                             args.device,
                             use_flash_attn=args.use_flash_attn)

        # Initialize semaphore at startup
        # It's an asyncio.Semaphore, should be initialized within the async context or before the loop starts
        # Initializing here is okay as uvicorn.run starts an async loop
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
        logger.info(f"Model semaphore initialized with limit: {args.limit_model_concurrency}")


        logger.info(f"Starting server on {args.host}:{args.port}")
        # Pass a list of applications to uvicorn.run if you have multiple,
        # but here 'app' is the FastAPI instance, so just pass app.
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    except Exception as e:
        logger.error(f"Failed to start worker: {e}", exc_info=True) # Log traceback
        print(f"Failed to start worker: {e}", file=sys.stderr)
        sys.exit(1) # Exit if model loading or setup fails