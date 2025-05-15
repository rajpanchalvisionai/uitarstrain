import asyncio
import asyncvnc
import sys

import argparse
import yaml

import qasync
import functools
from qasync import QApplication

import controller_core
from automaton import Automaton


def close_future(future, loop):
    loop.call_later(10, future.cancel)
    future.cancel()

async def run_client(config):

    loop = asyncio.get_event_loop()
    future = asyncio.Future()

    app = QApplication(sys.argv)
    if hasattr(app, "aboutToQuit"):
        getattr(app, "aboutToQuit").connect(functools.partial(close_future, future, loop))

    llm_api_client = None
    full_llm_api_config = config.get('llm_api', {}) # Use .get for safety, provide default empty dict

    # Determine which model is active and instantiate the correct client
    # Pass the string key and the full config dict to the client __init__

    if full_llm_api_config.get("GPT4V", None):
        from interface_api.gpt4_client import LanguageModelClient
        # Pass the string key "GPT4V" and the full config dict
        llm_api_client = LanguageModelClient("GPT4V", full_llm_api_config)

    elif full_llm_api_config.get("LLaVA", None): # <-- This block is executed
        from interface_api.llava_llm_client import LanguageModelClient # <-- Imports LLaVA client
        # --- FIX IS ON THE NEXT LINE ---
        # Pass the string key "LLaVA" and the full config dict
        llm_api_client = LanguageModelClient("LLaVA", full_llm_api_config)

    elif full_llm_api_config.get("CogAgent", None):
        from interface_api.cogagent_llm_client import LanguageModelClient
        # This already matches the expected signature (string, full_dict)
        llm_api_client = LanguageModelClient("CogAgent", full_llm_api_config)

    elif full_llm_api_config.get("ScreenAgent", None):
        from interface_api.cogagent_llm_client import LanguageModelClient
        # This also matches the expected signature (string, full_dict)
        llm_api_client = LanguageModelClient("ScreenAgent", full_llm_api_config)

    else:
        raise ValueError("No active LLM API configuration found in config.yml. Please uncomment and configure one model (e.g., LLaVA).")

    assert llm_api_client is not None, "Failed to instantiate LLM API client." # Improved assertion message

    automaton_config = config.get('automaton', {}) # Use .get for safety
    automaton = Automaton(automaton_config)

    remote_vnc_server_config = config.get('remote_vnc_server', {}) # Use .get for safety

    if remote_vnc_server_config.get('use_remote_clipboard'):
        from action import KeyboardAction
        KeyboardAction.set_remote_clipboard(remote_vnc_server_config)

    # Check if required VNC config keys are present
    if 'host' not in remote_vnc_server_config or 'port' not in remote_vnc_server_config or 'password' not in remote_vnc_server_config:
         raise ValueError("Missing required keys in remote_vnc_server configuration (host, port, password).")

    widget = controller_core.VNCWidget(remote_vnc_server_config, llm_client = llm_api_client, automaton = automaton)
    widget.show()

    await future


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read YAML config file.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()

    # read config file
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file {args.config}: {e}")
        sys.exit(1)


    try:
        qasync.run(run_client(config))
    except asyncio.exceptions.CancelledError:
        # This is a normal shutdown when the Qt application quits
        sys.exit(0)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Consider more specific exception handling here
        sys.exit(1)




# import asyncio
# import asyncvnc
# import sys

# import argparse
# import yaml

# import qasync
# import functools
# from qasync import QApplication

# import controller_core
# from automaton import Automaton


# def close_future(future, loop):
#     loop.call_later(10, future.cancel)
#     future.cancel()

# async def run_client(config):

#     loop = asyncio.get_event_loop()
#     future = asyncio.Future()

#     app = QApplication(sys.argv)
#     if hasattr(app, "aboutToQuit"):
#         getattr(app, "aboutToQuit").connect(functools.partial(close_future, future, loop))

#     llm_api_client = None
#     api_config = config['llm_api']

#     if api_config.get("GPT4V", None):
#         from interface_api.gpt4_client import LanguageModelClient
#         llm_api_client = LanguageModelClient(api_config)

#     elif api_config.get("LLaVA", None):
#         from interface_api.llava_llm_client import LanguageModelClient
#         llm_api_client = LanguageModelClient(api_config)

#     elif api_config.get("CogAgent", None):
#         from interface_api.cogagent_llm_client import LanguageModelClient
#         llm_api_client = LanguageModelClient("CogAgent", api_config)

#     elif api_config.get("ScreenAgent", None):
#         from interface_api.cogagent_llm_client import LanguageModelClient
#         llm_api_client = LanguageModelClient("ScreenAgent", api_config)

#     else:
#         raise ValueError("No LLM API config found")

#     assert llm_api_client is not None, "No LLM API client found"

#     automaton_config = config['automaton']
#     automaton = Automaton(automaton_config)

#     remote_vnc_server_config = config['remote_vnc_server']

#     if remote_vnc_server_config.get('use_remote_clipboard'):
#         from action import KeyboardAction
#         KeyboardAction.set_remote_clipboard(remote_vnc_server_config)
        
#     widget = controller_core.VNCWidget(remote_vnc_server_config, llm_client = llm_api_client, automaton = automaton)
#     widget.show()

#     await future


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description='Read YAML config file.')
#     parser.add_argument('-c', '--config', type=str, required=True, help='Path to the YAML config file.')
#     args = parser.parse_args()

#     # read config file
#     with open(args.config, "r") as file:
#         config = yaml.safe_load(file)

#     try:
#         qasync.run(run_client(config))
#     except asyncio.exceptions.CancelledError:
#         sys.exit(0)
