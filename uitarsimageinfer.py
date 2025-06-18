# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import re
import ast
import math

# Constants for image resizing logic
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 16384 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_RATIO = 200


def convert_point_to_coordinates(text, is_answer=False):
    # This function was likely for specific historical formats like <point>\d+ \d+</point>
    # The main parsing logic in parse_action_to_structure_output now handles coordinates
    # more directly. This function is kept for completeness but might not be the primary
    # coordinate parser used in the agent flow.

    # Pattern for integer coordinates like <point>200 300</point>
    pattern_int = r"<point>(\d+)\s+(\d+)</point>"
    text = re.sub(pattern_int, r"(\1,\2)", text) # Replace with tuple string (x,y)

    # Pattern for float coordinates like <point>X: 0.5 Y: 0.5</point>
    pattern_float = r"<point>X: (\d+\.?\d*) Y: (\d+\.?\d*)</point>"
    text = re.sub(pattern_float, r"(\1,\2)", text) # Replace with tuple string (x,y)

    # Remove [EOS] marker
    text = re.sub(r"\[EOS\]", "", text)

    return text.strip()


def parse_action(action_str):
    """
    Parses a single action string (e.g., "click(point='...')") into a structure.
    Uses ast.parse in exec mode to handle the string as a statement.
    Safely evaluates parameter values using ast.literal_eval.
    """
    action_str = action_str.strip()
    if not action_str:
        return None # Handle empty strings

    try:
        # Parse string as an executable statement
        # Expected format is a single function call expression
        node = ast.parse(action_str, mode='exec')

        # Ensure it's a module with exactly one statement
        if not isinstance(node, ast.Module) or len(node.body) != 1:
            raise ValueError("Not a single statement")

        # Get the single statement node
        statement = node.body[0]

        # Ensure the statement is an Expression statement (Expr)
        if not isinstance(statement, ast.Expr):
            raise ValueError("Not an expression statement")

        # Get the expression, which should be a Call node
        call = statement.value

        # Ensure the expression is a function call
        if not isinstance(call, ast.Call):
            raise ValueError("Expression is not a function call")

        # Get function name
        func_name = None
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            # Handles methods like obj.method, gets 'method'
            func_name = call.func.attr
        # else: func_name remains None if target is more complex

        if func_name is None:
             raise ValueError(f"Could not determine function name from {ast.dump(call.func)}")


        # Get keyword arguments
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if key is None:
                 print(f"Warning: Skipping positional argument in action '{action_str}'")
                 continue # Skip positional args if any were somehow parsed here

            # Safely evaluate the value node using ast.literal_eval
            try:
                # ast.literal_eval can parse strings, numbers, tuples, lists, dicts, booleans, None
                value = ast.literal_eval(kw.value)
                kwargs[key] = value
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"Warning: Failed to safely evaluate argument '{key}' for value '{ast.dump(kw.value)}' in action '{action_str}': {e}")
                # Store the raw source if literal_eval fails, or None
                try:
                    kwargs[key] = ast.get_source_segment(action_str, kw.value)
                except Exception:
                     kwargs[key] = str(kw.value) # Fallback to string representation of node
                 # Decide if failure to evaluate should make the action invalid
                 # For now, we include the raw string/None and handle potential issues downstream

        # UI-TARS actions primarily use keyword arguments. Positional arguments are ignored by this parser logic.

        return {'function': func_name, 'args': kwargs}

    except Exception as e:
        # Print the action string that caused the failure for easier debugging
        print(f"Failed to parse action string '{action_str}': {e}")
        return None


def escape_single_quotes(text):
    """Escapes single quotes within a string for safe use in Python string literals."""
    # Replace a single quote ' with \' but not if it's already escaped \\'
    # This regex matches ' that is not preceded by an odd number of backslashes.
    # A simpler robust way for most cases is just replacing ' with \'.
    # Let's stick to replacing single ' with \'.
    return text.replace("'", "\\'")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    if factor == 0: return number # Avoid division by zero
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
    if factor == 0: return number
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' that is divisible by 'factor'."""
    if factor == 0: return number
    return math.floor(number / factor) * factor


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Rescales the image so that the dimensions are divisible by 'factor',
    total pixels are within min/max, and aspect ratio is maintained/limited.
    This calculates the expected input dimensions of the model's vision encoder.
    """
    if min(height, width) <= 0:
         print(f"Warning: Got non-positive dimensions ({width}x{height}) for smart_resize. Returning minimal valid size.")
         return max(factor, 1), max(factor, 1)

    # Limit aspect ratio first
    current_ratio = max(height, width) / min(height, width)
    if current_ratio > MAX_RATIO:
        print(f"Warning: Aspect ratio {current_ratio:.2f} exceeds limit {MAX_RATIO}. Adjusting dimensions.")
        if height > width:
             height = int(width * MAX_RATIO)
        else:
             width = int(height * MAX_RATIO)
        # Recalculate current_ratio for info only
        current_ratio = max(height, width) / min(height, width)
        print(f"Adjusted dimensions for ratio limit: {width}x{height} (ratio: {current_ratio:.2f})")


    # Ensure dimensions are at least 'factor' and rounded by factor
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    current_pixels = h_bar * w_bar

    # Adjust based on pixel limits while maintaining aspect ratio relative to original dims
    original_pixels = height * width
    if current_pixels > max_pixels:
        print(f"Warning: Pixels {current_pixels} exceed max limit {max_pixels}. Scaling down.")
        scale_factor = math.sqrt(max_pixels / original_pixels)
        h_bar = floor_by_factor(height * scale_factor, factor)
        w_bar = floor_by_factor(width * scale_factor, factor)
        # Ensure they are still at least 'factor' after flooring
        h_bar = max(factor, h_bar)
        w_bar = max(factor, w_bar)
        print(f"Scaled down dims: {w_bar}x{h_bar}")

    elif current_pixels < min_pixels:
        print(f"Warning: Pixels {current_pixels} are below min limit {min_pixels}. Scaling up.")
        scale_factor = math.sqrt(min_pixels / original_pixels)
        h_bar = ceil_by_factor(height * scale_factor, factor)
        w_bar = ceil_by_factor(width * scale_factor, factor)
        # Ensure they are still at least 'factor' after ceiling
        h_bar = max(factor, h_bar)
        w_bar = max(factor, w_bar)
        print(f"Scaled up dims: {w_bar}x{h_bar}")

    # Final check to ensure divisibility by factor and minimum size after potential adjustments
    h_bar = max(factor, round_by_factor(h_bar, factor))
    w_bar = max(factor, round_by_factor(w_bar, factor))

    return h_bar, w_bar


def parse_action_to_structure_output(text,
                                     factor, # Appears unused for model_type="qwen25vl" coord scaling
                                     origin_resized_height, # Height of image as model likely processed it (from smart_resize)
                                     origin_resized_width,  # Width of image as model likely processed it (from smart_resize)
                                     model_type="qwen25vl", # Model type affects coordinate interpretation
                                     max_pixels=None, # Not directly used here, passed to smart_resize if needed
                                     min_pixels=None):# Not directly used here, passed to smart_resize if needed
    """
    Parses the raw model output text into a structured list of action dictionaries.
    Extracts Thought/Reflection/Summary, parses action calls, and normalizes
    coordinate parameters.
    """
    text = text.strip()
    if not text:
        return [] # Return empty list if text is empty

    # --- Handle Thought/Reflection/Action_Summary ---
    reflection, thought = None, None
    action_part = text # Assume the whole text is the action part initially

    # Regex patterns to find Thought, Reflection, Action_Summary, and Action parts
    # Use non-greedy matching and lookahead assertions to delineate sections
    # Capture everything after the label until the next known label or end of string
    reflection_pattern = r"Reflection: (.+?)(?=Action_Summary: |Thought: |Action: |$)"
    action_summary_pattern = r"Action_Summary: (.+?)(?=Thought: |Action: |$)"
    thought_pattern = r"Thought: (.+?)(?=Action: |$)"
    action_prefix_pattern = r"Action: (.*)" # Capture everything after "Action: " prefix

    # Find parts. The order matters for parsing the text sequentially.
    # Start by trying to match "Action: " prefix to separate action from thought sections.
    action_match = re.search(action_prefix_pattern, text, re.DOTALL)

    if action_match:
         action_part = action_match.group(1).strip()
         # The text before "Action: " might contain Thought/Reflection/Summary
         thought_section = text[:action_match.start()].strip()

         # Now parse the thought section
         reflection_match = re.search(reflection_pattern, thought_section, re.DOTALL)
         if reflection_match:
             reflection = reflection_match.group(1).strip()

         # Action_Summary and Thought might appear in the remaining part
         # Prioritize finding Thought after removing Reflection
         thought_section_remaining = re.sub(reflection_pattern, '', thought_section, count=1, flags=re.DOTALL).strip()

         action_summary_match = re.search(action_summary_pattern, thought_section_remaining, re.DOTALL)
         if action_summary_match:
             action_summary = action_summary_match.group(1).strip()

         # Find Thought in the remaining section
         thought_section_after_summary = re.sub(action_summary_pattern, '', thought_section_remaining, count=1, flags=re.DOTALL).strip()
         thought_match = re.search(thought_pattern, thought_section_after_summary, re.DOTALL)
         if thought_match:
              thought = thought_match.group(1).strip()
         # If Action_Summary was found but not Thought, put summary into thought
         elif action_summary is not None:
              thought = f"Action Summary: {action_summary}"
         # If both were found, combine
         if action_summary is not None and thought is not None and thought != f"Action Summary: {action_summary}":
              thought = f"Action Summary: {action_summary}\nThought: {thought}"


    else:
        # If "Action: " is not found, the format is wrong, or it's just thought/reflection without action
        print(f"Warning: Could not find 'Action: ' prefix in model output. No action will be parsed.")
        print(f"Output was:\n{text}")
        return [] # Return empty list if no action section


    # --- Parse Action String(s) ---
    # Split multiple actions separated by ")\n\n"
    # Use a regex that matches `)\n\n` but not inside quoted strings, or handle escaping.
    # A simple split might break if `)\n\n` appears in a 'type' content, even if escaped.
    # Let's use a lookahead based split to try and be safer, or process each line.
    # Given the examples, `)\n\n` seems to be the reliable separator between distinct action calls.

    # Let's try splitting based on finding patterns that look like start of an action `\nfunction_name(` after a `)`
    # This is still heuristic. A perfect parser might require a grammar definition.
    # For now, stick to splitting by ")\n\n" as it's common, but acknowledge fragility.
    raw_action_strings = action_part.split(")\n\n")

    # Ensure each split part ends with ')' if it was removed by the split, then trim whitespace.
    action_strings = [s.strip() + ')' if s.strip() and not s.strip().endswith(')') else s.strip() for s in raw_action_strings]
    action_strings = [s for s in action_strings if s] # Filter out any empty strings

    parsed_actions_list = [parse_action(action_str) for action_str in action_strings]

    actions = []
    for action_instance, raw_action_str in zip(parsed_actions_list, action_strings):
        if action_instance is None:
            print(f"Skipping unparseable action string: {raw_action_str}")
            continue # Skip this action if parsing failed

        action_type = action_instance.get("function")
        params = action_instance.get("args", {})

        action_inputs = {}
        for param_name, param_value in params.items():
            # Skip parameters with None value (e.g., if ast.literal_eval failed)
            if param_value is None:
                 print(f"Warning: Skipping None parameter '{param_name}' in action '{raw_action_str}'")
                 continue

            # Process coordinate parameters
            # Check for common coordinate parameter names
            is_coordinate_param = False
            normalized_coord_key = None # Key to use in action_inputs (e.g., 'start_point')

            if "box" in param_name:
                 is_coordinate_param = True
                 normalized_coord_key = param_name.strip() # Keep original box key if it contains 'box'
            elif "point" in param_name:
                 is_coordinate_param = True
                 normalized_coord_key = param_name.strip() # Keep original point key if it contains 'point'
                 if normalized_coord_key == 'point': # Map generic 'point' to 'start_point'
                      normalized_coord_key = 'start_point'
            elif param_name.strip() == 'start': # Map 'start' to 'start_point'
                 is_coordinate_param = True
                 normalized_coord_key = 'start_point'
            elif param_name.strip() == 'end': # Map 'end' to 'end_point'
                 is_coordinate_param = True
                 normalized_coord_key = 'end_point'


            if is_coordinate_param:
                # Param_value is already evaluated by ast.literal_eval in parse_action
                # It should be a string (e.g., '(319,57)') or a tuple/list (e.g., (319, 57) or [0.5, 0.5])
                coord_data = param_value

                # Convert the coordinate data to a standard format (list of floats [x,y] or [x1,y1,x2,y2])
                # and scale to 0-1 relative if they seem to be in model pixel space.
                scaled_numbers = None
                try:
                    numbers = []
                    if isinstance(coord_data, str):
                        # If it's still a string, try evaluating it as a tuple/list/number string
                        # This covers cases like point='(319,57)' where the value was parsed as string '(319,57)'
                        evaled_string_data = ast.literal_eval(coord_data)
                        if isinstance(evaled_string_data, (tuple, list)):
                            numbers = [float(n) for n in evaled_string_data]
                        elif isinstance(evaled_string_data, (int, float)):
                             # Handle cases where a single number might be given? (Unlikely for coords)
                             numbers = [float(evaled_string_data)]
                        else:
                             raise ValueError(f"Evaluated string is not a list/tuple/number: {type(evaled_string_data).__name__}")
                    elif isinstance(coord_data, (tuple, list)):
                         # If it's already a tuple or list
                         numbers = [float(n) for n in coord_data]
                    elif isinstance(coord_data, (int, float)):
                         # If it's a raw number (less likely for coords)
                         numbers = [float(coord_data)]
                    else:
                        raise ValueError(f"Coordinate value is unexpected type: {type(coord_data).__name__}")

                    # Now 'numbers' is a list of floats [x,y] or [x1,y1,x2,y2] (potentially in model pixel space)

                    # --- Scaling Logic ---
                    # Assume numbers are pixels relative to the model's input image dimensions.
                    # Scale these pixels to 0-1 relative coordinates.
                    # Need origin_resized_width/height from smart_resize output
                    if model_type == "qwen25vl" and origin_resized_width > 0 and origin_resized_height > 0:
                        if len(numbers) == 2: # (x, y) point in model pixel space
                             x, y = numbers
                             # Scale to 0-1 relative
                             scaled_numbers = [x / origin_resized_width, y / origin_resized_height]
                        elif len(numbers) == 4: # (x1, y1, x2, y2) box in model pixel space
                             x1, y1, x2, y2 = numbers
                             # Scale to 0-1 relative
                             scaled_numbers = [x1 / origin_resized_width, y1 / origin_resized_height,
                                               x2 / origin_resized_width, y2 / origin_resized_height]
                        else:
                             print(f"Warning: Coordinate numbers list has unexpected length ({len(numbers)}) for model type {model_type}: {numbers}")
                             scaled_numbers = None # Indicate failure
                    else:
                         # Handle cases where model_type is different or dimensions are invalid
                         print(f"Warning: Cannot scale coordinates for model type '{model_type}' or invalid dimensions ({origin_resized_width}x{origin_resized_height}). Using raw numbers.")
                         scaled_numbers = numbers # Use raw numbers if scaling logic isn't applicable/possible


                    if scaled_numbers is not None:
                        # Store the scaled 0-1 relative coordinates as a string representation of a list
                        action_inputs[normalized_coord_key] = str(scaled_numbers)
                        # print(f"Parsed and scaled '{coord_data}' (from param '{param_name}') -> {action_inputs[normalized_coord_key]}") # Debugging print
                    else:
                        action_inputs[normalized_coord_key] = None # Indicate failure
                        print(f"Failed to parse/scale coordinate data '{coord_data}' (from param '{param_name}').")

                except (ValueError, SyntaxError, TypeError) as e:
                    print(f"Error parsing/scaling coordinate data '{coord_data}' (from param '{param_name}'): {e}")
                    action_inputs[normalized_coord_key] = None # Indicate parsing failure

            # Process other parameters (like 'content', 'direction', 'key', 'hotkey')
            else:
                # For 'content', apply escape_single_quotes
                if param_name == 'content' and isinstance(param_value, str):
                    action_inputs[param_name.strip()] = escape_single_quotes(param_value)
                else:
                    # Store other parameters as they are (strings, numbers, booleans etc. from ast.literal_eval)
                    action_inputs[param_name.strip()] = param_value

        actions.append({
            "reflection": reflection, # Include Reflection if found
            "thought": thought,     # Include Thought (and Action_Summary) if found
            "action_type": action_type,
            "action_inputs": action_inputs,
            # "raw_action_string": raw_action_str # Optional: Keep the raw action string
        })
    return actions


def parsing_response_to_pyautogui_code(responses,
                                       image_height: int, # Original screenshot height (pixels)
                                       image_width: int,  # Original screenshot width (pixels)
                                       input_swap: bool = True) -> str:
    '''
    Takes a list of parsed action dictionaries (output of parse_action_to_structure_output)
    and converts them into a string of pyautogui code.
    Assumes coordinate values in action_inputs are string representations of
    0-1 relative coordinate lists ([x,y] or [x1,y1,x2,y2]).

    Args:
        responses: A list of parsed action dictionaries.
        image_height: The height of the original screenshot (for scaling).
        image_width: The width of the original screenshot (for scaling).
        input_swap: Whether to use clipboard for 'type' action.
    Returns:
        Generated pyautogui code string, or "DONE" if a finished action is present.
    '''

    pyautogui_code_lines = ["import pyautogui", "import time"]
    # Add pyperclip import only if input_swap is True
    if input_swap:
        pyautogui_code_lines.append("import pyperclip")

    # Ensure responses is a list
    if isinstance(responses, dict):
        responses = [responses]

    # Include thought/reflection in comments for the first action block if available
    # Check the first item for thought/reflection/observation keys
    if responses and (responses[0].get("reflection") or responses[0].get("thought") or responses[0].get("observation")):
         reflection_text = responses[0].get("reflection", "")
         thought_text = responses[0].get("thought", "")
         observation_text = responses[0].get("observation", "") # Assuming observation might be added later

         comment_block_lines = ["'''"]
         if observation_text:
             comment_block_lines.append("Observation:\n" + observation_text)
         if reflection_text:
             if len(comment_block_lines) > 1: comment_block_lines.append("") # Add newline if preceded by content
             comment_block_lines.append("Reflection:\n" + reflection_text)
         if thought_text:
             if len(comment_block_lines) > 1: comment_block_lines.append("") # Add newline if preceded by content
             comment_block_lines.append("Thought:\n" + thought_text)
         comment_block_lines.append("'''")

         if len(comment_block_lines) > 2: # Only add if there's content beyond just the quotes
             pyautogui_code_lines.append("\n".join(comment_block_lines))


    # --- Helper to get pixel coordinates from action_inputs ---
    def get_pixel_coords_from_input(action_inputs, possible_keys, width, height):
         """
         Looks for coordinate string (string repr of [x,y] or [x1,y1,x2,y2] 0-1 relative)
         in action_inputs using possible_keys and scales to pixels.
         Returns pixel coords (px1, py1, px2, py2) or (None, None, None, None).
         """
         coord_list_str = None
         found_key = None
         for key in possible_keys:
              coord_list_str = action_inputs.get(key)
              if coord_list_str is not None:
                   found_key = key
                   break # Found a coordinate key

         if coord_list_str is None:
              # print(f"Debug: No coordinate key found among {possible_keys} in inputs {action_inputs}") # Too verbose
              return None, None, None, None # No coordinate found

         # Ensure the found value is actually a string before trying to eval
         if not isinstance(coord_list_str, str):
              print(f"Warning: Value for key '{found_key}' was not a string: {coord_list_str} (type: {type(coord_list_str).__name__})")
              return None, None, None, None

         try:
             # Evaluate the string "[x,y]" or "[x1,y1,x2,y2]" into a Python list
             # The string comes from parse_action_to_structure_output and should be 0-1 relative
             coords_rel = eval(coord_list_str)

             if isinstance(coords_rel, (list, tuple)):
                 if len(coords_rel) == 2:
                     # It's a point [x_rel, y_rel] (0-1 relative)
                     x_rel, y_rel = coords_rel
                     px = round(float(x_rel) * width, 3)
                     py = round(float(y_rel) * height, 3)
                     # Ensure coordinates are within screen bounds (optional safety)
                     px = max(0, min(px, width - 1))
                     py = max(0, min(py, height - 1))
                     return px, py, px, py # Return point as min/max for consistency
                 elif len(coords_rel) == 4:
                     # It's a box [x1_rel, y1_rel, x2_rel, y2_rel] (0-1 relative)
                     x1_rel, y1_rel, x2_rel, y2_rel = coords_rel
                     px1 = round(float(x1_rel) * width, 3)
                     py1 = round(float(y1_rel) * height, 3)
                     px2 = round(float(x2_rel) * width, 3)
                     py2 = round(float(y2_rel) * height, 3)
                     # Ensure coordinates are within screen bounds (optional safety)
                     px1 = max(0, min(px1, width - 1))
                     py1 = max(0, min(py1, height - 1))
                     px2 = max(0, min(px2, width - 1))
                     py2 = max(0, min(py2, height - 1))
                     return px1, py1, px2, py2 # Return box coordinates
                 else:
                     print(f"Warning: get_pixel_coords_from_input received list of unexpected length ({len(coords_rel)}) for key '{found_key}': {coords_rel}")
                     return None, None, None, None
             else:
                 print(f"Warning: get_pixel_coords_from_input received non-list/tuple after eval for key '{found_key}': {type(coords_rel).__name__} = {coords_rel}")
                 return None, None, None, None

         except Exception as e:
             print(f"Error evaluating coordinate string '{coord_list_str}' for key '{found_key}': {e}")
             return None, None, None, None # Indicate failure


    # --- Action Type Handling ---
    for response_id, response in enumerate(responses):
        action_type = response.get("action_type")
        action_inputs = response.get("action_inputs", {})

        # Add a sleep before subsequent actions
        if response_id > 0:
            pyautogui_code_lines.append(f"\ntime.sleep(0.5)") # Add sleep before action
            pyautogui_code_lines.append(f"# --- Action {response_id + 1} ({action_type}) ---")
            # Optionally add thought/reflection for this action block if it was present (currently only added for the first)

        # Action Type specific code generation
        if action_type == "hotkey":
            key_param = action_inputs.get("key") or action_inputs.get("hotkey")
            if key_param and isinstance(key_param, str):
                keys = key_param.split()
                convert_keys = []
                for key in keys:
                    # Map common names to pyautogui keys
                    key_lower = key.lower()
                    if key_lower == "space": key = ' '
                    elif key_lower == "arrowleft": key = "left"
                    elif key_lower == "arrowright": key = "right"
                    elif key_lower == "arrowup": key = "up"
                    elif key_lower == "arrowdown": key = "down"
                    # Add more mappings if needed (e.g., esc -> escape, enter -> enter, win -> win)
                    elif key_lower == "esc": key = "escape"
                    elif key_lower == "enter": key = "enter"
                    elif key_lower == "win": key = "win" # Windows key
                    elif key_lower == "cmd": key = "command" # Mac command key

                    convert_keys.append(key)

                if convert_keys:
                     # Use repr() to ensure strings are quoted correctly in the generated code
                     pyautogui_code_lines.append(f"pyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})")
                else:
                     pyautogui_code_lines.append(f"# Hotkey action with empty or invalid keys: {key_param}")
            else:
                 pyautogui_code_lines.append(f"# Hotkey action missing 'key' or 'hotkey' parameter.")


        elif action_type in ["press", "keydown", "release", "keyup"]:
            # Note: 'press' is usually a single key press/release cycle in pyautogui.
            # The action space lists 'press', 'keydown', 'release'. Let's map:
            # 'press' -> keyDown then keyUp
            # 'keydown' -> keyDown
            # 'release' -> keyUp
            # 'keyup' -> keyUp (add alias)

            key_param = action_inputs.get("key") or action_inputs.get(action_type) # Check 'key' or action_type name
            if key_param and isinstance(key_param, str):
                key = key_param.lower()
                if key == "arrowleft": key = "left"
                elif key == "arrowright": key = "right"
                elif key == "arrowup": key = "up"
                elif key == "arrowdown": key = "down"
                elif key == "space": key = " "
                elif key == "esc": key = "escape"
                elif key == "enter": key = "enter"

                if action_type == "press":
                    pyautogui_code_lines.append(f"pyautogui.press({repr(key)})") # pyautogui.press handles down/up
                elif action_type == "keydown":
                    pyautogui_code_lines.append(f"pyautogui.keyDown({repr(key)})")
                elif action_type in ["release", "keyup"]:
                     pyautogui_code_lines.append(f"pyautogui.keyUp({repr(key)})")
            else:
                 pyautogui_code_lines.append(f"# {action_type.capitalize()} action missing 'key' parameter.")


        elif action_type == "type":
            content = action_inputs.get("content", "")
            # content should already have single quotes escaped from parse_action_to_structure_output
            content = str(content) if content is not None else ""

            ends_with_newline = content.endswith("\n") or content.endswith("\\n")
            stripped_content = content.rstrip("\\n").rstrip("\n") if ends_with_newline else content

            # Use repr() for the content string to handle internal quotes/backslashes correctly in the generated code
            content_repr = repr(stripped_content)

            if stripped_content:
                if input_swap:
                    # Ensure pyperclip is imported
                    if "import pyperclip" not in pyautogui_code_lines:
                         pyautogui_code_lines.insert(0, "import pyperclip")

                    pyautogui_code_lines.append(f"pyperclip.copy({content_repr})")
                    pyautogui_code_lines.append(f"pyautogui.hotkey('ctrl', 'v')")
                    pyautogui_code_lines.append(f"time.sleep(0.1)") # Short sleep after paste
                else:
                    # pyautogui.write does not interpret \n as Enter, it types \ then n.
                    # Need to handle newlines explicitly if not using input_swap.
                    # A simple approach: type content, then press enter if needed.
                    # Or, replace \n with actual Enter key press sequence.
                    # Let's use the simple approach: type stripped content, then press enter.
                    # If you need complex typing with newlines mid-string without input_swap,
                    # this would need more sophisticated handling (e.g., splitting by \n and typing chunks).
                    pyautogui_code_lines.append(f"pyautogui.write({content_repr}, interval=0.01)")
                    pyautogui_code_lines.append(f"time.sleep(0.1)")

            if ends_with_newline:
                 pyautogui_code_lines.append(f"pyautogui.press('enter')")
                 pyautogui_code_lines.append(f"time.sleep(0.1)") # Short sleep after enter


        elif action_type in ["drag", "select"]:
            # Look for 'start_point' or 'start_box'
            sx1, sy1, sx2, sy2 = get_pixel_coords_from_input(action_inputs, ['start_point', 'start_box'], image_width, image_height)
            # Look for 'end_point' or 'end_box'
            ex1, ey1, ex2, ey2 = get_pixel_coords_from_input(action_inputs, ['end_point', 'end_box'], image_width, image_height)

            if (sx1 is not None and sy1 is not None and ex1 is not None and ey1 is not None):
                # Use the center of the start area and end area for drag
                start_center_x = (sx1 + sx2) / 2
                start_center_y = (sy1 + sy2) / 2
                end_center_x = (ex1 + ex2) / 2
                end_center_y = (ey1 + ey2) / 2

                pyautogui_code_lines.append(f"pyautogui.moveTo({start_center_x}, {start_center_y})")
                pyautogui_code_lines.append(f"pyautogui.dragTo({end_center_x}, {end_center_y}, duration=0.5)") # Default duration 0.5s
            else:
                pyautogui_code_lines.append(f"# Failed to parse coordinates for {action_type}. Looked for start/end_point/box.")


        elif action_type == "scroll":
            # Look for the scroll target coordinate: 'point' or 'start_point' or 'start_box'
            px1, py1, px2, py2 = get_pixel_coords_from_input(action_inputs, ['point', 'start_point', 'start_box'], image_width, image_height)

            # Use the center if coordinate was found
            scroll_x, scroll_y = None, None
            if px1 is not None and py1 is not None:
                 scroll_x = (px1 + px2) / 2
                 scroll_y = (py1 + py2) / 2

            direction = action_inputs.get("direction", "").lower()
            scroll_amount = 10 # Default scroll amount (lines/units) - adjust as needed

            # Determine scroll direction and amount for pyautogui.scroll (vertical) or hscroll (horizontal)
            # pyautogui.scroll takes a positive number to scroll up, negative to scroll down.
            # pyautogui.hscroll takes a positive number to scroll right, negative to scroll left.
            v_scroll_amount = 0
            h_scroll_amount = 0

            if "up" in direction:
                v_scroll_amount = scroll_amount
            elif "down" in direction:
                v_scroll_amount = -scroll_amount
            elif "left" in direction:
                 h_scroll_amount = -scroll_amount
            elif "right" in direction:
                 h_scroll_amount = scroll_amount
            else:
                 pyautogui_code_lines.append(f"# Warning: Unknown scroll direction: {direction}. No scroll action generated.")
                 # Do nothing if direction is unclear

            if v_scroll_amount != 0:
                # pyautogui.scroll syntax: scroll(clicks, x=..., y=...)
                if scroll_x is not None and scroll_y is not None:
                    pyautogui_code_lines.append(f"pyautogui.scroll({v_scroll_amount}, x={scroll_x}, y={scroll_y})")
                else:
                    pyautogui_code_lines.append(f"pyautogui.scroll({v_scroll_amount})")
            elif h_scroll_amount != 0:
                 # pyautogui.hscroll syntax: hscroll(clicks, x=..., y=...)
                 if scroll_x is not None and scroll_y is not None:
                    pyautogui_code_lines.append(f"pyautogui.hscroll({h_scroll_amount}, x={scroll_x}, y={scroll_y})")
                 else:
                    pyautogui_code_lines.append(f"pyautogui.hscroll({h_scroll_amount})")


        elif action_type in [
                "click", "left_single", "left_double", "right_single", "hover", "long_press"
        ]:
            # Look for the primary target coordinate: 'point' or 'start_point' or 'start_box'
            px1, py1, px2, py2 = get_pixel_coords_from_input(action_inputs, ['point', 'start_point', 'start_box'], image_width, image_height)

            if px1 is not None and py1 is not None:
                 # Use the center of the area as the click/hover location
                 target_x = (px1 + px2) / 2
                 target_y = (py1 + py2) / 2

                 # Ensure mouse is at target before clicking/dragging
                 pyautogui_code_lines.append(f"pyautogui.moveTo({target_x}, {target_y})")
                 pyautogui_code_lines.append(f"time.sleep(0.1)") # Short pause after move

                 if action_type == "left_single" or action_type == "click":
                     pyautogui_code_lines.append(f"pyautogui.click(button='left')") # Click at current mouse position
                 elif action_type == "left_double":
                     pyautogui_code_lines.append(f"pyautogui.doubleClick(button='left')") # Double click at current position
                 elif action_type == "right_single":
                     pyautogui_code_lines.append(f"pyautogui.click(button='right')") # Right click at current position
                 elif action_type == "hover":
                     # moveTo already done, no further action needed for hover
                     pass
                 elif action_type == "long_press":
                     pyautogui_code_lines.append(f"pyautogui.mouseDown(button='left')")
                     pyautogui_code_lines.append(f"time.sleep(1.0)") # Long press duration (adjust if needed)
                     pyautogui_code_lines.append(f"pyautogui.mouseUp(button='left')")

            else:
                pyautogui_code_lines.append(f"# Failed to parse coordinates for {action_type}. Looked for point/start_point/start_box.")

        # Handle Mobile specific actions that might appear
        elif action_type in ["open_app", "press_home", "press_back"]:
             # These actions do not have standard PyAutoGUI equivalents for desktop control.
             # Add comments indicating they are mobile actions.
             pyautogui_code_lines.append(f"# Action '{action_type}' is typically for mobile tasks and is not implemented for computer control.")

        # The 'finished' action is handled at the beginning of the loop

        else:
            pyautogui_code_lines.append(f"\n# Unrecognized action type: {action_type}. No code generated.")

    return "\n".join(pyautogui_code_lines)


def add_box_token(input_string):
    # This function's purpose seems to be adding specific tokens around coordinates
    # in the text string itself, likely for visualization or training data prep.
    # It's not part of the core execution flow (model output -> parse -> action).
    # Keeping the original logic as provided, assuming its specific token requirements.

    # Look for the "Action: " part to separate it from Thought/Reflection
    parts = input_string.split("Action: ", 1)
    prefix = parts[0] + "Action: " if len(parts) > 1 else ""
    action_part = parts[-1] if len(parts) > 1 else input_string # If no "Action: ", process the whole string?

    # Split action part into individual action strings based on common separator
    # Using the heuristic ")\n\n" split, potentially fragile.
    raw_action_strings = action_part.split(")\n\n")

    processed_action_strings = []
    for action_str in raw_action_strings:
        action_str = action_str.strip()
        if not action_str.endswith(')'):
             action_str += ')' # Add back missing parenthesis if split removed it

        # Pattern to find parameter assignments like `param='value'`
        # Need to be careful not to match inside other quoted strings.
        # This regex finds `word='quoted_string'` or `word="quoted_string"`
        param_assignment_pattern = re.compile(r"(\w+)=(['\"])(.*?)\2", re.DOTALL) # Use DOTALL to match across lines

        updated_action = action_str
        # Iterate through parameter assignments within this action string
        for param_match in param_assignment_pattern.finditer(action_str):
             param_name = param_match.group(1)
             quote = param_match.group(2)
             param_value_content = param_match.group(3) # Content inside quotes
             original_param_assignment_text = param_match.group(0) # Full text `name='value'`

             # Check if this parameter name suggests it contains coordinates
             if "box" in param_name or "point" in param_name or param_name in ['start', 'end']:
                  # Now find the coordinate pattern *within* the param_value_content string
                  # Look for <point>...</point> or (N,N) or [N,N,N,N] etc.
                  # This regex looks for known coordinate formats within the parameter value string.
                  # It needs to be careful with nested quotes or escaped characters if the model outputs them.
                  # Assuming formats like: <point>N N</point>, <point>X: F Y: F</point>, (N,N), [N,N], (N,N,N,N), [N,N,N,N]
                  # Let's refine the coordinate pattern regex to be more specific and handle potential spaces around values.
                  coord_pattern_in_value = re.compile(
                      r"(<point>\s*X:\s*\d+\.?\d*\s*Y:\s*\d+\.?\d*\s*</point>|" # <point>X: F Y: F</point>
                      r"<point>\s*\d+\s+\d+\s*</point>|"                      # <point>N N</point>
                      r"\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\)|"                 # (N,N) or (F,F)
                      r"\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]|"                 # [N,N] or [F,F]
                      r"\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\)|" # (N,N,N,N) or (F,F,F,F)
                      r"\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]"  # [N,N,N,N] or [F,F,F,F]
                      r")"
                  )


                  updated_param_value_content = param_value_content
                  # Use findall to get all coordinate matches in the value
                  coord_matches_in_value = coord_pattern_in_value.findall(param_value_content)

                  # Iterate through found coordinate strings and replace them with tokenized version
                  # Need to be careful with replacement order if matches overlap (shouldn't happen with coords)
                  # or if the same coordinate string appears multiple times.
                  # Replacing using a temporary unique placeholder might be safest, or replace from end to start.
                  # Simple string replace might replace subsequent identical matches.
                  # Let's use finditer and replace in the original string portion.

                  # Rebuild the updated parameter value content piece by piece
                  last_end = 0
                  temp_updated_value_content = ""
                  for coord_match_iter in coord_pattern_in_value.finditer(param_value_content):
                       # Append text before this match
                       temp_updated_value_content += param_value_content[last_end : coord_match_iter.start()]
                       # Append the tokenized coordinate
                       found_coord_text = coord_match_iter.group(0)
                       temp_updated_value_content += f"<|box_start|>{found_coord_text}<|box_end|>"
                       last_end = coord_match_iter.end()

                  # Append any remaining text after the last match
                  temp_updated_value_content += param_value_content[last_end:]
                  updated_param_value_content = temp_updated_value_content


                  # If the parameter value content was updated, rebuild the full parameter assignment string
                  if updated_param_value_content != param_value_content:
                      new_param_assignment_text = f"{param_name}={quote}{updated_param_value_content}{quote}"
                      # Replace the original parameter assignment text with the new one in the action string
                      # Use replace with count=1 in case the same assignment appears multiple times (unlikely but safer)
                      updated_action = updated_action.replace(original_param_assignment_text, new_param_assignment_text, 1)

            # else: not a coordinate parameter, add as is

        processed_action_strings.append(updated_action)


    # Step 5: Reconstruct the final string
    # Add back the original prefix (Thought/Reflection/Action: ) if it existed
    final_string = prefix + "\n\n".join(processed_action_strings)

    return final_string