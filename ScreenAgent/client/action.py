import asyncio
from collections import OrderedDict, defaultdict
import enum
import queue
from abc import ABC, abstractmethod
from base import * # Assuming this imports Position, ClickableArea, VNCMouseButton, IncompleteActionDataError, MousePositionNotInClickableAreaWarning, ObjEncoder
import numpy as np
import zlib
import re # Import re for regex operations
import base64
from datetime import datetime
from typing import Union, Dict, List, Any
from functools import lru_cache, partial
from itertools import chain
from dataclasses import dataclass, asdict
import json # Ensure json is imported
import math # Import math for smart_resize_for_v15

import requests
import queue

from keysymdef import keysymdef  # type: ignore
vaild_keysymdef = [x[0] for x in keysymdef]
vaild_keysymdef_lower_map = {x.lower(): x for x in vaild_keysymdef}

# Assuming bleu is available and compute_bleu is a valid function
# from bleu import compute_bleu # Assuming this is available

# --- New dataclasses mirroring TypeScript interfaces ---

@dataclass
class ActionAttributeScore:
    attribute_name: str
    pred:str
    label:str

@dataclass
class ActionValueScore:
    attribute_name: str
    score: float
    metric: str
    pred: Union[str, int, float, None] = None
    label: Union[str, int, float, None] = None

# similarity 由 ActionAttributeScore 和 ActionValueScore 构成
@dataclass
class ActionSimilarity:
    score_point:int
    scores: List[Union[ActionAttributeScore, ActionValueScore]]

    def get_score(self):
        if self.score_point == 0:
            return 0.0

        action_score = 0.0
        for score in self.scores:
            if isinstance(score, ActionAttributeScore):
                if score.pred == score.label:
                    action_score += 1.0
            elif isinstance(score, ActionValueScore):
                action_score += score.score

        return action_score


# This dataclass mirrors the PredictionParsed interface in actionParser.ts
@dataclass
class PredictionParsed:
    reflection: Union[str, None]
    thought: str
    action_type: str
    action_inputs: Dict[str, Any]


# --- Existing Action Class Hierarchy (Keep as is) ---

class ActionMeta(type):
    def __new__(cls, name, bases, attrs):
        ordered_save_attrs = []
        for base in bases:
            if hasattr(base, 'save_attributes'):
                ordered_save_attrs.extend(base.save_attributes)

        if 'save_attributes' in attrs:
            for attr in attrs['save_attributes']:
                if attr not in ordered_save_attrs:
                    ordered_save_attrs.append(attr)

        attrs['save_attributes'] = ordered_save_attrs
        return super().__new__(cls, name, bases, attrs)

    def from_json(cls, json_dict):
        action_type = json_dict.get("action_type", None)
        if action_type is None:
            return None
        action_class = globals().get(action_type, None)
        if action_class is None:
            return None
        json_dict.pop("action_type")
        try:
            valid_params = set(action_class.__init__.__code__.co_varnames)
            valid_params.discard('self')
            if not any(param.startswith('**') for param in action_class.__init__.__code__.co_varnames):
                json_dict = {k: v for k, v in json_dict.items() if k in valid_params}

            action = action_class(**json_dict)
        except (TypeError, AttributeError, IncompleteActionDataError, KeyError) as e:
            # print(f"Error creating action {action_type} from json {json_dict}: {e}") # Optional: for debugging
            return None
        return action


class Action(metaclass=ActionMeta):
    # MicroAction class
    base_attributes = ["action_time", "before_action_obs", "after_action_obs"]
    save_attributes = ["action_time", "before_action_obs", "after_action_obs"]
    base64_attributes = ["before_action_obs", "after_action_obs"]
    is_required_update = True
    request_id = None

    def __init__(self, action_time=None, before_action_obs=None, after_action_obs=None):
        self.action_time = action_time
        self.before_action_obs = before_action_obs
        self.after_action_obs = after_action_obs

    @property
    def action_type(self):
        return type(self).__name__

    @abstractmethod
    async def step(self, vnc):
        pass

    def save_action(self):
        # return a dict save_attributes
        dic = {}
        for attr in self.save_attributes:
            dic[attr] = getattr(self, attr)
        dic["action_type"] = type(self).__name__
        return dic

    def __str__(self):
        attrs = []
        for attr in self.save_attributes:
            value = getattr(self, attr)
            if attr in ["before_action_obs", "after_action_obs"]:
                if isinstance(value, np.ndarray):
                    attrs.append(f"{attr}=shape{value.shape}")
            elif isinstance(value, str):
                # Truncate long strings for display
                display_value = value if len(value) < 50 else f"{value[:47]}..."
                attrs.append(f"{attr}='{display_value}'")
            else:
                attrs.append(f"{attr}={value}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def __repr__(self):
        return self.to_ideal_display_format()

    def to_ideal_dict_format(self):
        dic = OrderedDict()
        dic["action_type"] = type(self).__name__
        for attr in type(self).save_attributes:
            if attr not in self.base_attributes:
                value = getattr(self, attr)
                if value is not None:
                    if isinstance(value, enum.Enum):
                        value = value.name
                    elif isinstance(value, np.ndarray):
                        continue
                    if isinstance(value, enum.Enum):
                         value = value.name
                    elif isinstance(value, Position):
                         value = asdict(value)
                    elif isinstance(value, ClickableArea):
                         value = asdict(value)
                    dic[attr] = value
        return dic

    def to_ideal_json_format(self):
        dic = self.to_ideal_dict_format()
        try:
            return json.dumps(dic, cls=ObjEncoder, ensure_ascii=False)
        except TypeError:
            print(f"Warning: ObjEncoder failed for {self}. Falling back to default serialization.")
            return json.dumps(dic, ensure_ascii=False)


    def to_ideal_display_format(self):
        attrs = []
        for attr in self.save_attributes:
            if attr not in self.base_attributes:
                value = getattr(self, attr)
                if value is not None:
                    if isinstance(value, enum.Enum):
                        value = value.name
                    elif isinstance(value, np.ndarray):
                        continue
                    display_value = str(value)
                    if len(display_value) > 30:
                        display_value = f"{display_value[:27]}..."
                    attrs.append(display_value)
        return f"{self.__class__.__name__}({', '.join(attrs)})"


    @property
    def score_point(self):
        return 0

    @abstractmethod
    def similarity(self, pred_action):
        pass

class MouseActionType(enum.Enum):
    down = 0
    up = 1
    scroll_up = 2
    scroll_down = 3
    move = 4
    drag = 5
    click = 6
    double_click = 7


class MouseAction(Action):

    save_attributes = ("mouse_action_type", "mouse_button", "mouse_position", "scroll_repeat", "clickable_area")
    is_required_update = False

    def __init__(self, mouse_action_type: Union[MouseActionType, str] = None, mouse_button: Union[VNCMouseButton, str] = None, mouse_position: Union[Position, Dict] = None, scroll_repeat: int = None, clickable_area:ClickableArea=None, **kwargs):
        self.mouse_action_type = mouse_action_type
        if isinstance(mouse_action_type, str):
            try:
                self.mouse_action_type = MouseActionType[mouse_action_type.lower()]
            except KeyError:
                 raise IncompleteActionDataError(f"Invalid MouseActionType string: {mouse_action_type}")

        self.mouse_button = mouse_button
        if isinstance(mouse_button, str):
            try:
                self.mouse_button = VNCMouseButton[mouse_button.lower()]
            except KeyError:
                 print(f"Warning: Invalid VNCMouseButton string: {mouse_button}")
                 self.mouse_button = None


        self.mouse_position = mouse_position
        if isinstance(mouse_position, dict):
            self.mouse_position = Position(**mouse_position)
        elif mouse_position is not None and not isinstance(mouse_position, Position):
             raise IncompleteActionDataError(f"Invalid mouse_position type: {type(mouse_position)}")

        self.scroll_repeat = scroll_repeat

        if isinstance(clickable_area, dict):
            self.clickable_area = ClickableArea.from_json(clickable_area)
        else:
            self.clickable_area = clickable_area


        super().__init__(**kwargs)

        if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.drag] and self.mouse_button is None:
             raise IncompleteActionDataError(f"MouseAction of type {self.mouse_action_type.name} requires mouse_button")
        if self.mouse_action_type in [MouseActionType.move, MouseActionType.drag, MouseActionType.click, MouseActionType.double_click] and self.mouse_position is None:
             if self.mouse_action_type in [MouseActionType.move, MouseActionType.drag]:
                  raise IncompleteActionDataError(f"MouseAction of type {self.mouse_action_type.name} requires mouse_position")


    async def step(self, vnc):
        if self.mouse_action_type is None:
            raise IncompleteActionDataError("MouseAction has no mouse_action_type")

        if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.drag] and self.mouse_button is None:
             raise IncompleteActionDataError(f"MouseAction of type {self.mouse_action_type.name} requires mouse_button for step")

        if self.mouse_action_type in [MouseActionType.move, MouseActionType.drag, MouseActionType.click, MouseActionType.double_click] and self.mouse_position is None:
             if self.mouse_action_type in [MouseActionType.move, MouseActionType.drag]:
                  raise IncompleteActionDataError(f"MouseAction of type {self.mouse_action_type.name} requires mouse_position for step")
             raise IncompleteActionDataError(f"MouseAction of type {self.mouse_action_type.name} requires mouse_position for step")


        if self.mouse_action_type == MouseActionType.down:
            vnc.mouse.down(self.mouse_button.value)
        elif self.mouse_action_type == MouseActionType.up:
            vnc.mouse.up(self.mouse_button.value)
        elif self.mouse_action_type == MouseActionType.scroll_up:
            if self.scroll_repeat is None:
                self.scroll_repeat = 1
            vnc.mouse.scroll_up(repeat=self.scroll_repeat)
        elif self.mouse_action_type == MouseActionType.scroll_down:
            if self.scroll_repeat is None:
                self.scroll_repeat = 1
            vnc.mouse.scroll_down(repeat=self.scroll_repeat)
        elif self.mouse_action_type == MouseActionType.move:
            vnc.mouse.move(self.mouse_position.width, self.mouse_position.height)
        elif self.mouse_action_type == MouseActionType.drag:
            vnc.mouse.down(self.mouse_button.value)
            vnc.mouse.move(self.mouse_position.width, self.mouse_position.height)
            vnc.mouse.up(self.mouse_button.value)
        elif self.mouse_action_type == MouseActionType.click:
            vnc.mouse.move(self.mouse_position.width, self.mouse_position.height)
            vnc.mouse.click(self.mouse_button.value)
        elif self.mouse_action_type == MouseActionType.double_click:
            vnc.mouse.move(self.mouse_position.width, self.mouse_position.height)
            vnc.mouse.click(self.mouse_button.value)
            vnc.mouse.click(self.mouse_button.value)
        else:
            raise IncompleteActionDataError("MouseAction step logic failed for type: {}".format(self.mouse_action_type))


    def set_clickable_area(self, area:ClickableArea):
        self.clickable_area = area


    @property
    def score_point(self):
        score_point = 1
        if self.mouse_action_type is not None:
            score_point += 1

        if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.drag] and self.mouse_button is not None:
            score_point += 1

        if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.move, MouseActionType.drag] and self.clickable_area is not None:
             score_point += 1

        return score_point

    def similarity(self, pred_action: Action):
        scores = [
            ActionAttributeScore("action_type", pred_action.action_type if pred_action else None, self.action_type),
        ]

        if not isinstance(pred_action, MouseAction):
             if self.mouse_action_type is not None:
                  scores.append(ActionAttributeScore("mouse_action_type", None, self.mouse_action_type.name))
             if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.drag] and self.mouse_button is not None:
                  scores.append(ActionAttributeScore("mouse_button", None, self.mouse_button.name))
             if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.move, MouseActionType.drag] and self.clickable_area is not None:
                  scores.append(ActionValueScore("mouse_position", 0.0, "clickable_area", None, None))
             return ActionSimilarity(self.score_point, scores)


        scores.append(ActionAttributeScore("mouse_action_type", pred_action.mouse_action_type.name if pred_action.mouse_action_type else None, self.mouse_action_type.name if self.mouse_action_type else None))

        if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.drag] and self.mouse_button is not None:
             scores.append(ActionAttributeScore("mouse_button", pred_action.mouse_button.name if pred_action.mouse_button else None, self.mouse_button.name))

        if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.move, MouseActionType.drag] and self.clickable_area is not None:
             if pred_action.mouse_position is not None:
                 is_mouse_position_in_clickable_area = pred_action.mouse_position in self.clickable_area
                 scores.append(ActionValueScore("mouse_position", 1.0 if is_mouse_position_in_clickable_area else 0.0, "clickable_area", pred_action.mouse_position, self.clickable_area))
             else:
                 scores.append(ActionValueScore("mouse_position", 0.0, "clickable_area", None, self.clickable_area))


        return ActionSimilarity(self.score_point, scores)

class KeyboardActionType(enum.Enum):
    down = 0
    up = 1
    press = 2
    text = 3 # Maps to 'input' in TS parser sometimes

class KeyboardAction(Action):
    save_attributes = ["keyboard_action_type", "keyboard_key", "keyboard_text"]
    convert_dict = {
        "win": "Super_L",
        "windows": "Super_L",
        "windowskey": "Super_L",
        "windows key": "Super_L",
        "Windows key": "Super_L",
        "winkey": "Super_L",
        "ctrl": "Control_L",
        "alt": "Alt_L",
        "shift": "Shift_L",
        "tab": "Tab",
        "enter": "Return",
        "esc": "Escape",
        "backspace": "BackSpace",
        "delete": "Delete",
        "up": "Up",
        "down": "Down",
        "left": "Left",
        "right": "Right",
        "home": "Home",
        "end": "End",
        "pageup": "Page_Up",
        "pagedown": "Page_Down",
        "f1": "F1",
        "f2": "F2",
        "f3": "F3",
        "f4": "F4",
        "f5": "F5",
        "f6": "F6",
        "f7": "F7",
        "f8": "F8",
        "f9": "F9",
        "f10": "F10",
        "f11": "F11",
        "f12": "F12",
        "printscreen": "3270_PrintScreen",
        "prtscn": "3270_PrintScreen",
        "capslock": "Caps_Lock",
        "numlock": "Num_Lock",
        "scrolllock": "Scroll_Lock",
        "insert": "Insert",
        "pause": "Pause",
        "break": "Break",
        "~": "asciitilde",
        "!": "exclam",
        "@": "at",
        "#": "numbersign",
        "$": "dollar",
        "%": "percent",
        "^": "asciicircum",
        "&": "ampersand",
        "*": "asterisk",
        "(": "parenleft",
        ")": "parenright",
        "-": "minus",
        "_": "underscore",
        "=": "equal",
        "+": "plus",
        "[": "bracketleft",
        "]": "bracketright",
        "{": "braceleft",
        "}": "braceright",
        ";": "semicolon",
        ":": "colon",
        "'": "apostrophe",
        "\"": "quotedbl",
        "\\": "backslash",
        "|": "bar",
        ",": "comma",
        "<": "less",
        ".": "period",
        ">": "greater",
        "/": "slash",
        "?": "question",
        "`": "grave",
    }
    for char_code in range(ord('a'), ord('z') + 1):
        char = chr(char_code)
        if char not in convert_dict:
             convert_dict[char] = char
    for digit_code in range(ord('0'), ord('9') + 1):
        digit = chr(digit_code)
        if digit not in convert_dict:
             convert_dict[digit] = digit


    use_remote_clipboard = False
    remote_clipboard_host = "localhost"
    remote_clipboard_port = 8001
    remote_clipboard_secret_token = None

    @classmethod
    def set_remote_clipboard(cls, config):
        cls.use_remote_clipboard = config.get("use_remote_clipboard", False)
        cls.remote_clipboard_host = config.get("remote_clipboard_host","localhost")
        cls.remote_clipboard_port = config.get("remote_clipboard_port", 8001)
        cls.remote_clipboard_secret_token = config.get("remote_clipboard_secret_token", None)

    def __init__(self, keyboard_action_type: Union[KeyboardActionType, str] = None, keyboard_key: Union[str, List[str]] = None, keyboard_text: str = None, keyboard_input: str = None, **kwargs):
        if keyboard_text is None and keyboard_input is not None:
            keyboard_text = keyboard_input

        self.keyboard_action_type = keyboard_action_type
        if isinstance(keyboard_action_type, str):
            if keyboard_action_type.lower() == "input":
                self.keyboard_action_type = KeyboardActionType.text
            else:
                try:
                    self.keyboard_action_type = KeyboardActionType[keyboard_action_type.lower()]
                except KeyError:
                    raise IncompleteActionDataError(f"Invalid KeyboardActionType string: {keyboard_action_type}")

        if self.keyboard_action_type is None:
            if keyboard_key is not None:
                self.keyboard_action_type = KeyboardActionType.press
            elif keyboard_text is not None:
                self.keyboard_action_type = KeyboardActionType.text
            else:
                raise IncompleteActionDataError("KeyboardAction is incomplete: requires keyboard_action_type, keyboard_key, or keyboard_text")

        self.keyboard_key = keyboard_key
        self.keyboard_text = keyboard_text

        if self.keyboard_action_type in [KeyboardActionType.down, KeyboardActionType.up, KeyboardActionType.press]:
            if self.keyboard_key is None:
                 raise IncompleteActionDataError(f"KeyboardAction of type {self.keyboard_action_type.name} requires keyboard_key")

            if isinstance(self.keyboard_key, list):
                 processed_keys = []
                 for key_part in self.keyboard_key:
                      if not isinstance(key_part, str):
                           raise IncompleteActionDataError(f"Invalid key part type in list: {type(key_part)}")
                      processed_keys.append(self._convert_key(key_part.strip()))
                 self.keyboard_key = processed_keys
            elif isinstance(self.keyboard_key, str):
                 if "+" in self.keyboard_key:
                     keyboard_key_list = [k.strip() for k in self.keyboard_key.split("+") if k.strip()]
                     if not keyboard_key_list:
                          raise IncompleteActionDataError("KeyboardAction key string is empty after splitting '+'")
                     self.keyboard_key = [self._convert_key(key_part) for key_part in keyboard_key_list]
                 else:
                     self.keyboard_key = self._convert_key(self.keyboard_key.strip())
            else:
                 raise IncompleteActionDataError(f"Invalid keyboard_key type: {type(self.keyboard_key)}")

        elif self.keyboard_action_type == KeyboardActionType.text:
            if not isinstance(self.keyboard_text, str) or not self.keyboard_text:
                 raise IncompleteActionDataError(f"KeyboardAction of type {self.keyboard_action_type.name} requires non-empty keyboard_text")
            self.keyboard_text = self.keyboard_text

        if self.keyboard_action_type != KeyboardActionType.text:
            self.keyboard_text = None
        if self.keyboard_action_type == KeyboardActionType.text:
             self.keyboard_key = None


        super().__init__(**kwargs)

    def _convert_key(self, key_str: str) -> str:
         """Converts a string key representation to a valid keysym."""
         if not key_str:
              raise IncompleteActionDataError("Empty key string provided for conversion")

         key_lower = key_str.lower()

         if key_lower in self.convert_dict:
             return self.convert_dict[key_lower]

         if key_lower in vaild_keysymdef_lower_map:
             return vaild_keysymdef_lower_map[key_lower]

         if len(key_str) == 1:
              if key_str in vaild_keysymdef:
                   return key_str
              if key_lower in vaild_keysymdef_lower_map:
                   return vaild_keysymdef_lower_map[key_lower]


         raise IncompleteActionDataError(f"Invalid or unknown keyboard key string: '{key_str}'")


    async def step(self, vnc):
        if self.keyboard_action_type is None:
             raise IncompleteActionDataError("KeyboardAction has no keyboard_action_type")

        if self.keyboard_action_type in [KeyboardActionType.down, KeyboardActionType.up, KeyboardActionType.press]:
            if self.keyboard_key is None:
                raise IncompleteActionDataError(f"KeyboardAction of type {self.keyboard_action_type.name} requires keyboard_key for step")

            if isinstance(self.keyboard_key, list):
                if not all(isinstance(k, str) for k in self.keyboard_key):
                     raise IncompleteActionDataError("Keyboard key list contains non-string items")
                vnc.keyboard.press(*self.keyboard_key)
            elif isinstance(self.keyboard_key, str):
                vnc.keyboard.press(self.keyboard_key)
            else:
                 raise IncompleteActionDataError(f"Invalid processed keyboard_key type for step: {type(self.keyboard_key)}")

        elif self.keyboard_action_type == KeyboardActionType.text:
            if not isinstance(self.keyboard_text, str) or not self.keyboard_text:
                raise IncompleteActionDataError(f"KeyboardAction of type {self.keyboard_action_type.name} requires non-empty keyboard_text for step")

            if self.use_remote_clipboard:
                url = f"http://{self.remote_clipboard_host}:{self.remote_clipboard_port}/clipboard"
                data = {
                    "text": self.keyboard_text,
                    "token": self.remote_clipboard_secret_token
                }
                try:
                    r = requests.post(url, json=data)
                    print("remote clipboard server response:", r)
                    if r.status_code == 200:
                        await asyncio.sleep(0.1)
                        vnc.keyboard.press('Control_L', 'v')
                    else:
                        print("remote clipboard server error:", r.text)
                        vnc.keyboard.write(self.keyboard_text)
                except Exception as e:
                    print(f"remote clipboard server request failed: {e}")
                    vnc.keyboard.write(self.keyboard_text)
            else:
                vnc.keyboard.write(self.keyboard_text)
        else:
             raise IncompleteActionDataError(f"KeyboardAction step logic failed for type: {self.keyboard_action_type}")


    @property
    def keys_or_text(self):
        if self.keyboard_action_type in [KeyboardActionType.down, KeyboardActionType.up, KeyboardActionType.press]:
            if isinstance(self.keyboard_key, list):
                return "+".join(self.keyboard_key)
            elif isinstance(self.keyboard_key, str):
                return self.keyboard_key
            else:
                return None
        elif self.keyboard_action_type == KeyboardActionType.text:
            return self.keyboard_text
        else:
            return None

    def to_ideal_dict_format(self):
        dic = OrderedDict()
        dic["action_type"] = type(self).__name__
        if self.keyboard_action_type in [KeyboardActionType.down, KeyboardActionType.up, KeyboardActionType.press] and self.keyboard_key is not None:
             key_value = self.keyboard_key
             if isinstance(key_value, list):
                  key_value = "+".join(key_value)
             dic["keyboard_key"] = key_value
        elif self.keyboard_action_type == KeyboardActionType.text and self.keyboard_text is not None:
             dic["keyboard_text"] = self.keyboard_text

        if "keyboard_action_type" not in self.base_attributes:
             if self.keyboard_action_type is not None:
                 dic["keyboard_action_type"] = self.keyboard_action_type.name

        ordered_keys = ["action_type", "keyboard_action_type", "keyboard_key", "keyboard_text"]
        ordered_dic = OrderedDict((k, dic[k]) for k in ordered_keys if k in dic)

        return ordered_dic


    @property
    def score_point(self):
        score_point = 1
        if self.keyboard_action_type is not None:
            score_point += 1

        if self.keyboard_action_type in [KeyboardActionType.down, KeyboardActionType.up, KeyboardActionType.press] and self.keyboard_key is not None:
            score_point += 1
        elif self.keyboard_action_type == KeyboardActionType.text and self.keyboard_text is not None:
            score_point += 1

        return score_point

    def similarity(self, pred_action: Action):
        scores = [
            ActionAttributeScore("action_type", pred_action.action_type if pred_action else None, self.action_type),
        ]

        if not isinstance(pred_action, KeyboardAction):
             if self.keyboard_action_type is not None:
                  scores.append(ActionAttributeScore("keyboard_action_type", None, self.keyboard_action_type.name))
             if self.keys_or_text is not None:
                  scores.append(ActionValueScore("keyboard_key_or_text", 0.0, "same_or_bleu", None, self.keys_or_text))
             return ActionSimilarity(self.score_point, scores)

        scores.append(ActionAttributeScore("keyboard_action_type", pred_action.keyboard_action_type.name if pred_action.keyboard_action_type else None, self.keyboard_action_type.name if self.keyboard_action_type else None))

        label_key_or_text = self.keys_or_text
        pred_key_or_text = pred_action.keys_or_text

        if label_key_or_text is not None:
             if pred_key_or_text is not None:
                 if self.keyboard_action_type in [KeyboardActionType.down, KeyboardActionType.up, KeyboardActionType.press]:
                      is_same = (pred_key_or_text.lower() == label_key_or_text.lower())
                      scores.append(ActionValueScore("keyboard_key", 1.0 if is_same else 0.0, "exact_match", pred_key_or_text, label_key_or_text))
                 elif self.keyboard_action_type == KeyboardActionType.text:
                      if pred_key_or_text == label_key_or_text:
                           scores.append(ActionValueScore("keyboard_text", 1.0, "bleu", pred_key_or_text, label_key_or_text))
                      else:
                           pred_tokens = pred_key_or_text.split()
                           label_tokens = label_key_or_text.split()
                           if not label_tokens:
                                score = 1.0 if not pred_tokens else 0.0
                                scores.append(ActionValueScore("keyboard_text", score, "bleu", pred_key_or_text, label_key_or_text))
                           else:
                                try:
                                    score_tuple = compute_bleu(translation_corpus=[pred_tokens], reference_corpus=[[label_tokens]])
                                    bleu_score = score_tuple[0]
                                    scores.append(ActionValueScore("keyboard_text", bleu_score, "bleu", pred_key_or_text, label_key_or_text))
                                except Exception as e:
                                    print(f"Warning: Failed to compute BLEU for keyboard text: {e}")
                                    is_same = (pred_key_or_text == label_key_or_text)
                                    scores.append(ActionValueScore("keyboard_text", 1.0 if is_same else 0.0, "exact_match_fallback", pred_key_or_text, label_key_or_text))
                 else:
                      pass
             else:
                  scores.append(ActionValueScore("keyboard_key_or_text", 0.0, "missing_pred", None, label_key_or_text))


        return ActionSimilarity(self.score_point, scores)


class WaitAction(Action):
    save_attributes = ["wait_time"]

    def __init__(self, wait_time: float = 0.5, **kwargs):
        self.wait_time = float(wait_time)
        if self.wait_time < 0:
             self.wait_time = 0.0
        super().__init__(**kwargs)

    async def step(self, vnc):
        if self.wait_time > 0:
             await asyncio.sleep(self.wait_time)
        else:
             pass


    @property
    def score_point(self):
        return 0

    def similarity(self, pred_action: Action):
        scores = [
            ActionAttributeScore("action_type", pred_action.action_type if pred_action else None, self.action_type),
        ]

        return ActionSimilarity(self.score_point, scores)

class PlanAction(Action):
    save_attributes = ["element"]

    def __init__(self, element: str = None, **kwargs):
        if not isinstance(element, str) or not element.strip():
             raise IncompleteActionDataError("PlanAction requires a non-empty 'element' string")
        self.element = element.strip()
        super().__init__(**kwargs)

    @property
    def score_point(self):
        score_point = 1
        if self.element:
             score_point += 1
        return score_point

    def similarity(self, pred_action: Action):

        scores = [
            ActionAttributeScore("action_type", pred_action.action_type if pred_action else None, self.action_type),
        ]

        if not isinstance(pred_action, PlanAction):
             if self.element:
                  scores.append(ActionValueScore("element", 0.0, "bleu", None, self.element))
             return ActionSimilarity(self.score_point, scores)

        if self.element is not None:
             pred_element = pred_action.element
             label_element = self.element

             if pred_element is not None and pred_element.strip():
                 if pred_element.strip() == label_element.strip():
                     scores.append(ActionValueScore("element", 1.0, "bleu", pred_element, label_element))
                 else:
                     pred_tokens = pred_element.strip().split()
                     label_tokens = label_element.strip().split()

                     if not label_tokens:
                         score = 1.0 if not pred_tokens else 0.0
                         scores.append(ActionValueScore("element", score, "bleu", pred_element, label_element))
                     else:
                         try:
                              score_tuple = compute_bleu(translation_corpus = [pred_tokens], reference_corpus=[[label_tokens]])
                              bleu_score = score_tuple[0]
                              scores.append(ActionValueScore("element", bleu_score, "bleu", pred_element, label_element))
                         except Exception as e:
                             print(f"Warning: Failed to compute BLEU for plan element: {e}")
                             is_same = (pred_element.strip() == label_element.strip())
                             scores.append(ActionValueScore("element", 1.0 if is_same else 0.0, "exact_match_fallback", pred_element, label_element))
             else:
                 scores.append(ActionValueScore("element", 0.0, "missing_pred", None, label_element))

        return ActionSimilarity(self.score_point, scores)

class EvaluateSubTaskAction(Action):
    save_attributes = ["situation", "advice"]

    def __init__(self, situation=None,  advice=None, **kwargs):
        super().__init__(**kwargs)
        if situation == "goal_success":
            self.situation = "sub_task_success"
        else:
            self.situation = situation
        self.advice = advice


    @classmethod
    def check(cls, action: Action):
        if isinstance(action, EvaluateSubTaskAction):
            if action.situation == "sub_task_success":
                return True
            elif action.situation in ["need_retry", "need_reformulate"]:
                if action.advice is not None and action.advice.strip():
                     return True
                else:
                     print(f"Validation Failed: EvaluateSubTaskAction '{action.situation}' requires non-empty advice.")
                     return False
            else:
                 print(f"Validation Failed: EvaluateSubTaskAction has unknown situation '{action.situation}'.")
                 return False
        return False

    @property
    def score_point(self):
        score_point = 1
        if self.situation in ["sub_task_success", "need_retry", "need_reformulate"]:
            score_point += 1
        return score_point

    def similarity(self, pred_action: Action):
        scores = [
            ActionAttributeScore("action_type", pred_action.action_type if pred_action else None, self.action_type),
        ]

        if not isinstance(pred_action, EvaluateSubTaskAction):
             if self.situation in ["sub_task_success", "need_retry", "need_reformulate"]:
                  label_score_situation = "sub_task_success" if self.situation == "sub_task_success" else "sub_task_fail"
                  scores.append(ActionAttributeScore("situation", None, label_score_situation))
             return ActionSimilarity(self.score_point, scores)


        pred_situation = pred_action.situation
        pred_score_situation = "sub_task_success" if pred_situation == "sub_task_success" or pred_situation == "goal_success" else "sub_task_fail"
        label_score_situation = "sub_task_success" if self.situation == "sub_task_success" else "sub_task_fail"

        if self.situation in ["sub_task_success", "need_retry", "need_reformulate"]:
             scores.append(ActionAttributeScore("situation",  pred_score_situation, label_score_situation))

        return ActionSimilarity(self.score_point, scores)

# --- Existing JSON Encoding and Parsing (Keep the ObjEncoder) ---
class ObjEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Position) or isinstance(obj, ClickableArea):
            return asdict(obj)
        elif issubclass(obj.__class__, Action):
            return obj.save_action()
        elif isinstance(obj, np.ndarray):
            # Ensure the data type is compatible with tobytes(). E.g., obj.astype(np.uint8).tobytes()
            # Assuming obs are uint8 images
            return base64.b64encode(zlib.compress(obj.tobytes())).decode('utf-8')
        elif isinstance(obj, enum.Enum):
            return obj.value
        elif isinstance(obj, ActionAttributeScore) or isinstance(obj, ActionValueScore) or isinstance(obj, ActionSimilarity):
            return asdict(obj)
        elif isinstance(obj, PredictionParsed):
             return asdict(obj)
        return json.JSONEncoder.default(self, obj)

    def iterencode(self, obj, **kwargs):
        # if isinstance(obj, dict):
        #     obj = OrderedDict(obj)
        return super().iterencode(obj, **kwargs)


# --- Old JSON Parsing Functions (Can be kept for backward compatibility or removed) ---

def _try_find_json(last_stream_response: str):
    try:
        last_stream_response = last_stream_response.strip().replace("\\_", '_')
        last_stream_response = re.sub(r',\s*}', '}', last_stream_response)
        last_stream_response = re.sub(r',\s*]', ']', last_stream_response)
        action_json = json.loads(last_stream_response)
        return action_json
    except json.JSONDecodeError as e:
        return None

def _parse_json_to_action(maybe_action_json: Union[Dict, List]):
    actions = []
    if isinstance(maybe_action_json, list):
        for action_dict in maybe_action_json:
            if isinstance(action_dict, dict):
                action = Action.from_json(action_dict)
                if action is not None:
                    actions.append(action)
    elif isinstance(maybe_action_json, dict):
        action = Action.from_json(maybe_action_json)
        if action is not None:
            actions.append(action)
    return actions

def parse_action_from_text(last_stream_response: str):
    actions = []
    one_maybe_json = _try_find_json(last_stream_response)
    if one_maybe_json is not None:
        actions.extend(_parse_json_to_action(one_maybe_json))
    else:
        pattern = r'```json(.*?)```'
        res = re.findall(pattern, last_stream_response, re.DOTALL)
        if len(res) > 0:
            for json_block_content in res:
                json_block_content = json_block_content.strip()
                if not json_block_content:
                    continue

                maybe_action = _try_find_json(json_block_content)
                if maybe_action is not None:
                    actions.extend(_parse_json_to_action(maybe_action))
                else:
                    for one_line in json_block_content.split("\n"):
                        maybe_action = _try_find_json(one_line.strip())
                        if maybe_action is not None:
                            actions.extend(_parse_json_to_action(maybe_action))

    actions = [action for action in actions if action is not None]
    return actions

def find_non_json_span_from_text(last_stream_response: str):
    non_json_span = []
    if _try_find_json(last_stream_response) is not None:
        return non_json_span

    json_block_spans = []
    pattern = r'```json.*?```'
    for match in re.finditer(pattern, last_stream_response, re.DOTALL):
        start, end = match.span()
        json_block_spans.append((start, end))

    last_end = 0
    for start, end in json_block_spans:
        if start > last_end:
            non_json_span.append((last_end, start))
        last_end = end

    if last_end < len(last_stream_response):
        non_json_span.append((last_end, len(last_stream_response)))

    return non_json_span

def split_json_span_and_non_json_span_from_text(last_stream_response: str):
    json_spans = []
    non_json_spans = []
    if _try_find_json(last_stream_response) is not None:
        json_spans.append((0, len(last_stream_response)))
        return json_spans, non_json_spans

    pattern = r'```json.*?```'
    for match in re.finditer(pattern, last_stream_response, re.DOTALL):
        start, end = match.span()
        json_spans.append((start, end))

    last_end = 0
    for start, end in json_spans:
        if start > last_end:
            non_json_spans.append((last_end, start))
        last_end = end

    if last_end < len(last_stream_response):
        non_json_spans.append((last_end, len(last_stream_response)))

    return json_spans, non_json_spans


# --- New Functions for Parsing UI-tars2b / VLM Output ---

# Constants from TS code
MAX_RATIO = 2.5
IMAGE_FACTOR = 32
MIN_PIXELS = 320 * 320
MAX_PIXELS_V1_5 = 800 * 800

class UITarsModelVersion(enum.Enum):
    V1_0 = "V1_0"
    V1_5 = "V1_5"


def round_by_factor(num: float, factor: int) -> int:
    """Rounds a number to the nearest multiple of the factor."""
    return max(factor, round(num / factor) * factor)

def floor_by_factor(num: float, factor: int) -> int:
    """Floors a number to the nearest multiple of the factor."""
    return max(factor, math.floor(num / factor) * factor)

def ceil_by_factor(num: float, factor: int) -> int:
    """Ceilings a number to the nearest multiple of the factor."""
    return max(factor, math.ceil(num / factor) * factor)

def smart_resize_for_v15(
    height: int,
    width: int,
    max_ratio: float = MAX_RATIO,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS_V1_5,
) -> Union[tuple[int, int], None]:
    """
    Calculates smart resize dimensions for V1.5 model.
    Returns (resized_width, resized_height) or None if aspect ratio is invalid.
    """
    if min(height, width) <= 0:
        return None

    if max(height, width) / min(height, width) > max_ratio:
        # print(f"Error: Aspect ratio {max(height, width) / min(height, width)} exceeds max {max_ratio}")
        return None

    w_bar = round_by_factor(width, factor)
    h_bar = round_by_factor(height, factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
        while h_bar * w_bar > max_pixels and (h_bar > factor or w_bar > factor):
             if h_bar > w_bar:
                 h_bar = floor_by_factor(h_bar * 0.95, factor)
             else:
                 w_bar = floor_by_factor(w_bar * 0.95, factor)
             h_bar = max(factor, h_bar)
             w_bar = max(factor, w_bar)

    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
        while h_bar * w_bar < min_pixels:
             if h_bar < w_bar:
                 h_bar = ceil_by_factor(h_bar * 1.05, factor)
             else:
                 w_bar = ceil_by_factor(w_bar * 1.05, factor)


    return (w_bar, h_bar)


def _parse_single_action_string(action_str: str) -> Union[Dict[str, Any], None]:
    """
    Parses a single action function call string (e.g., "click(start_box='(279,81)')").
    Returns a dict { 'function': 'click', 'args': {...} } or None on failure.
    Mimics the logic of the TS `parseAction` function.
    """
    if not isinstance(action_str, str):
         return None

    try:
        action_str = action_str.replace('<|box_start|>', '').replace('<|box_end|>', '')
        action_str = action_str.replace('point=', 'start_box=')

        function_match = re.match(r'^(\w+)\((.*)\)$', action_str.strip())

        if not function_match:
            return None

        function_name = function_match.group(1)
        args_str = function_match.group(2).strip()

        kwargs = {}

        if args_str:
            # Regex to find key=value pairs, handling quoted strings, parentheses tuples, or simple words
            arg_pattern_refined = re.compile(r'\s*(\w+)\s*=\s*('
                                             r"'[^']*'|"
                                             r'"[^"]*"|'
                                             r'\([^)]*\)|'
                                             r'[^,]+'
                                             r')')

            arg_matches = list(arg_pattern_refined.finditer(args_str))

            last_end = 0
            for match in arg_matches:
                 if match.start() > last_end:
                      unmatched_text = args_str[last_end:match.start()]
                      if not re.fullmatch(r'[\s,]*', unmatched_text):
                           pass
                 last_end = match.end()

            if last_end < len(args_str):
                 unmatched_text = args_str[last_end:].strip()
                 if unmatched_text and unmatched_text != ',':
                      pass

            for match in arg_matches:
                key = match.group(1)
                value = match.group(2).strip()

                trimmed_value = value

                if (trimmed_value.startswith("'") and trimmed_value.endswith("'")) or \
                   (trimmed_value.startswith('"') and trimmed_value.endswith('"')):
                    trimmed_value = trimmed_value[1:-1]

                # Support format: click(start_box='<bbox>637 964 637 964</bbox>')
                if '<bbox>' in trimmed_value:
                    trimmed_value = trimmed_value.replace('<bbox>', '').replace('</bbox>', '')
                    # CORRECTED LINE: Use re.sub for Python regex
                    trimmed_value = re.sub(r'\s+', ',', trimmed_value)
                    trimmed_value = f'({trimmed_value})'

                # Support format: click(point='<point>510 150</point>')
                if '<point>' in trimmed_value:
                    trimmed_value = trimmed_value.replace('<point>', '').replace('</point>', '')
                    # CORRECTED LINE: Use re.sub for Python regex
                    trimmed_value = re.sub(r'\s+', ',', trimmed_value)
                    trimmed_value = f'({trimmed_value})'

                kwargs[key.strip()] = trimmed_value


        return {
            'function': function_name,
            'args': kwargs,
        }

    except Exception as e:
        # print(f"Failed to parse action string '{action_str}': {e}")
        return None


def parse_action_vlm(
    text: str,
    factors: Union[int, tuple[int, int]] = (1000, 1000),
    mode: str = 'bc',
    screen_context: Union[Dict[str, int], None] = None,
    scale_factor: float = 1.0,
    model_ver: UITarsModelVersion = UITarsModelVersion.V1_0,
) -> List[PredictionParsed]:
    """
    Parses raw text output from a VLM (like UI-tars2b) into structured actions.
    Mimics the logic of the TS `parseActionVlm` function.
    """
    reflection: Union[str, None] = None
    thought: Union[str, None] = None
    action_str = ''

    if isinstance(factors, int):
        width_factor, height_factor = factors, factors
    elif isinstance(factors, tuple) and len(factors) == 2:
        width_factor, height_factor = factors
    else:
        print(f"Warning: Invalid factors format: {factors}. Using (1000, 1000).")
        width_factor, height_factor = 1000, 1000

    smart_resize_factors: Union[tuple[int, int], None] = None
    if model_ver == UITarsModelVersion.V1_5 and screen_context and screen_context.get('height') is not None and screen_context.get('width') is not None:
        smart_resize_factors = smart_resize_for_v15(
            screen_context['height'],
            screen_context['width']
        )
        if smart_resize_factors is None:
            print("Warning: smart_resize_for_v15 returned None. Using default factors.")
            smart_resize_factors = (width_factor, height_factor)


    text = text.strip()

    if mode == 'bc':
        action_parts = text.split('Action:')
        if len(action_parts) > 1:
            pre_action_text = 'Action:'.join(action_parts[:-1]).strip()
            action_str = action_parts[-1].strip()

            if pre_action_text.startswith('Thought:'):
                 thought_match = re.match(r'Thought: ([\s\S]+)', pre_action_text)
                 if thought_match:
                      thought = thought_match.group(1).strip()
            elif pre_action_text.startswith('Reflection:'):
                reflection_summary_match = re.match(r'Reflection: ([\s\S]+?)Action_Summary: ([\s\S]+)', pre_action_text)
                if reflection_summary_match:
                    reflection = reflection_summary_match.group(1).strip()
                    thought = reflection_summary_match.group(2).strip()
                else:
                    reflection = pre_action_text.replace('Reflection:', '', 1).strip()
                    thought = reflection

            elif pre_action_text.startswith('Action_Summary:'):
                summary_match = re.match(r'Action_Summary: ([\s\S]+)', pre_action_text)
                if summary_match:
                    thought = summary_match.group(1).strip()
            elif pre_action_text:
                thought = pre_action_text


        else:
            action_str = text
            thought = None
            reflection = None

    elif mode == 'o1':
        thought_match = re.search(r'<Thought>\s*(.*?)\s*<\/Thought>', text, re.DOTALL)
        action_summary_match = re.search(r'Action_Summary:\s*(.*?)\s*Action:', text, re.DOTALL)
        action_match = re.search(r'Action:\s*(.*?)\s*<\/Output>', text, re.DOTALL)
        output_match = re.search(r'<Output>([\s\S]+?)<\/Output>', text, re.DOTALL)

        thought_content = thought_match.group(1).strip() if thought_match else None
        action_summary_content = action_summary_match.group(1).strip() if action_summary_match else None
        action_content = action_match.group(1).strip() if action_match else None

        thought_parts = []
        if thought_content:
             thought_parts.append(thought_content)
        if action_summary_content:
             thought_parts.append(f"<Action_Summary>\n{action_summary_content}")

        thought = "\n".join(thought_parts) if thought_parts else None

        action_str = action_content if action_content is not None else ''

        if not action_str and output_match:
            output_content = output_match.group(1).strip()
            if thought_match:
                 output_content = output_content.replace(thought_match.group(0), '').strip()
            action_str = output_content


    else:
        print(f"Warning: Unknown parsing mode '{mode}'. Treating entire text as action string.")
        action_str = text
        thought = None
        reflection = None


    all_raw_actions = [a.strip() for a in action_str.split('\n\n') if a.strip()]

    parsed_actions: List[PredictionParsed] = []

    for raw_str in all_raw_actions:
        action_instance_data = _parse_single_action_string(raw_str)

        if action_instance_data is None:
             continue

        action_type = action_instance_data.get('function', '')
        raw_inputs = action_instance_data.get('args', {})
        action_inputs: Dict[str, Any] = {}

        for param_name, param_value_str in raw_inputs.items():
            if not param_value_str:
                 action_inputs[param_name.strip()] = ""
                 continue

            if 'box' in param_name.lower() or 'point' in param_name.lower():
                ori_box_str = param_value_str

                numbers_str = re.findall(r'\d+(\.\d+)?', ori_box_str.replace('(', '').replace(')', '').replace('[', '').replace(']', ''))
                try:
                    float_numbers = [float(num_str[0] + (num_str[1] if num_str[1] else '')) for num_str in numbers_str]
                except ValueError:
                    action_inputs[param_name.strip()] = ori_box_str
                    continue

                normalized_coords = []
                for i, num in enumerate(float_numbers):
                    factor_index = i % 2
                    current_factor = smart_resize_factors[factor_index] if smart_resize_factors else (width_factor if factor_index == 0 else height_factor)
                    if current_factor == 0:
                         normalized_coords.append(0.0)
                    else:
                         normalized_coords.append(num / current_factor)


                if len(normalized_coords) == 2:
                    normalized_coords.extend(normalized_coords)
                elif len(normalized_coords) > 4:
                    normalized_coords = normalized_coords[:4]
                elif len(normalized_coords) < 4:
                     action_inputs[param_name.strip()] = ori_box_str
                     continue

                action_inputs[param_name.strip()] = ori_box_str

                if screen_context and screen_context.get('width') is not None and screen_context.get('height') is not None:
                    screen_width = screen_context['width']
                    screen_height = screen_context['height']

                    x1, y1, x2, y2 = normalized_coords

                    center_x_normalized = (x1 + x2) / 2.0
                    center_y_normalized = (y1 + y2) / 2.0

                    final_screen_x = round(center_x_normalized * screen_width * scale_factor)
                    final_screen_y = round(center_y_normalized * screen_height * scale_factor)

                    coord_key = 'start_coords' if 'start_box' in param_name.lower() or 'point' in param_name.lower() else 'end_coords'
                    action_inputs[coord_key] = [final_screen_x, final_screen_y]

            else:
                action_inputs[param_name.strip()] = param_value_str


        parsed_actions.append(PredictionParsed(
            reflection=reflection,
            thought=thought if thought is not None else '',
            action_type=action_type,
            action_inputs=action_inputs,
        ))

    return parsed_actions


# --- Main Entry Point for VLM Parsing ---

def parse_ui_tars_prediction(
    prediction: str,
    factor: Union[int, tuple[int, int]],
    screen_context: Union[Dict[str, int], None] = None,
    scale_factor: float = 1.0,
    mode: str = 'bc',
    model_ver: str = 'V1_0'
) -> List[PredictionParsed]:
    """
    Public entry point to parse a raw prediction string from UI-tars-like VLMs.
    """
    try:
        model_enum = UITarsModelVersion[model_ver.upper()]
    except KeyError:
        print(f"Warning: Unknown model version '{model_ver}'. Defaulting to V1_0.")
        model_enum = UITarsModelVersion.V1_0

    return parse_action_vlm(
        text=prediction,
        factors=factor,
        mode=mode,
        screen_context=screen_context,
        scale_factor=scale_factor,
        model_ver=model_enum,
    )


# --- Existing Action Sequence Comparison (Keep as is) ---

# Using dynamic programming for optimal path calculation instead of brute force recursion over all alignments
def calculate_optimal_path_dp(score_matrix: np.ndarray):
    """
    Calculates the optimal alignment path using dynamic programming.
    """
    n, m = score_matrix.shape

    dp = np.zeros((n + 1, m + 1))
    bp = [[None for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = dp[i-1][j-1] + score_matrix[i-1][j-1]
            skip_label_score = dp[i-1][j]
            skip_pred_score = dp[i][j-1]

            if match_score >= skip_label_score and match_score >= skip_pred_score:
                dp[i][j] = match_score
                bp[i][j] = 'diag'
            elif skip_label_score >= skip_pred_score:
                dp[i][j] = skip_label_score
                bp[i][j] = 'up'
            else:
                dp[i][j] = skip_pred_score
                bp[i][j] = 'left'

    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bp[i][j] == 'diag':
            alignment.append((i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or bp[i][j] == 'up'):
            i -= 1
        elif j > 0 and (i == 0 or bp[i][j] == 'left'):
            j -= 1

    alignment.reverse()

    best_score = dp[n][m]

    matched_alignment = [(lbl_idx, pred_idx) for lbl_idx, pred_idx in alignment if lbl_idx is not None and pred_idx is not None]


    return matched_alignment, best_score


def compare_action_sequences(pred_seq: List[Action], label_seq: List[Action]):
    # Use DP search to find the best alignment of actions

    if not label_seq:
         print("Warning: Label sequence is empty.")
         return float('nan'), [], []

    score_matrix = np.zeros((len(label_seq), len(pred_seq)))
    info_matrix = [[None for _ in range(len(pred_seq))] for _ in range(len(label_seq))]

    for i in range(len(label_seq)):
        for j in range(len(pred_seq)):
            score_info = label_seq[i].similarity(pred_seq[j])
            info_matrix[i][j] = score_info
            score_matrix[i, j] = score_info.get_score()


    best_alignment, best_alignment_score = calculate_optimal_path_dp(score_matrix)

    all_score_point = sum(item.score_point for item in label_seq)

    best_alignment_similarity_info = []
    for i, j in best_alignment:
        if 0 <= i < len(label_seq) and 0 <= j < len(pred_seq):
             best_alignment_similarity_info.append(info_matrix[i][j])


    if all_score_point == 0:
        print("Warning: Total score points from label sequence is 0. Cannot calculate normalized score.")
        return float('nan'), best_alignment_similarity_info, best_alignment

    normalized_score = best_alignment_score / all_score_point

    return normalized_score, best_alignment_similarity_info, best_alignment