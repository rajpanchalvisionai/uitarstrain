import uuid
import queue
import asyncio
from asyncio import open_connection
from functools import partial
import time

import numpy as np
from PIL import Image

from datetime import datetime

# --- Import necessary PyQt5 modules ---
from PyQt5.QtCore import Qt, QTimer, QMetaObject, Q_ARG, pyqtSlot, QObject
from PyQt5.QtWidgets import QWidget, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QTextEdit, QCheckBox, QApplication, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QCursor, QColor, QPalette

from qasync import asyncSlot, asyncClose

from asyncvnc import Client

# --- Import necessary types ---
from typing import Union, Dict, List, Any, Optional

# Assuming base.py contains Position, ClickableArea, VNCMouseButton, convert_mouse_button_qt, convert_qt2keysymdef_key_mapping
# and potential custom exceptions like IncompleteActionDataError
from base import *

# Assuming action.py contains Action hierarchy, PredictionParsed, parse_ui_tars_prediction,
# MouseAction, KeyboardAction, WaitAction, and potentially exceptions
from action import *

# --- Import Conversation from the corrected path ---
from interface_api.conversation import Conversation, conv_templates, UITarsPromptVersion, UITarsLanguage, UITarsUseCase


# --- Mouse Button Conversion (assuming not in base.py) ---
# If convert_mouse_button_qt is not in base.py, add it here
# from PyQt5.QtCore import Qt
# import enum
# class VNCMouseButton(enum.Enum): # Example enum, should match base.py
#     left = 1
#     middle = 2
#     right = 4
# def convert_mouse_button_qt(qt_button):
#     if qt_button == Qt.LeftButton:
#         return VNCMouseButton.left
#     elif qt_button == Qt.RightButton:
#         return VNCMouseButton.right
#     elif qt_button == Qt.MiddleButton:
#         return VNCMouseButton.middle
#     else:
#         return None # Unknown button


# --- Keyboard Conversion (assuming not in base.py) ---
# If convert_qt2keysymdef_key_mapping is not in base.py, add it here
# from PyQt5.QtCore import Qt
# from keysymdef import keysymdef # assuming keysymdef is importable
# import re
# def convert_qt2keysymdef_key_mapping(qt_key_code):
#     # Basic mapping, needs comprehensive coverage of Qt key codes
#     # This is complex and would ideally be in a dedicated file or utility
#     # Mapping some common keys as examples
#     if qt_key_code == Qt.Key_Return: return 'Return'
#     if qt_key_code == Qt.Key_Enter: return 'Return' # often same as Return
#     if qt_key_code == Qt.Key_Escape: return 'Escape'
#     if qt_key_code == Qt.Key_Space: return 'space'
#     if qt_key_code == Qt.Key_Left: return 'Left'
#     if qt_key_code == Qt.Key_Right: return 'Right'
#     if qt_key_code == Qt.Key_Up: return 'Up'
#     if qt_key_code == Qt.Key_Down: return 'Down'
#     if qt_key_code == Qt.Key_Home: return 'Home'
#     if qt_key_code == Qt.Key_End: return 'End'
#     if qt_key_code == Qt.Key_PageUp: return 'Page_Up'
#     if qt_key_code == Qt.Key_PageDown: return 'Page_Down'
#     if qt_key_code == Qt.Key_Backspace: return 'BackSpace'
#     if qt_key_code == Qt.Key_Delete: return 'Delete'
#     if qt_key_code == Qt.Key_Tab: return 'Tab'
#     if qt_key_code == Qt.Key_Control: return 'Control_L' # Assuming left control
#     if qt_key_code == Qt.Key_Shift: return 'Shift_L' # Assuming left shift
#     if qt_key_code == Qt.Key_Alt: return 'Alt_L' # Assuming left alt
#     if qt_key_code == Qt.Key_Meta: return 'Super_L' # Assuming left Super (Windows key)
#     # Add more mappings based on keysymdef and Qt key codes

#     # Handle alphanumeric and symbol keys (basic attempt)
#     text = event.text() # event would be passed to keyPressEvent, need to adapt this function signature
#     if text and len(text) == 1:
#          # Convert character to keysym name if possible
#          # This requires a mapping or lookup, possibly using keysymdef
#          # For simplicity, return the character itself if it's printable ASCII
#          if ' ' <= text <= '~':
#              # Basic conversion for single chars, may not handle Shift correctly without state
#              # Rely on KeyboardAction's _convert_key if possible
#              return text
#          # More complex lookup needed for non-ASCII or special single keys

#     print(f"Warning: No keysym mapping for Qt key code: {qt_key_code}")
#     return None


# --- VNC Frame Widget Definition (Moved Before VNCWidget) ---
class VNCFrame(QLabel):
    def __init__(self, parent, action_queue):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.action_queue = action_queue

        self.is_in_focus = False
        self.out_focus_style()

    def in_focus_style(self):
        palette = self.palette()
        palette.setColor(QPalette.WindowText, QColor(Qt.green))
        self.setPalette(palette)
        self.setStyleSheet("border: 5px solid green;")

    def out_focus_style(self):
        palette = self.palette()
        palette.setColor(QPalette.WindowText, QColor(Qt.black)) # Or red as in original
        self.setPalette(palette)
        self.setStyleSheet("border: 5px solid red;")

    # --- Manual User Input Handlers ---
    def mousePressEvent(self, event):
        if self.is_in_focus:
            cursor_pos_qt = self.mapFromGlobal(QCursor.pos())
            if 0 <= cursor_pos_qt.x() < self.width() and 0 <= cursor_pos_qt.y() < self.height():
                 cursor_pos = Position(cursor_pos_qt.x(), cursor_pos_qt.y())
                 mouse_button_vnc = convert_mouse_button_qt(event.button())
                 if mouse_button_vnc is not None:
                     action = MouseAction(mouse_action_type=MouseActionType.down, mouse_button=mouse_button_vnc, mouse_position=cursor_pos)
                     self.action_queue.put(action)

    def mouseMoveEvent(self, event):
        cursor_pos_qt = self.mapFromGlobal(QCursor.pos())
        is_over_frame = (0 <= cursor_pos_qt.x() < self.width() and 0 <= cursor_pos_qt.y() < self.height())

        if is_over_frame and self.is_in_focus is False:
            self.is_in_focus = True
            self.in_focus_style()
        elif not is_over_frame and self.is_in_focus is True:
            self.is_in_focus = False
            self.out_focus_style()

        if self.is_in_focus and is_over_frame:
            cursor_pos = Position(cursor_pos_qt.x(), cursor_pos_qt.y())
            action = MouseAction(mouse_action_type=MouseActionType.move, mouse_position=cursor_pos)
            self.action_queue.put(action)


    def mouseReleaseEvent(self, event):
        if self.is_in_focus:
            cursor_pos_qt = self.mapFromGlobal(QCursor.pos())
            if 0 <= cursor_pos_qt.x() < self.width() and 0 <= cursor_pos_qt.y() < self.height():
                cursor_pos = Position(cursor_pos_qt.x(), cursor_pos_qt.y())
                mouse_button_vnc = convert_mouse_button_qt(event.button())
                if mouse_button_vnc is not None:
                    action = MouseAction(mouse_action_type=MouseActionType.up, mouse_button=mouse_button_vnc, mouse_position=cursor_pos)
                    self.action_queue.put(action)

    def wheelEvent(self, event):
        if self.is_in_focus:
            scroll_repeat = int(event.angleDelta().y() / 120)
            if scroll_repeat != 0:
                mouse_action_type = MouseActionType.scroll_up if scroll_repeat > 0 else MouseActionType.scroll_down
                action = MouseAction(
                    mouse_action_type=mouse_action_type, scroll_repeat=abs(scroll_repeat))
                self.action_queue.put(action)


    def keyPressEvent(self, event):
        if self.is_in_focus:
            keyboard_key = convert_qt2keysymdef_key_mapping(event.key())
            if keyboard_key is not None:
                 action = KeyboardAction(
                    keyboard_action_type=KeyboardActionType.down, keyboard_key=keyboard_key)
                 self.action_queue.put(action)


    def keyReleaseEvent(self, event):
        if self.is_in_focus:
            keyboard_key = convert_qt2keysymdef_key_mapping(event.key())
            if keyboard_key is not None:
                 action = KeyboardAction(
                    keyboard_action_type=KeyboardActionType.up, keyboard_key=keyboard_key)
                 self.action_queue.put(action)

    def update_screen(self, qimage):
        self.setPixmap(QPixmap.fromImage(qimage))


# --- Helper function to convert PredictionParsed to executable Actions ---
def convert_prediction_to_actions(parsed_prediction: PredictionParsed, screen_context: Dict[str, int]) -> List[Action]:
    """
    Converts a single PredictionParsed object into a list of executable Action objects.
    Handles mapping predicted action names and parameters to Python Action classes.
    Returns a list because a single PredictionParsed can potentially contain multiple actions (though typically one).
    """
    executable_actions: List[Action] = []
    action_type_str = parsed_prediction.action_type.lower()
    action_inputs = parsed_prediction.action_inputs

    try:
        if action_type_str == "click":
            mouse_button = VNCMouseButton.left # Default to left click
            if 'mouse_button' in action_inputs and action_inputs['mouse_button']:
                try:
                    mouse_button = VNCMouseButton[action_inputs['mouse_button'].lower()]
                except KeyError:
                    print(f"Warning: Invalid mouse_button '{action_inputs['mouse_button']}' for click. Using 'left'.")
                    mouse_button = VNCMouseButton.left

            mouse_position = None
            # Use 'start_coords' calculated by the parser
            if 'start_coords' in action_inputs and isinstance(action_inputs['start_coords'], list) and len(action_inputs['start_coords']) == 2:
                 mouse_position = Position(int(action_inputs['start_coords'][0]), int(action_inputs['start_coords'][1]))


            if mouse_position is not None:
                 executable_actions.append(MouseAction(
                    mouse_action_type=MouseActionType.click,
                    mouse_button=mouse_button,
                    mouse_position=mouse_position
                 ))
            else:
                 print(f"Warning: Skipping click action due to missing/invalid mouse_position derived from input: {action_inputs.get('start_box') or action_inputs.get('point')}. Parsed coords: {action_inputs.get('start_coords')}")


        elif action_type_str == "left_double":
            mouse_button = VNCMouseButton.left
            mouse_position = None
            if 'start_coords' in action_inputs and isinstance(action_inputs['start_coords'], list) and len(action_inputs['start_coords']) == 2:
                 mouse_position = Position(int(action_inputs['start_coords'][0]), int(action_inputs['start_coords'][1]))

            if mouse_position is not None:
                 executable_actions.append(MouseAction(
                    mouse_action_type=MouseActionType.double_click,
                    mouse_button=mouse_button,
                    mouse_position=mouse_position
                 ))
            else:
                 print(f"Warning: Skipping left_double action due to missing/invalid mouse_position derived from input: {action_inputs.get('start_box') or action_inputs.get('point')}. Parsed coords: {action_inputs.get('start_coords')}")


        elif action_type_str == "right_single":
            mouse_button = VNCMouseButton.right
            mouse_position = None
            if 'start_coords' in action_inputs and isinstance(action_inputs['start_coords'], list) and len(action_inputs['start_coords']) == 2:
                 mouse_position = Position(int(action_inputs['start_coords'][0]), int(action_inputs['start_coords'][1]))

            if mouse_position is not None:
                 executable_actions.append(MouseAction(
                    mouse_action_type=MouseActionType.click,
                    mouse_button=mouse_button,
                    mouse_position=mouse_position
                 ))
            else:
                 print(f"Warning: Skipping right_single action due to missing/invalid mouse_position derived from input: {action_inputs.get('start_box') or action_inputs.get('point')}. Parsed coords: {action_inputs.get('start_coords')}")


        elif action_type_str == "drag":
            mouse_button = VNCMouseButton.left # Default to left drag
            if 'mouse_button' in action_inputs and action_inputs['mouse_button']:
                try:
                    mouse_button = VNCMouseButton[action_inputs['mouse_button'].lower()]
                except KeyError:
                    print(f"Warning: Invalid mouse_button '{action_inputs['mouse_button']}' for drag. Using 'left'.")
                    mouse_button = VNCMouseButton.left

            start_position = None
            if 'start_coords' in action_inputs and isinstance(action_inputs['start_coords'], list) and len(action_inputs['start_coords']) == 2:
                 start_position = Position(int(action_inputs['start_coords'][0]), int(action_inputs['start_coords'][1]))

            end_position = None
            if 'end_coords' in action_inputs and isinstance(action_inputs['end_coords'], list) and len(action_inputs['end_coords']) == 2:
                 end_position = Position(int(action_inputs['end_coords'][0]), int(action_inputs['end_coords'][1]))

            if start_position is not None and end_position is not None:
                 executable_actions.append(MouseAction(mouse_action_type=MouseActionType.move, mouse_position=start_position))
                 executable_actions.append(WaitAction(wait_time=0.05))
                 executable_actions.append(MouseAction(mouse_action_type=MouseActionType.down, mouse_button=mouse_button, mouse_position=start_position))
                 executable_actions.append(WaitAction(wait_time=0.05))
                 executable_actions.append(MouseAction(mouse_action_type=MouseActionType.move, mouse_position=end_position))
                 executable_actions.append(WaitAction(wait_time=0.05))
                 executable_actions.append(MouseAction(mouse_action_type=MouseActionType.up, mouse_button=mouse_button, mouse_position=end_position))
            else:
                 print(f"Warning: Skipping drag action due to missing/invalid start or end position derived from inputs: {action_inputs.get('start_box') or action_inputs.get('point')}, {action_inputs.get('end_box')}. Parsed coords: {action_inputs.get('start_coords')}, {action_inputs.get('end_coords')}")


        elif action_type_str == "hotkey":
            keyboard_key_str = action_inputs.get('key')
            if isinstance(keyboard_key_str, str) and keyboard_key_str.strip():
                 try:
                      executable_actions.append(KeyboardAction(
                         keyboard_action_type=KeyboardActionType.press,
                         keyboard_key=keyboard_key_str.strip()
                      ))
                 except IncompleteActionDataError as e:
                      print(f"Warning: Failed to create KeyboardAction (hotkey) from '{keyboard_key_str}': {e}")
            else:
                 print(f"Warning: Skipping hotkey action due to missing/invalid 'key' input: {action_inputs.get('key')}")

        elif action_type_str == "press":
             keyboard_key_str = action_inputs.get('key')
             if isinstance(keyboard_key_str, str) and keyboard_key_str.strip():
                 try:
                      executable_actions.append(KeyboardAction(
                         keyboard_action_type=KeyboardActionType.down,
                         keyboard_key=keyboard_key_str.strip()
                      ))
                 except IncompleteActionDataError as e:
                      print(f"Warning: Failed to create KeyboardAction (press/down) from '{keyboard_key_str}': {e}")
             else:
                 print(f"Warning: Skipping press action due to missing/invalid 'key' input: {action_inputs.get('key')}")

        elif action_type_str == "release":
             keyboard_key_str = action_inputs.get('key')
             if isinstance(keyboard_key_str, str) and keyboard_key_str.strip():
                 try:
                      executable_actions.append(KeyboardAction(
                         keyboard_action_type=KeyboardActionType.up,
                         keyboard_key=keyboard_key_str.strip()
                      ))
                 except IncompleteActionDataError as e:
                      print(f"Warning: Failed to create KeyboardAction (release/up) from '{keyboard_key_str}': {e}")
             else:
                 print(f"Warning: Skipping release action due to missing/invalid 'key' input: {action_inputs.get('key')}")

        elif action_type_str == "type":
            keyboard_text = action_inputs.get('content')
            if isinstance(keyboard_text, str):
                 executable_actions.append(KeyboardAction(
                    keyboard_action_type=KeyboardActionType.text,
                    keyboard_text=keyboard_text
                 ))
            else:
                 print(f"Warning: Skipping type action due to missing/invalid 'content' input: {action_inputs.get('content')}")


        elif action_type_str == "scroll":
            direction_str = action_inputs.get('direction')
            if isinstance(direction_str, str):
                 direction_lower = direction_str.strip().lower()
                 scroll_repeat = 1

                 if direction_lower == 'up':
                      executable_actions.append(MouseAction(mouse_action_type=MouseActionType.scroll_up, scroll_repeat=scroll_repeat))
                 elif direction_lower == 'down':
                      executable_actions.append(MouseAction(mouse_action_type=MouseActionType.scroll_down, scroll_repeat=scroll_repeat))
                 elif direction_lower == 'left' or direction_lower == 'right':
                      print(f"Warning: Horizontal scroll direction '{direction_str}' predicted but not directly supported by MouseAction.")
                 else:
                      print(f"Warning: Unknown scroll direction '{direction_str}' predicted. Skipping action.")
            else:
                 print(f"Warning: Skipping scroll action due to missing/invalid 'direction' input: {action_inputs.get('direction')}")


        elif action_type_str == "wait":
            wait_time = 5.0
            executable_actions.append(WaitAction(wait_time=wait_time))

        elif action_type_str == "finished":
            pass

        elif action_type_str == "call_user":
             pass

        else:
            print(f"Warning: Unknown predicted action type '{action_type_str}'. Skipping.")


    except Exception as e:
        print(f"Critical Error: Failed to convert parsed prediction {parsed_prediction} to executable Action(s): {e}")
        executable_actions = [WaitAction(wait_time=10.0)]


    return executable_actions


class VNCWidget(QMainWindow):

    def __init__(self, remote_vnc_server_config, llm_client, automaton):
        super().__init__()

        self.remote_vnc_server_config = remote_vnc_server_config

        self.task_list_path = remote_vnc_server_config.get("task_list", "task_list.txt")

        self.action_queue = queue.Queue()

        self.video_width = 640
        self.video_height = 480
        self.now_screenshot = None

        self.llm_client_gpt = llm_client

        self._request_recall_func_cache = {}
        self._request_action_num_counter = {}

        # --- Initialize UI Elements and core attributes FIRST ---
        # Initialize attributes that will be accessed early
        self.task_prompt_list = []
        self.parsed_prediction_list: List[PredictionParsed] = []
        self._parsed_prediction_display_map: Dict[QListWidgetItem, PredictionParsed] = {}
        self._parsed_prediction_item_to_row: Dict[PredictionParsed, int] = {}

        # Initialize UI widgets before they are used in methods like clear_vlm_history or load_prompts_from_file
        self.task_prompt_selection = QListWidget(self)
        self.sub_task_display = QListWidget(self)
        self.task_prompt_display = QTextEdit(self)
        self.current_instruction_display = QTextEdit(self)
        self.send_prompt_display = QTextEdit(self)
        self.LLM_response_display = QTextEdit(self)
        self.LLM_response_editer = QTextEdit(self)
        self.parse_action_display = QListWidget(self)
        self.vnc_frame = VNCFrame(self, self.action_queue) # Instantiate VNCFrame after its class definition


        # --- Layout Setup ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setMouseTracking(True)

        main_layout = QHBoxLayout(central_widget)

        # Left Column
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Connect Control"))
        reconnect_button = QPushButton("Re-connect")
        reconnect_button.clicked.connect(self.reconnect)
        left_layout.addWidget(reconnect_button)

        left_layout.addWidget(QLabel("Task Selection"))
        left_layout.addWidget(self.task_prompt_selection) # Use the initialized widget
        self.task_prompt_selection.itemDoubleClicked.connect(self.select_task_prompt)


        reload_button = QPushButton("Reload Task List")
        reload_button.clicked.connect(self.load_prompts_from_file)
        left_layout.addWidget(reload_button)

        left_layout.addWidget(QLabel("Automaton Control"))

        self.start_automaton_button = QPushButton("Start Automaton")
        self.start_automaton_button.clicked.connect(self.start_automaton)
        left_layout.addWidget(self.start_automaton_button)

        self.automaton_control_buttons = []
        if hasattr(automaton, 'machine') and hasattr(automaton.machine, 'states'):
             for state in automaton.machine.states:
                 state_str = str(state)
                 button = QPushButton(state_str)
                 button.clicked.connect(partial(self.set_automaton_state, state_str))
                 self.automaton_control_buttons.append(button)
                 left_layout.addWidget(button)
        else:
             print("Warning: Automaton machine or states not available, skipping state control buttons.")


        self.set_auto_transitions_checkbox = QCheckBox("Enable Auto Transitions")
        if hasattr(automaton, 'set_auto_transitions'):
             self.set_auto_transitions_checkbox.setChecked(True)
             def set_auto_transitions():
                 automaton.set_auto_transitions(self.set_auto_transitions_checkbox.isChecked())
             self.set_auto_transitions_checkbox.stateChanged.connect(set_auto_transitions)
             left_layout.addWidget(self.set_auto_transitions_checkbox)
        else:
             print("Warning: Automaton does not have set_auto_transitions method, hiding checkbox.")
             self.set_auto_transitions_checkbox.hide()


        left_layout.addWidget(QLabel("Sub Tasks (Planner View - may not apply to VLM)"))
        left_layout.addWidget(self.sub_task_display) # Use the initialized widget
        if hasattr(automaton, 'set_current_task_index'):
             self.sub_task_display.itemDoubleClicked.connect(self.set_current_task_index)
        else:
             print("Warning: Automaton does not handle setting current task index, double-click on subtask list disabled.")

        main_layout.addLayout(left_layout) # Add left layout

        # Middle Column
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(QLabel(f"Task Prompt (Overall Goal)", self))
        middle_layout.addWidget(self.task_prompt_display) # Use the initialized widget
        self.task_prompt_display.setReadOnly(False)
        self.task_prompt_display.setPlainText("Please first select a task prompt from the task selection")
        self.task_prompt_display.setFixedHeight(60)

        middle_layout.addWidget(QLabel(f"Current Instruction for VLM", self))
        middle_layout.addWidget(self.current_instruction_display) # Use the initialized widget
        self.current_instruction_display.setReadOnly(False)
        self.current_instruction_display.setPlainText("Enter specific instruction for the next VLM turn here.")
        self.current_instruction_display.setFixedHeight(60)


        middle_layout.addWidget(QLabel(f"Prompt Sent to VLM", self))
        middle_layout.addWidget(self.send_prompt_display) # Use the initialized widget
        self.send_prompt_display.setReadOnly(True)
        self.send_prompt_display.setFixedHeight(150)


        middle_layout.addWidget(self.vnc_frame) # Use the initialized widget
        self.vnc_frame.setGeometry(0, 0, 640, 480)

        main_layout.addLayout(middle_layout) # Add middle layout


        # Right Column
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("VLM Response and Action Display"))

        clear_LLM_button = QPushButton("Clear VLM Response History")
        clear_LLM_button.clicked.connect(self.clear_vlm_history)
        right_layout.addWidget(clear_LLM_button)

        right_layout.addWidget(QLabel("VLM Original Response (read only)"))
        right_layout.addWidget(self.LLM_response_display) # Use the initialized widget
        self.LLM_response_display.setReadOnly(True)
        self.LLM_response_display.setFixedHeight(150)


        right_layout.addWidget(QLabel("VLM Response Editor"))
        right_layout.addWidget(self.LLM_response_editer) # Use the initialized widget
        self.LLM_response_editer.setReadOnly(False)
        self.LLM_response_editer.setFixedHeight(300)

        parse_action_button = QPushButton("Parse Actions from Editor (using VLM parser)")
        parse_action_button.clicked.connect(self.parse_actions_from_editor_text)
        right_layout.addWidget(parse_action_button)

        right_layout.addWidget(QLabel("Parsed Predictions (Thought/Actions)"))
        right_layout.addWidget(self.parse_action_display) # Use the initialized widget
        self.parse_action_display.setWordWrap(True)
        # Dictionaries are initialized earlier


        self.set_auto_execute_actions_checkbox = QCheckBox("Enable Auto Execute Predicted Actions")
        if hasattr(automaton, 'set_auto_execute_actions'):
             self.set_auto_execute_actions_checkbox.setChecked(True)
             def set_auto_execute_actions():
                 automaton.set_auto_execute_actions(self.set_auto_execute_actions_checkbox.isChecked())
             self.set_auto_execute_actions_checkbox.stateChanged.connect(set_auto_execute_actions)
             right_layout.addWidget(self.set_auto_execute_actions_checkbox)
        else:
             print("Warning: Automaton does not have set_auto_execute_actions method, hiding checkbox.")
             self.set_auto_execute_actions_checkbox.hide()


        run_parsed_predictions_button = QPushButton("Run Parsed Predicted Actions")
        run_parsed_predictions_button.clicked.connect(self.run_parsed_predictions_func)
        right_layout.addWidget(run_parsed_predictions_button)

        main_layout.addLayout(right_layout) # Add right layout


        # --- Post-Layout Initialization (after adding layouts) ---

        self.video_width = 640
        self.video_height = 480
        self.now_screenshot = None
        self.vnc = None
        self.automaton = automaton
        self.llm_client_gpt = llm_client


        self.vnc_frame.setFixedSize(self.video_width, self.video_height)
        self.vnc_frame.setMouseTracking(True)


        self.task_prompt_display.setFixedWidth(self.vnc_frame.width())
        self.current_instruction_display.setFixedWidth(self.vnc_frame.width())
        self.send_prompt_display.setFixedWidth(self.vnc_frame.width())
        self.LLM_response_display.setFixedWidth(self.vnc_frame.width())
        self.LLM_response_editer.setFixedWidth(self.vnc_frame.width())
        self.parse_action_display.setFixedWidth(self.vnc_frame.width())


        # Load prompts now that task_prompt_selection and task_prompt_display are guaranteed to exist
        self.load_prompts_from_file()


        # General initial setup
        self.setWindowTitle("ScreenAgent VNC Viewer")
        self.setGeometry(100, 100, 1200, 800)
        self.setMouseTracking(True)


        conversation_template_name = remote_vnc_server_config.get("conversation_template", "uitars_doubao_20b_en_normal")
        print(f"Using conversation template: '{conversation_template_name}'")
        template = conv_templates.get(conversation_template_name)
        if template is None:
             print(f"Warning: Conversation template '{conversation_template_name}' not found. Falling back to 'uitars_doubao_20b_en_normal'.")
             template = conv_templates["uitars_doubao_20b_en_normal"]
        self.conversation: Conversation = template.copy()


        self.task_prompt: str = "" # Set by load_prompts_from_file or select_task_prompt
        self.current_instruction: str = "" # Instruction for the current VLM turn
        self.send_prompt: str = "" # Actual prompt text sent
        self.last_vlm_response_raw: str = "" # Raw text response
        # self.parsed_prediction_list is initialized earlier


        self.last_message: str = ""

        self._request_recall_func_cache = {}
        self._request_action_num_counter = {}


        if hasattr(self.automaton, 'link_to_vncwidget'):
            self.automaton.link_to_vncwidget(self)
        else:
            print("Warning: Automaton class does not have a 'link_to_vncwidget' method.")

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(10)
        self.refresh_timer.timeout.connect(self.render)
        # Timer is started in connect_vnc


        self.refreshing_screen = False
        self.wait_for_screen_refreshed = False

        # Start the VNC connection process using QMetaObject to call asyncSlot
        QMetaObject.invokeMethod(self, "connect_vnc", Qt.QueuedConnection)


    @pyqtSlot()
    def parse_actions_from_editor_text(self):
        """Manually parses text from the VLM Response Editor."""
        print("[func:parse_actions_from_editor_text] Parsing text from editor...")
        raw_text = self.LLM_response_editer.toPlainText()
        self.last_vlm_response_raw = raw_text # Update the stored raw response

        screen_context = {'width': self.video_width, 'height': self.video_height}
        vlm_factors = self.remote_vnc_server_config.get("vlm_image_factors", (1000, 1000))
        vlm_scale_factor = self.remote_vnc_server_config.get("vlm_scale_factor", 1.0)
        vlm_prompt_mode = self.remote_vnc_server_config.get("vlm_prompt_mode", "bc")
        vlm_model_ver_str = self.conversation.model_ver_ts.value if self.conversation else "V1_0"

        try:
            parsed_preds = parse_ui_tars_prediction(
                prediction=raw_text,
                factor=vlm_factors,
                screen_context=screen_context,
                scale_factor=vlm_scale_factor,
                mode=vlm_prompt_mode,
                model_ver=vlm_model_ver_str
            )
            self.parsed_prediction_list = parsed_preds
            print(f"[func:parse_actions_from_editor_text] Parsed {len(parsed_preds)} predictions.")

        except Exception as e:
            print(f"[func:parse_actions_from_editor_text] Error during parsing: {e}")
            self.last_message = f"Parsing Error: {e}"
            self.parsed_prediction_list = []
            self.set_status_text()


        self.update_parsed_action_display()

        if hasattr(self.automaton, 'set_parsed_predictions'):
             self.automaton.set_parsed_predictions(self.parsed_prediction_list)

        self.last_message = f"Parsed {len(self.parsed_prediction_list)} predictions from editor."
        self.set_status_text()


    @pyqtSlot()
    def run_parsed_predictions_func(self):
        """Manually runs the actions derived from the currently parsed predictions."""
        print("[func:run_parsed_predictions_func] Running parsed predicted actions...")
        if not self.parsed_prediction_list:
            print("No parsed predictions to run.")
            self.last_message = "No predictions to run."
            self.set_status_text()
            return

        executable_actions_list: List[Action] = []
        screen_context = {'width': self.video_width, 'height': self.video_height}

        for i, pred in enumerate(self.parsed_prediction_list):
             try:
                 actions_from_pred = convert_prediction_to_actions(pred, screen_context)
                 executable_actions_list.extend(actions_from_pred)
             except Exception as e:
                  print(f"Error converting prediction {i} to actions: {e}")


        if executable_actions_list:
            execution_request_id = uuid.uuid4().hex
            print(f"[func:run_parsed_predictions_func] Queuing {len(executable_actions_list)} executable actions for request_id: {execution_request_id}")

            execution_recall_func = partial(self.automaton.manual_action_execution_complete, execution_request_id) if hasattr(self.automaton, 'manual_action_execution_complete') else None

            self.queue_executable_actions(execution_request_id, executable_actions_list, execution_recall_func)

            self.highlight_predicted_batch_queued(self.parsed_prediction_list)

            self.last_message = f"Queued {len(executable_actions_list)} actions from parsed predictions."

        else:
             print("No executable actions generated from parsed predictions.")
             self.last_message = "No executable actions generated."

        self.set_status_text()


    def load_prompts_from_file(self):
        """Loads task prompts from the specified task list file."""
        try:
            with open(self.task_list_path, "r", encoding="utf8") as f:
                task_prompt_list = [line.strip() for line in f if line.strip()]
                self.task_prompt_list = task_prompt_list
            if hasattr(self, 'task_prompt_selection') and self.task_prompt_selection is not None:
                 self.task_prompt_selection.clear()
                 self.task_prompt_selection.addItems(self.task_prompt_list)
                 if self.task_prompt_list:
                      self.task_prompt_selection.setCurrentRow(0)
                      first_item = self.task_prompt_selection.item(0)
                      if first_item:
                           self.select_task_prompt(first_item)
            else:
                 print("Warning: task_prompt_selection widget not initialized yet when loading prompts.")

            self.last_message = f"Loaded {len(self.task_prompt_list)} tasks."

        except FileNotFoundError:
            self.task_prompt_list = []
            if hasattr(self, 'task_prompt_selection') and self.task_prompt_selection is not None:
                 self.task_prompt_selection.clear()
            self.last_message = f"Task list file not found: {self.task_list_path}"
            print(self.last_message)
        except Exception as e:
            self.task_prompt_list = []
            if hasattr(self, 'task_prompt_selection') and self.task_prompt_selection is not None:
                 self.task_prompt_selection.clear()
            self.last_message = f"Error loading task list: {e}"
            print(self.last_message)

        self.set_status_text()


    @asyncSlot()
    async def reconnect(self):
        """Disconnects and attempts to reconnect to the VNC server."""
        print("[func:reconnect] Attempting to reconnect...")
        self.action_queue.queue.clear()
        self.reset()

        if self.vnc is not None:
            try:
                 await self.vnc.disconnect()
                 print("[func:reconnect] Disconnected from VNC.")
            except Exception as e:
                 print(f"[func:reconnect] Error during disconnect: {e}")
            self.vnc = None

        await asyncio.sleep(1)
        QMetaObject.invokeMethod(self, "connect_vnc", Qt.QueuedConnection)


    @asyncSlot()
    async def connect_vnc(self):
        """Establishes VNC connection and initializes UI based on screen dimensions."""
        self.statusBar().showMessage("Connecting...")
        print("[func:connect_vnc] Connecting...")

        host = self.remote_vnc_server_config.get("host", "localhost")
        port = self.remote_vnc_server_config.get("port", 5900)
        password = self.remote_vnc_server_config.get("password")

        try:
            self._reader, self._writer = await open_connection(host, port)
            self.vnc = await Client.create(reader=self._reader, writer=self._writer, password=password)
            print(f"[func:connect_vnc] Connected to {host}:{port}")

            self.video_height = self.vnc.video.height
            self.video_width = self.vnc.video.width
            print(f"[func:connect_vnc] Screen resolution: {self.video_width}x{self.video_height}")

            self.now_screenshot = np.zeros((self.video_height, self.video_width, 4), dtype='uint8')

            if hasattr(self, 'vnc_frame') and self.vnc_frame is not None:
                 self.vnc_frame.setFixedSize(self.video_width, self.video_height)
                 self.vnc_frame.setMouseTracking(True)
            else:
                 print("Warning: vnc_frame widget not initialized yet during connect_vnc.")


            if hasattr(self, 'task_prompt_display') and self.task_prompt_display is not None:
                 self.task_prompt_display.setFixedWidth(self.vnc_frame.width())
            if hasattr(self, 'current_instruction_display') and self.current_instruction_display is not None:
                 self.current_instruction_display.setFixedWidth(self.vnc_frame.width())
            if hasattr(self, 'send_prompt_display') and self.send_prompt_display is not None:
                 self.send_prompt_display.setFixedWidth(self.vnc_frame.width())
            if hasattr(self, 'LLM_response_display') and self.LLM_response_display is not None:
                 self.LLM_response_display.setFixedWidth(self.vnc_frame.width())
            if hasattr(self, 'LLM_response_editer') and self.LLM_response_editer is not None:
                 self.LLM_response_editer.setFixedWidth(self.vnc_frame.width())
            if hasattr(self, 'parse_action_display') and self.parse_action_display is not None:
                 self.parse_action_display.setFixedWidth(self.vnc_frame.width())


            self.refresh_timer.start()

            self.statusBar().showMessage("Connected. Ready.")
            print("[func:connect_vnc] VNC Ready.")
            self.reset()

        except ConnectionRefusedError:
            msg = f"Connection refused to {host}:{port}"
            self.statusBar().showMessage(msg)
            print(f"Error: {msg}")
            self.vnc = None
        except TimeoutError:
             msg = f"Connection timed out to {host}:{port}"
             self.statusBar().showMessage(msg)
             print(f"Error: {msg}")
             self.vnc = None
        except Exception as e:
            msg = f"Connection error: {e}"
            self.statusBar().showMessage(msg)
            print(f"Error connecting to VNC: {e}")
            self.vnc = None

        self.set_status_text()


    def reset(self):
        """Resets the UI and internal state for a new task or session."""
        print("[func:reset] Resetting state...")

        if hasattr(self, 'LLM_response_display') and self.LLM_response_display is not None:
             self.LLM_response_display.clear()
        if hasattr(self, 'LLM_response_editer') and self.LLM_response_editer is not None:
             self.LLM_response_editer.clear()
        if hasattr(self, 'parse_action_display') and self.parse_action_display is not None:
             self.parse_action_display.clear()

        self.last_vlm_response_raw = ""
        self.parsed_prediction_list.clear()
        self._parsed_prediction_display_map.clear()
        self._parsed_prediction_item_to_row.clear()


        self.action_queue.queue.clear()

        self._request_recall_func_cache.clear()
        self._request_action_num_counter.clear()

        self.saved_image_name = None

        conversation_template_name = self.remote_vnc_server_config.get("conversation_template", "uitars_doubao_20b_en_normal")
        template = conv_templates.get(conversation_template_name)
        if template is None:
             template = conv_templates["uitars_doubao_20b_en_normal"]
        self.conversation: Conversation = template.copy()
        print(f"[func:reset] Reset Conversation history using template '{conversation_template_name}'.")

        self.current_instruction = ""
        if hasattr(self, 'current_instruction_display') and self.current_instruction_display is not None:
             self.current_instruction_display.setPlainText("Enter specific instruction for the next VLM turn here.")
        if hasattr(self, 'send_prompt_display') and self.send_prompt_display is not None:
             self.send_prompt_display.clear()
        self.send_prompt = ""


        self.last_message = "System Reset."
        self.set_status_text()

        if hasattr(self.automaton, 'reset'):
             self.automaton.reset()
             print("[func:reset] Automaton reset.")

        if hasattr(self, 'sub_task_display') and self.sub_task_display is not None:
             self.sub_task_display.clear()


    def clear_vlm_history(self):
         """Clears only the VLM conversation history."""
         print("[func:clear_vlm_history] Clearing VLM conversation history...")
         conversation_template_name = self.remote_vnc_server_config.get("conversation_template", "uitars_doubao_20b_en_normal")
         template = conv_templates.get(conversation_template_name)
         if template is None:
             template = conv_templates["uitars_doubao_20b_en_normal"]
         self.conversation: Conversation = template.copy()

         self.last_vlm_response_raw = ""
         if hasattr(self, 'LLM_response_display') and self.LLM_response_display is not None:
              self.LLM_response_display.clear()
         if hasattr(self, 'LLM_response_editer') and self.LLM_response_editer is not None:
              self.LLM_response_editer.clear()

         self.parsed_prediction_list.clear()
         self._parsed_prediction_display_map.clear()
         self._parsed_prediction_item_to_row.clear()
         self.update_parsed_action_display()
         self.last_message = "VLM history cleared."
         self.set_status_text()


    def set_status_text(self):
        """Updates the status bar with current information."""
        all_status_text = []
        state_str = self.automaton.state if self.automaton and hasattr(self.automaton, 'state') else 'N/A'
        all_status_text.append(f"State: {state_str}")
        all_status_text.append(self.last_message)

        if action_queue_size:=self.action_queue.qsize():
            all_status_text.append(f"{action_queue_size} Actions Queued.")
        if hasattr(self, 'vnc_frame') and self.vnc_frame and hasattr(self.vnc_frame, 'is_in_focus') and self.vnc_frame.is_in_focus and hasattr(self.vnc_frame, 'get_local_cursor_pos') and (local_cursor_pos:=self.vnc_frame.get_local_cursor_pos()):
            all_status_text.append(f"Manual Cursor: {str(local_cursor_pos)}")

        self.statusBar().showMessage(" | ".join(all_status_text))


    @asyncSlot()
    async def update_screen(self):
        """Captures a new screenshot from VNC and updates the display."""
        if self.vnc is None:
            print("[update_screen] VNC client is not connected.")
            return

        try:
            new_screenshot_rgba = await self.vnc.screenshot()
            if new_screenshot_rgba is not None:
                self.now_screenshot = new_screenshot_rgba
                if new_screenshot_rgba.shape[0] == self.video_height and new_screenshot_rgba.shape[1] == self.video_width:
                    qimage = QImage(self.now_screenshot.tobytes(), self.video_width, self.video_height, QImage.Format_RGBA8888)
                    if hasattr(self, 'vnc_frame') and self.vnc_frame is not None:
                         self.vnc_frame.update_screen(qimage)
                    else:
                        print("Warning: vnc_frame not available to update screen display.")
                else:
                    print(f"Warning: Screenshot dimensions {new_screenshot_rgba.shape[0]}x{new_screenshot_rgba.shape[1]} mismatch VNC dimensions {self.video_height}x{self.video_width}.")


            self.wait_for_screen_refreshed = False

        except Exception as e:
            print(f"[update_screen] Error capturing or updating screen: {e}")
            if self.vnc is not None and "closed" in str(e).lower():
                 print("[update_screen] VNC connection appears closed. Attempting reconnect.")
                 if hasattr(self, 'reconnect'):
                     print("[update_screen] 'reconnect' method exists on self. Calling now.")
                     QMetaObject.invokeMethod(self, "reconnect", Qt.QueuedConnection)
                 else:
                     print("[update_screen] CRITICAL: 'reconnect' method DOES NOT exist on self! Cannot attempt recovery.")
                     if hasattr(self.automaton, 'handle_critical_error'):
                         self.automaton.handle_critical_error(f"VNC reconnect failed during update screen due to missing method.")
            else:
                 print(f"[update_screen] Unhandled exception: {e}")
                 if hasattr(self.automaton, 'handle_non_fatal_error'):
                      self.automaton.handle_non_fatal_error(f"Screenshot error: {e}")


    @asyncSlot()
    async def render(self):
        """Main render and action execution loop, triggered by QTimer."""
        self.refresh_timer.stop()

        await self.update_screen()

        if self.now_screenshot is None:
             print("[render] Skipping action execution due to failed screenshot capture.")
             self.refreshing_screen = False
             self.refresh_timer.start()
             return

        executed_count = 0
        completed_request_ids = set()

        try:
            while not self.action_queue.empty():
                action: Action = self.action_queue.get_nowait()

                if action.request_id is None:
                    action.request_id = "system_manual"

                if action.before_action_obs is None:
                     action.before_action_obs = self.now_screenshot

                print(f"Executing action (Request: {action.request_id[:8]}...): {action.action_type}")

                await action.step(self.vnc)
                executed_count += 1

                await asyncio.sleep(0.05)

                action_request_id = action.request_id
                if action_request_id in self._request_action_num_counter:
                    self._request_action_num_counter[action_request_id] -= 1
                    if self._request_action_num_counter[action_request_id] <= 0:
                        completed_request_ids.add(action_request_id)


        except queue.Empty:
            pass
        except Exception as e:
            print(f"[render] Error during action execution: {e}")
            if self.vnc is not None and "closed" in str(e).lower():
                 print("[render] VNC connection appears closed during action execution. Attempting reconnect.")
                 if hasattr(self, 'reconnect'):
                     print("[render] 'reconnect' method exists on self. Calling now.")
                     QMetaObject.invokeMethod(self, "reconnect", Qt.QueuedConnection)
                 else:
                     print("[render] CRITICAL: 'reconnect' method DOES NOT exist on self! Cannot attempt recovery.")
                     if hasattr(self.automaton, 'handle_critical_error'):
                         self.automaton.handle_critical_error(f"VNC reconnect failed during action execution due to missing method.")
            else:
                 print(f"[render] Unhandled exception: {e}")
                 if hasattr(self.automaton, 'handle_non_fatal_error'):
                      self.automaton.handle_non_fatal_error(f"Action execution error: {e}")


        for request_id in completed_request_ids:
             request_recall_func = self._request_recall_func_cache.get(request_id)
             if request_recall_func:
                 try:
                     print(f"[render] Calling recall_func for completed batch request_id: {request_id}")
                     QMetaObject.invokeMethod(self, "execute_callback_on_main_thread", Qt.QueuedConnection, Q_ARG(object, request_recall_func))
                 except Exception as cb_e:
                     print(f"Error executing callback for request_id {request_id}: {cb_e}")

             self._request_recall_func_cache.pop(request_id, None)
             self._request_action_num_counter.pop(request_id, None)


        self.set_status_text()

        self.refreshing_screen = False
        self.refresh_timer.start()


    def select_task_prompt(self, item):
        """Handles selection of a task from the task list."""
        self.task_prompt = item.text()
        if hasattr(self, 'task_prompt_display') and self.task_prompt_display is not None:
             self.task_prompt_display.setText(self.task_prompt)
        else:
             print("Warning: task_prompt_display not available during select_task_prompt.")

        self.last_message = f"Task selected: {self.task_prompt[:50]}..."
        self.set_status_text()

        self.clear_vlm_history()

        if hasattr(self.automaton, 'set_task_prompt') and hasattr(self.automaton, 'set_current_instruction'):
             self.automaton.set_task_prompt(self.task_prompt)
             self.current_instruction = self.task_prompt
             self.automaton.set_current_instruction(self.current_instruction)
             if hasattr(self, 'current_instruction_display') and self.current_instruction_display is not None:
                  self.current_instruction_display.setPlainText(self.current_instruction)
             print(f"[func:select_task_prompt] Automaton task set to: {self.task_prompt[:50]}...")
        else:
             print("Warning: Automaton does not have set_task_prompt or set_current_instruction methods.")


    def start_automaton(self):
        """Starts the automaton (single-VLM) flow."""
        print(f"[func:start_automaton] Starting automaton for task: {self.task_prompt}")
        if not self.task_prompt:
            print("Error: No task prompt selected.")
            self.last_message = "Error: No task prompt selected."
            self.set_status_text()
            return

        if self.vnc is None:
             print("Error: VNC connection is not established.")
             self.last_message = "Error: Cannot start automaton (VNC not connected)."
             self.set_status_text()
             return


        self.reset()

        self.current_instruction = self.task_prompt
        if hasattr(self.automaton, 'set_current_instruction'):
             self.automaton.set_current_instruction(self.current_instruction)
             if hasattr(self, 'current_instruction_display') and self.current_instruction_display is not None:
                  self.current_instruction_display.setPlainText(self.current_instruction)
        else:
            print("Warning: Automaton does not have set_current_instruction method.")


        if hasattr(self.automaton, 'start'):
             # Call the start method on the automaton instance
             # The automaton's start method will trigger the state machine's 'initiate_start' trigger internally.
             self.automaton.start(
                 task_prompt=self.task_prompt,
                 video_width=self.video_width,
                 video_height=self.video_height,
                 conversation=self.conversation
             )
             self.last_message = f"Automaton started for task: {self.task_prompt[:50]}..."
        else:
             print("Error: Automaton does not have a 'start' method.")
             self.last_message = "Error: Automaton not configured correctly."
        self.set_status_text()


    def set_automaton_state(self, state: str):
        """Manually sets the automaton state."""
        print(f"[func:set_automaton_state] Manually setting state to: {state}")
        if hasattr(self.automaton, 'set_state'):
             self.automaton.set_state(state)
             self.last_message = f"Automaton state manually set to: {state}"
        else:
             print("Error: Automaton does not have a 'set_state' method.")
             self.last_message = "Error: Cannot manually set automaton state."
        self.set_status_text()


    @pyqtSlot(str) # Mark as a slot accepting a string
    def automaton_state_changed(self, state):
        """Callback triggered by the automaton when its state changes."""
        print(f"[func:automaton_state_changed] Automaton transitioned to state: {state}")
        if hasattr(self, 'automaton_control_buttons'):
             for button in self.automaton_control_buttons:
                 if button.text() == str(state):
                     button.setStyleSheet("background-color: green")
                 else:
                     button.setStyleSheet("")
        self.set_status_text()


    def queue_executable_actions(self, request_id: str, actions: List[Action], recall_func=None):
        """
        Receives a list of executable Action objects from another component (e.g., automaton),
        queues them for execution in the render loop, and registers a callback
        to be executed when all actions in this batch are complete.
        """
        print(f"[func:queue_executable_actions] Received {len(actions)} actions for request_id: {request_id}")
        if not actions:
            print(f"[func:queue_executable_actions] No actions to queue for request_id: {request_id}")
            if recall_func:
                try:
                    print(f"[func:queue_executable_actions] Calling recall_func immediately for empty batch {request_id}")
                    QMetaObject.invokeMethod(self, "execute_callback_on_main_thread", Qt.QueuedConnection, Q_ARG(object, recall_func))
                except Exception as cb_e:
                     print(f"Error executing empty batch callback for request_id {request_id}: {cb_e}")
            self.clear_request_cache(request_id)
            return

        self._request_recall_func_cache[request_id] = recall_func
        self._request_action_num_counter[request_id] = len(actions)

        for action in actions:
            action.request_id = request_id
            self.action_queue.put(action)

        self.last_message = f"Queued {len(actions)} actions for execution (Request: {request_id[:8]}...)."
        self.set_status_text()


    @pyqtSlot(object)
    def execute_callback_on_main_thread(self, callback_func):
        """Wrapper to execute a function on the main thread."""
        try:
            print(f"[execute_callback_on_main_thread] Executing callback: {callback_func}")
            callback_func()
        except Exception as e:
            print(f"[execute_callback_on_main_thread] Error in scheduled callback function: {e}")


    def set_current_instruction_display(self, instruction: str):
        """Sets the text in the current instruction display."""
        self.current_instruction = instruction
        if hasattr(self, 'current_instruction_display') and self.current_instruction_display is not None:
             self.current_instruction_display.setPlainText(self.current_instruction)
        else:
             print("Warning: current_instruction_display not available to set text.")

        self.last_message = f"Current instruction: {self.current_instruction[:50]}..."
        self.set_status_text()
        if hasattr(self.automaton, 'set_current_instruction'):
             self.automaton.set_current_instruction(self.current_instruction)


    def set_send_prompt_display(self, send_prompt: str):
        """Sets the text in the prompt sent to VLM display."""
        self.send_prompt = send_prompt
        display_prompt = send_prompt
        if len(display_prompt) > 500:
             display_prompt = display_prompt[:497] + "..."
        if hasattr(self, 'send_prompt_display') and self.send_prompt_display is not None:
             self.send_prompt_display.setPlainText(display_prompt)
        else:
             print("Warning: send_prompt_display not available to set text.")


    def set_llm_response(self, last_stream_response: str):
        """Sets the raw VLM response text in the UI."""
        self.last_vlm_response_raw = last_stream_response
        display_response = last_stream_response
        if len(display_response) > 500:
             display_response = display_response[:497] + "..."
        if hasattr(self, 'LLM_response_display') and self.LLM_response_display is not None:
             self.LLM_response_display.setPlainText(display_response)
        else:
             print("Warning: LLM_response_display not available to set text.")
        if hasattr(self, 'LLM_response_editer') and self.LLM_response_editer is not None:
             self.LLM_response_editer.setPlainText(last_stream_response)
        else:
            print("Warning: LLM_response_editer not available to set text.")


    def request_vlm_prediction(self, current_instruction: str, recall_func):
        """
        Requests a prediction from the VLM.
        Constructs the prompt using the Conversation object, sends it with the screenshot,
        and registers a callback for when the response is received and parsed.
        """
        print(f"[func:request_vlm_prediction] Requesting VLM prediction for instruction: {current_instruction[:50]}...")
        if self.vnc is None or self.now_screenshot is None:
             print("[func:request_vlm_prediction] Error: VNC not connected or no screenshot available.")
             self.last_message = "Error: Cannot request VLM prediction (no VNC/screenshot)."
             self.set_status_text()
             if recall_func:
                  QMetaObject.invokeMethod(self, "execute_callback_on_main_thread", Qt.QueuedConnection, Q_ARG(object, partial(recall_func, [])))
             return

        self.set_current_instruction_display(current_instruction)

        if self.conversation is None:
             print("[func:request_vlm_prediction] Error: Conversation object is None.")
             self.last_message = "Error: Cannot request VLM (Conversation object missing)."
             self.set_status_text()
             if recall_func:
                  QMetaObject.invokeMethod(self, "execute_callback_on_main_thread", Qt.QueuedConnection, Q_ARG(object, partial(recall_func, [])))
             return

        full_prompt = self.conversation.get_vlm_prompt(current_instruction=current_instruction)
        self.set_send_prompt_display(full_prompt)

        try:
             screenshot_img = Image.fromarray(self.now_screenshot).convert('RGB')
        except Exception as e:
             print(f"[func:request_vlm_prediction] Error converting screenshot to PIL Image: {e}")
             self.last_message = f"Error getting screenshot for VLM: {e}"
             self.set_status_text()
             if recall_func:
                  QMetaObject.invokeMethod(self, "execute_callback_on_main_thread", Qt.QueuedConnection, Q_ARG(object, partial(recall_func, [])))
             return


        request_id = uuid.uuid4().hex
        self._request_recall_func_cache[request_id] = recall_func

        print(f"[func:request_vlm_prediction] Sending request {request_id} to LLM client.")
        self.last_message = f"Asking VLM ({request_id[:8]}...)"
        self.set_status_text()

        if hasattr(self.llm_client_gpt, 'send_request_to_server'):
             try:
                  self.llm_client_gpt.send_request_to_server(
                      full_prompt,
                      screenshot_img,
                      request_id,
                      self._handle_vlm_response_received
                  )
             except Exception as e:
                  print(f"[func:request_vlm_prediction] Error sending request to LLM client: {e}")
                  self.last_message = f"Error sending to VLM: {e}"
                  self.set_status_text()
                  self.clear_request_cache(request_id)
                  if recall_func:
                       QMetaObject.invokeMethod(self, "execute_callback_on_main_thread", Qt.QueuedConnection, Q_ARG(object, partial(recall_func, [])))
        else:
             print("Error: LLM client does not have 'send_request_to_server' method.")
             self.last_message = "Error: LLM client not configured correctly."
             self.set_status_text()
             self.clear_request_cache(request_id)
             if recall_func:
                  QMetaObject.invokeMethod(self, "execute_callback_on_main_thread", Qt.QueuedConnection, Q_ARG(object, partial(recall_func, [])))


    def _handle_vlm_response_received(self, response_text: str, fail_message: str, request_id: str):
        """
        Callback executed when the LLM client receives a response or encounters an error.
        Schedules the main processing of the response on the main (GUI) thread.
        """
        print(f"[_handle_vlm_response_received] Response received for request_id: {request_id}. Success: {response_text is not None and response_text != ''}")
        original_recall_func = self._request_recall_func_cache.get(request_id)
        self._request_recall_func_cache.pop(request_id, None)
        self._request_action_num_counter.pop(request_id, None)


        if original_recall_func is None:
             print(f"Warning: Callback for request_id {request_id} not found in cache upon response receipt.")
             return

        QMetaObject.invokeMethod(self, "process_vlm_response_on_main_thread", Qt.QueuedConnection,
                                 Q_ARG(str, response_text if response_text is not None else ""),
                                 Q_ARG(str, fail_message if fail_message is not None else ""),
                                 Q_ARG(object, original_recall_func))


    @pyqtSlot(str, str, object)
    def process_vlm_response_on_main_thread(self, response_text: str, fail_message: str, original_recall_func):
        """
        Processes the VLM response text on the main thread.
        Parses actions, updates UI, stores history, and calls the original requesting callback.
        """
        print(f"[process_vlm_response_on_main_thread] Processing response.")

        parsed_preds: List[PredictionParsed] = []

        if response_text:
            self.set_llm_response(response_text)

            screen_context = {'width': self.video_width, 'height': self.video_height}
            vlm_factors = self.remote_vnc_server_config.get("vlm_image_factors", (1000, 1000))
            vlm_scale_factor = self.remote_vnc_server_config.get("vlm_scale_factor", 1.0)
            vlm_prompt_mode = self.remote_vnc_server_config.get("vlm_prompt_mode", "bc")
            vlm_model_ver_str = self.conversation.model_ver_ts.value if self.conversation else "V1_0"

            try:
                parsed_preds = parse_ui_tars_prediction(
                    prediction=response_text,
                    factor=vlm_factors,
                    screen_context=screen_context,
                    scale_factor=vlm_scale_factor,
                    mode=vlm_prompt_mode,
                    model_ver=vlm_model_ver_str
                )
                self.parsed_prediction_list = parsed_preds

                self.update_parsed_action_display()

                self.last_message = f"Parsed {len(self.parsed_prediction_list)} predictions."
                if len(self.parsed_prediction_list) == 0:
                    self.last_message = f"No valid predictions parsed from response."
                else:
                    if self.conversation:
                         self.conversation.append_model_output(response_text)
                         print(f"[process_vlm_response_on_main_thread] Added raw response to conversation history.")
                    else:
                         print("Warning: Conversation object is None, cannot add history.")


            except Exception as e:
                print(f"[process_vlm_response_on_main_thread] Error during parsing: {e}")
                self.last_message = f"VLM Response Parsing Error: {e}"
                self.parsed_prediction_list = []
                parsed_preds = []
                self.update_parsed_action_display()

        else:
            self.last_message = "VLM request failed: " + fail_message if fail_message else "VLM request failed."
            self.parsed_prediction_list = []
            parsed_preds = []
            self.update_parsed_action_display()


        self.set_status_text()

        try:
            print(f"[process_vlm_response_on_main_thread] Calling original recall_func with {len(parsed_preds)} predictions.")
            # The original recall_func (from automaton) receives the list of PredictionParsed objects
            original_recall_func(parsed_preds)
        except Exception as cb_e:
            print(f"[process_vlm_response_on_main_thread] Error executing original recall_func: {cb_e}")

        print(f"[process_vlm_response_on_main_thread] Finished processing response.")


    def update_sub_task_display(self, sub_task_list: List[str], now_sub_task_index: int):
        """Updates the display list for sub-tasks managed by the automaton."""
        print(f"[func:update_sub_task_display] Updating subtasks: {len(sub_task_list)} tasks, current index {now_sub_task_index}")
        if hasattr(self, 'sub_task_display') and self.sub_task_display is not None:
             self.sub_task_display.clear()
             for i, sub_task in enumerate(sub_task_list):
                 self.sub_task_display.addItem(f"{i}. {sub_task}")

             if 0 <= now_sub_task_index < self.sub_task_display.count():
                 for row in range(self.sub_task_display.count()):
                      item = self.sub_task_display.item(row)
                      if item:
                           item.setForeground(QColor("black"))

                 item = self.sub_task_display.item(now_sub_task_index)
                 if item:
                      item.setForeground(QColor("green"))
             elif self.sub_task_display.count() > 0:
                  print(f"Warning: Current subtask index {now_sub_task_index} is out of bounds for {self.sub_task_display.count()} tasks.")
        else:
             print("Warning: sub_task_display widget not available to update.")


    def set_current_task_index(self):
        """Handles double-click on sub-task list to set current task index in automaton."""
        if not hasattr(self, 'sub_task_display') or self.sub_task_display is None:
             print("Warning: Manual set_current_task_index attempted, but sub_task_display is missing.")
             return

        current_row = self.sub_task_display.currentRow()
        if current_row >= 0 and hasattr(self.automaton, 'set_current_task_index'):
            print(f"[func:set_current_task_index] Manually setting current task index to: {current_row}")
            self.automaton.set_current_task_index(current_row)
            self.last_message = f"Current task index set to {current_row}."
            self.set_status_text()
        else:
             print("Warning: Manual set_current_task_index attempted, but automaton method is missing or invalid row.")


    def get_current_task_description(self) -> Optional[str]:
        """
        Gets the description of the current sub-task or the main task.
        This method is likely called by the automaton when constructing the VLM prompt.
        """
        if self.automaton is not None and hasattr(self.automaton, 'get_current_task_description'):
            task_desc = self.automaton.get_current_task_description()
            if task_desc is not None:
                 return task_desc
            return self.task_prompt
        return self.task_prompt


    def clear_request_cache(self, request_id: str):
        """Clears cache entries for a given request ID."""
        self._request_recall_func_cache.pop(request_id, None)
        self._request_action_num_counter.pop(request_id, None)


    def update_parsed_action_display(self):
        """Updates the QListWidget display with the current parsed_prediction_list."""
        if not hasattr(self, 'parse_action_display') or self.parse_action_display is None:
             print("Warning: parse_action_display widget not available to update.")
             return

        self.parse_action_display.clear()
        self._parsed_prediction_display_map.clear()
        self._parsed_prediction_item_to_row.clear()

        if not self.parsed_prediction_list:
            self.parse_action_display.addItem("No predictions parsed.")
            return

        for i, pred in enumerate(self.parsed_prediction_list):
            input_str_parts = []
            for k, v in pred.action_inputs.items():
                if isinstance(v, str) and len(v) > 50:
                     v_display = f"'{v[:47]}...'"
                elif isinstance(v, str):
                     v_display = f"'{v}'"
                elif isinstance(v, list):
                     v_display = f"[{', '.join(map(str, v))}]"
                else:
                     v_display = str(v)
                input_str_parts.append(f"{k}={v_display}")

            input_str = ", ".join(input_str_parts)

            display_text = f"Thought: {pred.thought}\nAction: {pred.action_type}({input_str})"
            if pred.reflection:
                 display_text = f"Reflection: {pred.reflection}\n" + display_text

            item = QListWidgetItem(display_text)
            self.parse_action_display.addItem(item)
            self._parsed_prediction_display_map[item] = pred
            self._parsed_prediction_item_to_row[pred] = i

        print(f"[func:update_parsed_action_display] Displayed {len(self.parsed_prediction_list)} predictions.")


    def highlight_parsed_prediction_item(self, prediction_parsed: PredictionParsed, color: QColor = QColor("blue")):
         """Highlights the QListWidgetItem corresponding to a PredictionParsed object."""
         row = self._parsed_prediction_item_to_row.get(prediction_parsed)
         if hasattr(self, 'parse_action_display') and self.parse_action_display is not None and row is not None and 0 <= row < self.parse_action_display.count():
              item = self.parse_action_display.item(row)
              if item:
                  item.setForeground(color)


    def highlight_predicted_batch_queued(self, prediction_list: List[PredictionParsed]):
        """Highlights a batch of PredictionParsed items when their corresponding actions are queued."""
        if not hasattr(self, 'parse_action_display') or self.parse_action_display is None:
             print("Warning: parse_action_display not available for highlighting.")
             return

        for row in range(self.parse_action_display.count()):
             item = self.parse_action_display.item(row)
             if item:
                  item.setForeground(QColor("black"))

        for pred in prediction_list:
             self.highlight_parsed_prediction_item(pred, QColor("blue"))


    def clear_parsed_prediction_highlighting(self):
         """Clears highlighting from all parsed prediction items."""
         if not hasattr(self, 'parse_action_display') or self.parse_action_display is None:
             return
         for row in range(self.parse_action_display.count()):
              item = self.parse_action_display.item(row)
              if item:
                  item.setForeground(QColor("black"))


    @asyncClose
    async def closeEvent(self, event):
        """Handles window closing event, ensuring VNC disconnect and automaton stop."""
        print("[func:closeEvent] Closing application...")
        self.statusBar().showMessage("Closing")
        self.refresh_timer.stop()

        if hasattr(self.automaton, 'stop'):
             self.automaton.stop()
             await asyncio.sleep(0.1)
        else:
             print("Warning: Automaton does not have a 'stop' method.")


        if self.vnc is not None:
            try:
                await self.vnc.disconnect()
                print("[func:closeEvent] Disconnected from VNC.")
            except Exception as e:
                print(f"[func:closeEvent] Error during VNC disconnect: {e}")
            self.vnc = None

        print("[func:closeEvent] Application close event processed.")
        event.accept()
