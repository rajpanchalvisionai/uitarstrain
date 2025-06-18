import os
import uuid
from typing import Callable, List, Dict, Any, Optional

from transitions import Machine, State
from jinja2 import Template as JinjaTemplate # Ensure Jinja2 is installed (pip install Jinja2)
from PIL import Image

# --- Import necessary PyQt5 modules ---
from PyQt5.QtCore import QObject, pyqtSignal, QMetaObject, Qt, Q_ARG


# Assuming action.py contains PredictionParsed, Action, MouseAction, KeyboardAction, WaitAction,
# EvaluateSubTaskAction, PlanAction, and convert_prediction_to_actions
from action import *

# --- Import Conversation from the corrected path ---
from interface_api.conversation import Conversation


# Note: The concept of BaseState and prompt_template_name per state
# is largely based on the old multi-agent flow. With the single VLM,
# the prompt content comes from interface_api/conversation.py templates.
# Keeping BaseState structure for now but the core logic will be in Automaton.
class Template(JinjaTemplate):
    # Corrected __init__ to explicitly call JinjaTemplate.__init__
    def __init__(self, source, *args, **kwargs):
        # Explicitly call the parent JinjaTemplate.__init__ with 'source' and other args/kwargs
        try:
            JinjaTemplate.__init__(self, source, *args, **kwargs) # Pass source here
        except Exception as e:
             # This error message is from Jinja2's init, which expects specific arguments
             # when not initialized with a source string directly.
             # The warning "object.__init__() takes exactly one argument" suggests that
             # JinjaTemplate.__init__ is somehow defaulting to object.__init__ in this context.
             # This might happen if 'source' is not a string or None unexpectedly.
             # Given the warning, the BaseState initialization logic might be trying to
             # load empty or invalid files, leading to Template('') or similar, which might trigger this.
             # For now, logging the error and setting template to None seems the most robust way
             # to handle potentially missing or invalid template files without crashing.
             # Let's keep the original error logging in BaseState.
             raise e # Re-raise the exception to ensure it's caught by the BaseState logic

        self.source = source # Keep source attribute if needed

    def __str__(self):
        return self.source

    def __repr__(self):
        return f"Template({self.source!r})"

# Note: The BaseState class and its subclasses (Prepare, Planning, Acting, Evaluating, Finish)
# were designed for the old multi-agent architecture. With the single VLM,
# the state machine transitions will trigger methods on the Automaton instance
# that handle the VLM interaction and action execution.
# Keeping the classes for now but their 'before' methods will be replaced or called differently.
class BaseState:
    prompt_template_name = None

    def __init__(self, automaton):
        self.automaton = automaton
        if self.prompt_template_name is not None:
            template_path = os.path.join(automaton.prompt_tepmlate_dir, f"{self.prompt_template_name}_{automaton.operation_system}_{automaton.language}.txt")
            if os.path.exists(template_path):
                try:
                    # Use the corrected Template class
                    with open(template_path, "r", encoding="utf8") as f:
                        prompt_template_content = f.read()
                        self.prompt_template = Template(prompt_template_content)
                        # print(f"Loaded prompt template: {template_path}") # Confirm loading
                except Exception as e:
                    # The "object.__init__() takes exactly one argument" warning likely originates from here
                    print(f"Warning: Could not load prompt template {template_path}: {e}")
                    self.prompt_template = None # Set to None on error
            else:
                print(f"Warning: Prompt template file not found: {template_path}")
                self.prompt_template = None


    def before(self):
        pass


class Prepare(BaseState):
    prompt_template_name = None


class Planning(BaseState):
    prompt_template_name = "planner_agent"


class Acting(BaseState):
    prompt_template_name = "actor_agent"

    def __init__(self, automaton):
        super().__init__(automaton)


class Evaluating(BaseState):
    prompt_template_name = "evaluator_agent"


class Finish(BaseState):
    prompt_template_name = None


# --- Automaton Class (Updated for Single VLM) ---

class Automaton(QObject):
    state_changed = pyqtSignal(str)


    def __init__(self, config):
        super().__init__()

        self.prompt_tepmlate_dir = config.get('prompt_tepmlate_dir', '.')
        self.language = config.get('language', 'en')
        self.operation_system = config.get('operation_system', 'linux')
        self.auto_transitions = config.get('auto_transitions', True)
        self.auto_execute_actions = config.get('auto_execute_actions', True)

        self.vncwidget = None

        self.conversation: Optional[Conversation] = None
        self._current_instruction: str = ""
        self._current_prediction_batch_for_highlight: List[PredictionParsed] = []
        self._next_trigger_after_processing: Optional[str] = None # Store the trigger to fire after processing prediction

        self.task_prompt: str = ""
        self.sub_task_list: List[str] = []
        self.current_task_index: int = 0

        self.advice: Optional[str] = None
        self.base_info: Dict[str, Any] = {}

        # --- State Machine Definition (Updated for Single VLM Flow) ---
        self.states_list = [
            'IDLE',
            'REQUESTING_PREDICTION',
            'PROCESSING_PREDICTION',
            'EXECUTING_ACTIONS',
            'TASK_COMPLETE',
            'USER_NEEDED',
            'ERROR',
            'STOPPED',
        ]

        states_config = [
            State(name='IDLE', on_enter='_on_enter_IDLE'),
            State(name='REQUESTING_PREDICTION', on_enter='_on_enter_requesting_prediction'),
            State(name='PROCESSING_PREDICTION', on_enter='_on_enter_processing_prediction'), # Decision logic is here
            State(name='EXECUTING_ACTIONS', on_enter='_on_enter_executing_actions'),
            State(name='TASK_COMPLETE', on_enter='_on_enter_TASK_COMPLETE'),
            State(name='USER_NEEDED', on_enter='_on_enter_USER_NEEDED'),
            State(name='ERROR', on_enter='_on_enter_ERROR'),
            State(name='STOPPED', on_enter='_on_enter_STOPPED'),
        ]

        self.machine = Machine(model=self, states=states_config, initial='IDLE')

        # --- Define Transitions ---
        # Renamed the 'start' trigger to 'initiate_start' to avoid conflict with the method name
        self.machine.add_transition(trigger='initiate_start', source='IDLE', dest='REQUESTING_PREDICTION', before='_prepare_for_prediction_request')

        # Triggered by VNCWidget when VLM response is received and parsed
        # The parsed predictions are passed to the trigger method, which will pass them to the on_enter method
        self.machine.add_transition(trigger='prediction_received', source='REQUESTING_PREDICTION', dest='PROCESSING_PREDICTION') # Arguments passed to trigger go to on_enter


        # Triggered from PROCESSING_PREDICTION after deciding to queue actions
        self.machine.add_transition(trigger='queue_actions', source='PROCESSING_PREDICTION', dest='EXECUTING_ACTIONS', before='_queue_predicted_actions')

        # Triggered from PROCESSING_PREDICTION after deciding task is finished
        self.machine.add_transition(trigger='task_finished', source='PROCESSING_PREDICTION', dest='TASK_COMPLETE')

        # Triggered from PROCESSING_PREDICTION after deciding user is needed
        self.machine.add_transition(trigger='call_user', source='PROCESSING_PREDICTION', dest='USER_NEEDED')

        # Triggered from PROCESSING_PREDICTION if no executable actions or terminal action, or processing error
        self.machine.add_transition(trigger='no_actions_or_signal', source='PROCESSING_PREDICTION', dest='IDLE') # Back to idle for manual intervention/review

        # Triggered by VNCWidget when action batch execution is complete
        # If auto_transitions is enabled, go back to REQUESTING_PREDICTION for the next step
        self.machine.add_transition(trigger='actions_executed', source='EXECUTING_ACTIONS', dest='REQUESTING_PREDICTION', conditions=['_auto_transitions_enabled'], before='_prepare_for_next_prediction')
        # If auto_transitions is disabled, go back to IDLE
        self.machine.add_transition(trigger='actions_executed', source='EXECUTING_ACTIONS', dest='IDLE', unless=['_auto_transitions_enabled'])

        # Manual Transitions (from UI buttons or internal logic)
        self.machine.add_transition(trigger='reset', source='*', dest='IDLE', before='reset') # Reset to IDLE from any state
        self.machine.add_transition(trigger='stop', source='*', dest='STOPPED') # Trigger the 'stop' transition

        self.machine.add_transition(trigger='go_to_error', source='*', dest='ERROR') # Go to ERROR from any state

        # Add transitions for manual control from IDLE to specific states if needed
        # E.g., manual trigger to request prediction again:
        self.machine.add_transition(trigger='manual_request_prediction', source='IDLE', dest='REQUESTING_PREDICTION', before='_prepare_for_prediction_request')


        # --- Old State Class Instances (kept but less central) ---
        self.prepare_state_logic = Prepare(self)
        self.planning_state_logic = Planning(self)
        self.acting_state_logic = Acting(self)
        self.evaluating_state_logic = Evaluating(self)
        self.finish_state_logic = Finish(self)


        # --- Other Initialization ---
        self.before_action_screen = None
        self.action_list: List[Action] = []
        self.after_action_screen = None

        self.base_info: Dict[str, Any] = {
            "video_width": 640,
            "video_height": 480,
            "task_prompt": "",
        }


    # --- Condition methods for transitions ---
    def _auto_transitions_enabled(self):
        return self.auto_transitions

    # --- Methods triggered by state transitions (on_enter, before) ---

    # on_enter for IDLE state
    def _on_enter_IDLE(self):
        print("[Automaton] Entering state: IDLE")
        self.action_list = []
        self._current_prediction_batch_for_highlight = []
        self._next_trigger_after_processing = None # Clear pending trigger
        if self.vncwidget and hasattr(self.vncwidget, 'clear_parsed_prediction_highlighting'):
             self.vncwidget.clear_parsed_prediction_highlighting()


    # on_enter for REQUESTING_PREDICTION state
    def _on_enter_requesting_prediction(self):
        """Actions to perform when entering the REQUESTING_PREDICTION state."""
        print("[Automaton] Entering state: REQUESTING_PREDICTION")
        # The actual request is initiated by the 'before' method _prepare_for_prediction_request,
        # which is triggered by the transition leading to this state.
        # This method serves as confirmation or for additional actions upon entering the state.

    # before method for transitions going to REQUESTING_PREDICTION
    def _prepare_for_prediction_request(self):
        """Prepares for sending a VLM prediction request."""
        print(f"[Automaton:REQUESTING_PREDICTION] Preparing prediction request for instruction: {self._current_instruction[:50]}...")

        if self.vncwidget is None:
            print("[Automaton:REQUESTING_PREDICTION] Error: VNCWidget is not linked.")
            self.go_to_error()
            return

        try:
             # Use the VNCWidget's method to request the prediction
             # VNCWidget will call _handle_vlm_response_received, which in turn calls process_vlm_response_on_main_thread
             # process_vlm_response_on_main_thread calls the original recall_func passed here.
             # The original recall_func is what will fire the 'prediction_received' trigger.
             self.vncwidget.request_vlm_prediction(
                 current_instruction=self._current_instruction,
                 recall_func=partial(self.prediction_received) # Pass the trigger method itself as the callback
             )

        except Exception as e:
            print(f"[Automaton:REQUESTING_PREDICTION] Error requesting VLM prediction: {e}")
            self.go_to_error()


    # on_enter for PROCESSING_PREDICTION state
    # This method receives the parsed_predictions argument from the trigger
    def _on_enter_processing_prediction(self, parsed_predictions: List[PredictionParsed]):
         """Actions to perform when entering the PROCESSING_PREDICTION state."""
         print("[Automaton] Entering state: PROCESSING_PREDICTION")
         # Decision logic based on parsed_predictions
         print(f"[Automaton:PROCESSING_PREDICTION] Processing {len(parsed_predictions)} parsed predictions.")

         self._current_prediction_batch_for_highlight = parsed_predictions # Store for highlighting

         if not parsed_predictions:
             print("[Automaton:PROCESSING_PREDICTION] No valid predictions parsed.")
             self.no_actions_or_signal() # Decision: transition to IDLE

         else:
             first_pred = parsed_predictions[0]
             predicted_action_type = first_pred.action_type.lower()
             print(f"[Automaton:PROCESSING_PREDICTION] First predicted action type: {predicted_action_type}")

             if predicted_action_type == 'finished':
                 print("[Automaton:PROCESSING_PREDICTION] VLM signaled task finished.")
                 report = first_pred.action_inputs.get('content', '')
                 self._current_instruction = f"Task finished: {report}"
                 if self.vncwidget: self.vncwidget.set_current_instruction_display(self._current_instruction)
                 self.task_finished() # Decision: transition to TASK_COMPLETE

             elif predicted_action_type == 'call_user':
                 print("[Automaton:PROCESSING_PREDICTION] VLM signaled call user.")
                 message = first_pred.action_inputs.get('content', '')
                 self._current_instruction = f"User intervention needed: {message}"
                 if self.vncwidget: self.vncwidget.set_current_instruction_display(self._current_instruction)
                 self.call_user() # Decision: transition to USER_NEEDED

             else:
                 # Not a terminal action, assume it's a UI action (click, type, scroll, etc.)
                 # Convert the parsed predictions into executable Action objects
                 try:
                     if self.vncwidget and hasattr(self.vncwidget, 'video_width') and hasattr(self.vncwidget, 'video_height'):
                         screen_context = {'width': self.vncwidget.video_width, 'height': self.vncwidget.video_height}
                         executable_actions: List[Action] = []
                         for pred in parsed_predictions:
                              executable_actions.extend(convert_prediction_to_actions(pred, screen_context))

                         self.action_list = executable_actions # Store the list of executable actions

                         if executable_actions:
                              if self.auto_execute_actions:
                                   print("[Automaton:PROCESSING_PREDICTION] Auto-execute enabled. Decision: queue actions.")
                                   self.queue_actions() # Decision: transition to EXECUTING_ACTIONS
                              else:
                                   print("[Automaton:PROCESSING_PREDICTION] Auto-execute disabled. Actions parsed but not queued. Decision: no actions or signal.")
                                   self.no_actions_or_signal() # Decision: return to IDLE

                         else:
                              print("[Automaton:PROCESSING_PREDICTION] No executable actions generated from prediction. Decision: no actions or signal.")
                              self.no_actions_or_signal() # Decision: return to IDLE

                     else:
                         print("[Automaton:PROCESSING_PREDICTION] Error: VNCWidget screen dimensions not available for action conversion.")
                         self.go_to_error() # Decision: transition to ERROR

                 except Exception as e:
                     print(f"[Automaton:PROCESSING_PREDICTION] Error converting predictions to actions: {e}")
                     self.go_to_error() # Decision: transition to ERROR


    # Method called as 'before' method for the 'prediction_received' trigger
    # This method receives the parsed predictions but DOES NOT fire triggers.
    # Its logic has been moved to _on_enter_processing_prediction.
    # It now just exists as a placeholder callback signature.
    def handle_vlm_prediction_response(self, parsed_predictions: List[PredictionParsed]):
        """
        Callback method executed by VNCWidget when a VLM prediction response is received and parsed.
        This method is the target of the `recall_func` in request_vlm_prediction.
        It simply receives the predictions and the `prediction_received` trigger handles
        the transition to the PROCESSING_PREDICTION state, passing the predictions to
        the _on_enter_processing_prediction method.
        """
        print(f"[Automaton] Callback handle_vlm_prediction_response received {len(parsed_predictions)} predictions.")
        # No logic needed here anymore, the on_enter method handles it


    # on_enter for EXECUTING_ACTIONS state
    def _on_enter_executing_actions(self):
        print("[Automaton] Entering state: EXECUTING_ACTIONS")
        # The _queue_predicted_actions method is triggered as 'before' the transition.
        # This on_enter method is for any setup needed after queuing starts.


    # before method for transition to EXECUTING_ACTIONS
    def _queue_predicted_actions(self):
        """Queues the stored executable actions for execution by VNCWidget."""
        print(f"[Automaton:EXECUTING_ACTIONS] Queuing {len(self.action_list)} actions for execution.")

        if self.vncwidget is None:
            print("[Automaton:EXECUTING_ACTIONS] Error: VNCWidget is not linked.")
            self.go_to_error()
            return

        if not self.action_list:
            print("[Automaton:EXECUTING_ACTIONS] No actions to queue.")
            self.actions_executed(request_id="empty_batch") # Signal completion immediately
            return # Exit the method


        execution_request_id = uuid.uuid4().hex
        execution_recall_func = partial(self.action_execution_complete, execution_request_id)

        try:
             self.vncwidget.queue_executable_actions(
                 request_id=execution_request_id,
                 actions=self.action_list,
                 recall_func=execution_recall_func
             )

             if hasattr(self, '_current_prediction_batch_for_highlight') and self.vncwidget and hasattr(self.vncwidget, 'highlight_predicted_batch_queued'):
                  self.vncwidget.highlight_predicted_batch_queued(self._current_prediction_batch_for_highlight)


        except Exception as e:
            print(f"[Automaton:EXECUTING_ACTIONS] Error queuing actions via VNCWidget: {e}")
            self.go_to_error()


    # Callback method executed by VNCWidget when an action batch completes
    def action_execution_complete(self, request_id):
        """
        Callback method executed by VNCWidget when an action batch triggered by the automaton
        has finished executing. Triggers the next state transition based on auto_transitions.
        """
        print(f"[Automaton:EXECUTING_ACTIONS] Action batch {request_id} execution complete.")
        # The 'actions_executed' trigger has conditions based on auto_transitions.
        # Fire the trigger.
        self.actions_executed(request_id=request_id)


    # before method for transition back to REQUESTING_PREDICTION
    def _prepare_for_next_prediction(self):
        print("[Automaton] Preparing for next prediction after actions.")
        if self.vncwidget and hasattr(self.vncwidget, 'clear_parsed_prediction_highlighting'):
             self.vncwidget.clear_parsed_prediction_highlighting()


    # on_enter for TASK_COMPLETE state
    def _on_enter_TASK_COMPLETE(self):
         print("[Automaton] Entering state: TASK_COMPLETE")


    # on_enter for USER_NEEDED state
    def _on_enter_USER_NEEDED(self):
         print("[Automaton] Entering state: USER_NEEDED")


    # on_enter for ERROR state
    def _on_enter_ERROR(self):
         print("[Automaton] Entering state: ERROR")


    # on_enter for STOPPED state
    def _on_enter_STOPPED(self):
         print("[Automaton] Entering state: STOPPED")


    # --- Methods for UI Interaction (Called by VNCWidget) ---

    def link_to_vncwidget(self, vncwidget):
        self.vncwidget = vncwidget
        print("[Automaton] Linked to VNCWidget.")
        if hasattr(vncwidget, 'automaton_state_changed'):
            QMetaObject.invokeMethod(vncwidget, "automaton_state_changed", Qt.QueuedConnection, Q_ARG(str, self.state))
        else:
            print("Warning: VNCWidget does not have 'automaton_state_changed' method.")


    # This method is no longer used directly to *send* LLM requests,
    # but it might be called by internal state logic if you need to trigger a VLM request.
    # It should now use vncwidget.request_vlm_prediction.
    # Let's keep the name but update its implementation.
    # It takes prompt and image, but the prompt is now generated by Conversation.
    # Let's rename it to make its role clearer in the new flow.
    # Renaming to request_vlm_prediction_from_automaton.
    def request_vlm_prediction_from_automaton(self, instruction: str):
         print(f"[Automaton] Triggering VLM prediction request for instruction: {instruction[:50]}...")
         self._current_instruction = instruction
         if self.state == 'IDLE':
             self.manual_request_prediction()

         elif self.state == 'REQUESTING_PREDICTION':
              print("[Automaton] Already requesting prediction, ignoring redundant request.")
         elif self.state == 'EXECUTING_ACTIONS':
              print("[Automaton] Currently executing actions, cannot request prediction yet.")


    # Method called by VNCWidget when the 'Start Automaton' button is clicked.
    # Its signature was updated in controller_core.py.
    def start(self, task_prompt: str, video_width: int, video_height: int, conversation: Conversation):
        """
        Starts the automaton process for a given task.
        Called by VNCWidget when the 'Start Automaton' button is clicked.
        """
        print(f"[Automaton] Start method called with task: {task_prompt[:50]}...")
        # Store initial information and conversation object
        self.task_prompt = task_prompt
        self.base_info = {
            "video_width": video_width,
            "video_height": video_height,
            "task_prompt": task_prompt,
        }
        self.conversation = conversation
        self.advice = None

        # Set the initial instruction for the VLM to be the overall task prompt
        self._current_instruction = task_prompt
        # Update UI display for the current instruction (VNCWidget handles this)
        if self.vncwidget: self.vncwidget.set_current_instruction_display(self._current_instruction)


        # Reset other internal state variables
        self.sub_task_list = [] # Clear old plan/subtasks
        self.current_task_index = 0
        self.action_list = [] # Clear pending actions
        self._current_prediction_batch_for_highlight = [] # Clear highlighting batch

        # Clear old subtask display in UI
        if self.vncwidget and hasattr(self.vncwidget, 'update_sub_task_display'):
             self.vncwidget.update_sub_task_display(self.sub_task_list, self.current_task_index)


        # Ensure we are in the IDLE state to accept the 'initiate_start' trigger
        if self.state != 'IDLE':
            print(f"[Automaton] Warning: Start called while not in IDLE state ({self.state}). Resetting to IDLE.")
            self.reset() # Reset to IDLE state first

        # Corrected: Fire the state machine trigger 'initiate_start'
        self.initiate_start()


    # Method to manually set the automaton state (called by UI buttons)
    def set_state(self, state_name):
        """Manually sets the state of the automaton."""
        print(f"[Automaton] Manual set_state called: {state_name}")
        try:
            # Check if the state_name is a valid state in the machine
            if state_name in self.states_list:
                 self.machine.to(state_name) # Attempt to transition to the target state
            else:
                print(f"[Automaton] Warning: Attempted to set unknown state: {state_name}")
                # Optionally transition to an error state
                # self.go_to_error()

        except Exception as e:
            print(f"[Automaton] Error during manual state transition to {state_name}: {e}")
            # Transition to an error state on failure
            self.go_to_error()


    # Method to reset the automaton (called by reset button or internally)
    def reset(self):
        """Resets the automaton to its initial IDLE state."""
        print("[Automaton] Resetting automaton state and internal variables.")
        self.task_prompt = ""
        self._current_instruction = ""
        self.sub_task_list = []
        self.current_task_index = 0
        self.advice = None
        self.base_info = {
            "video_width": 640,
            "video_height": 480,
            "task_prompt": "",
        }
        self.action_list = []
        self._current_prediction_batch_for_highlight = []
        self.before_action_screen = None
        self.after_action_screen = None
        # self.conversation is reset by VNCWidget's reset()

        # Transition to the IDLE state if not already there
        if self.state != 'IDLE':
             self.to_IDLE() # Assuming a direct transition trigger 'to_IDLE' exists from '*'


    # Method to stop the automaton (called by UI close event)
    def stop(self):
        """Signals the automaton to stop."""
        print("[Automaton] Stop method called.")
        # Fire the 'stop' transition if not already stopped
        if self.state != 'STOPPED':
             self.stop()


    # Method to get the current state name (called by VNCWidget for status display)
    def get_state(self):
        """Returns the current state name."""
        return self.state


    # Method to update subtask display (called by VNCWidget)
    def update_sub_task_display(self, sub_task_list: List[str], now_sub_task_index: int):
        """
        Updates the automaton's internal subtask list and current index,
        and triggers the UI update.
        This method is likely called by VNCWidget after processing a PlanAction
        (if you still use planning separately) or potentially manually.
        """
        print(f"[Automaton] Updating subtasks internally: {len(sub_task_list)} tasks, current index {now_sub_task_index}")
        self.sub_task_list = sub_task_list
        self.current_task_index = now_sub_task_index
        if 0 <= self.current_task_index < len(self.sub_task_list):
            self._current_instruction = self.sub_task_list[self.current_task_index]
            if self.vncwidget: self.vncwidget.set_current_instruction_display(self._current_instruction)
        elif len(self.sub_task_list) > 0:
             if self.current_task_index >= len(self.sub_task_list):
                  self._current_instruction = f"Completed subtask {len(self.sub_task_list) - 1}. Task might be finished."
             else: # index is negative?
                  self._current_instruction = self.task_prompt
             if self.vncwidget: self.vncwidget.set_current_instruction_display(self._current_instruction)
        else:
             self._current_instruction = self.task_prompt
             if self.vncwidget: self.vncwidget.set_current_instruction_display(self._current_instruction)


        if self.vncwidget:
            self.vncwidget.update_sub_task_display(self.sub_task_list, self.current_task_index)


    # Method to manually set the current task index (called by UI double-click)
    def set_current_task_index(self, current_task_index: int):
        """
        Manually sets the current task index. Updates internal state and UI.
        This might be used in a hybrid flow where the user manually selects the next plan step.
        """
        print(f"[Automaton] Manual set_current_task_index called: {current_task_index}")
        if 0 <= current_task_index < len(self.sub_task_list):
            self.current_task_index = current_task_index
            self._current_instruction = self.sub_task_list[self.current_task_index]
            print(f"[Automaton] Current instruction updated to subtask {self.current_task_index}: {self._current_instruction[:50]}...")

            if self.vncwidget:
                 self.vncwidget.set_current_instruction_display(self._current_instruction)
                 self.vncwidget.update_sub_task_display(self.sub_task_list, self.current_task_index)

            if self.auto_transitions and self.state != 'REQUESTING_PREDICTION':
                 print("[Automaton] Auto-transitions enabled. Requesting prediction for manually selected subtask.")
                 self.manual_request_prediction()

        else:
            print(f"[Automaton] Warning: Manual set_current_task_index {current_task_index} is out of bounds for {len(self.sub_task_list)} tasks.")


    # Method to get the description of the current task/instruction (called by VNCWidget for prompt)
    def get_current_task_description(self) -> Optional[str]:
        return self._current_instruction if self._current_instruction else self.task_prompt


    # Method to set auto transitions (called by UI checkbox)
    def set_auto_transitions(self, auto_transitions: bool):
        self.auto_transitions = auto_transitions
        print(f"[Automaton] Auto transitions enabled: {self.auto_transitions}")


    # Method to set auto execute actions (called by UI checkbox)
    def set_auto_execute_actions(self, auto_execute_actions: bool):
        self.auto_execute_actions = auto_execute_actions
        print(f"[Automaton] Auto execute actions enabled: {self.auto_execute_actions}")


    # Method to set parsed predictions (called by VNCWidget after manual parsing from editor)
    def set_parsed_predictions(self, parsed_predictions: List[PredictionParsed]):
        print(f"[Automaton] Received {len(parsed_predictions)} manually parsed predictions from editor.")
        self._current_prediction_batch_for_highlight = parsed_predictions
        # When manually setting predictions, process them and decide next state (usually IDLE)
        # Call the processing logic directly, but don't rely on state machine transitions here
        # as manual input might bypass the normal flow.
        # This could be handled by a dedicated method for manual processing.
        # For now, let's call the main handler, but be aware manual flow might need refinement.
        self.handle_vlm_prediction_response(parsed_predictions) # This will set _next_trigger_after_processing

        # After processing manual predictions, the automaton should likely return to IDLE
        # if auto-execute was off, or EXECUTING_ACTIONS if auto-execute was on.
        # The handle_vlm_prediction_response method already sets the _next_trigger_after_processing.
        # We need to fire that trigger manually here if processing happened as a result of manual input.
        # However, handle_vlm_prediction_response is a 'before' method, called *by* a trigger.
        # A better approach for manual processing might be a separate method that
        # processes the predictions and then *decides* the next state/action sequence,
        # potentially bypassing the full state machine cycle for that one step.

        # Let's simplify: For manual parsing, just process the predictions and update the UI.
        # The user will then manually trigger execution via the button, which uses queue_executable_actions directly.
        # So, set_parsed_predictions only needs to update internal state and UI displays.
        # The call to handle_vlm_prediction_response above already updates internal state and UI.
        # So, this method is effectively just a wrapper.

        # The handle_vlm_prediction_response method sets _next_trigger_after_processing.
        # This stored trigger is meant to be fired by _on_enter_processing_prediction.
        # When manual parsing happens, we don't enter PROCESSING_PREDICTION state.
        # We need to manually trigger the decision-firing logic.
        # Let's call the logic from _on_enter_processing_prediction directly if auto-exec is enabled.
        if self.auto_execute_actions and self._next_trigger_after_processing:
             print("[Automaton] Manual parse and auto-exec enabled. Attempting to fire stored trigger.")
             trigger_name = self._next_trigger_after_processing
             self._next_trigger_after_processing = None # Clear the stored trigger
             try:
                 if hasattr(self, trigger_name) and callable(getattr(self, trigger_name)):
                    getattr(self, trigger_name)()
                 else:
                    print(f"[Automaton] Error: Stored trigger '{trigger_name}' is not a valid or callable method on the automaton.")
                    self.go_to_error()
             except Exception as e:
                   print(f"[Automaton] Error firing stored trigger '{trigger_name}' after manual parse: {e}")
                   self.go_to_error()
        else:
             print("[Automaton] Manual parse processed. Auto-exec disabled or no actions resulted.")
             # State remains IDLE or current state.


    def handle_critical_error(self, message: str):
        print(f"[Automaton] CRITICAL ERROR received from VNCWidget: {message}")
        self._current_instruction = f"Critical Error: {message}"
        if self.vncwidget: self.vncwidget.set_current_instruction_display(self._current_instruction)
        self.go_to_error()


    def handle_non_fatal_error(self, message: str):
        print(f"[Automaton] Non-fatal error received from VNCWidget: {message}")
        self._current_instruction = f"Non-fatal Error: {message}"
        if self.vncwidget: self.vncwidget.set_current_instruction_display(self._current_instruction)


    pass


    def machine_get_state_list(self):
         return [state.name for state in self.machine.states]