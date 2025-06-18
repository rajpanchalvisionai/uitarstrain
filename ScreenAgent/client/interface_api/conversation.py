# --- START OF FILE interface_api/conversation.py ---

import dataclasses
import enum
from enum import auto, Enum
from typing import List, Tuple, Dict, Any, Optional
import re
import math # Kept math in case smart resize needed it, though not directly in prompt generation

# No longer importing Image, base64, BytesIO as old image handling is removed from Conversation

# --- VLM Specific Enums and Constants (from prompts_ts.txt context) ---
class UITarsLanguage(str, enum.Enum): # Inherit from str for easier comparison
    ZH = "zh"
    EN = "en"

class UITarsUseCase(str, enum.Enum): # Inherit from str
    NORMAL = "normal"
    POKI = "poki"

# Define UI-tars VLM versions relevant to prompt generation
class UITarsPromptVersion(str, enum.Enum): # Inherit from str
    V1_0 = "V1_0"
    V1_5 = "V1_5"
    POKI = "POKI"
    DOUBAO_15_15B = "DOUBAO_15_15B"
    DOUBAO_15_20B = "DOUBAO_15_20B"

# --- Prompt Text Templates (Translated from prompts_ts.txt) ---

_THOUGHT_EXAMPLES_ZH = """- Example1. Thought: 第一行、第三列出现了一个数字2；第二列原有数字4与第四列新出现的数字4合并后变为8。注意观察第二列数字8与左边数字8的颜色比较浅一点，数字2的颜色看起来没有数字8的深。我猜测不同的颜色深的程度代表数值的大小，颜色较深的代表数值较大。这不，为了验证这个，我继续按下向左键让这两个8合并成为更大的数。
- Example2. Thought: 真好！第一行第三列的数字2向左移动了两格合并到了第一行第一列，并且颜色比原先数字8的颜色深了许多。证明我的猜想没错，确实是这样！所以只有同样颜色深浅的数字才能够进行合并，而合并后的数字将变为原来数字的二倍并且颜色深度较深。而且!第一行第三列的2向左移动了两格，但是并没有和第一行第一列的2进行合并！由此可得，只有相同连续的格子才能够进行数字的合并。我按下向下键，16可以一步步进行合并得到2048，但是过程可能有些难。像我这样所做的操作并不是一步一步合并得到的。我这样做是为了更好的后续进行合并，得到更加大的数。
- Example3. Thought: 又重新再来了。刚才的下键并没有起到什么作用。新格子还是刷到了第三行第四列的位置，表明下键此时并没有什么太大作用，我猜测是不是特定的布局无法支持一些方位的操作，为了验证，我得多尝试一些方位，我按下左键看看。
- Example4. Thought: 哦，我知道了，同样的位置选择了同样的操作时不会发生改变的。除非是选择不同的方位！点击向上键以后，3、4行的数字都向上移动了一格，而它们原来所在的位置都被刷新出来了新数字，分别是4和2。同样，第三行第四列的数字2没有发生移动也刷新了新格子。明白了这一切后，我操作向左键试试看。
- Example5. Thought: 经过我不懈的努力，在我的仔细观察选中的策略下，我成功地获得了胜利。这验证了我之前的猜想，移动按键只有我的头部移动到含数字的区域才会改变移动按键，蛇的身体移动到含数字的区域并不会影响移动按键。
- Example6. Thought: 小蛇还是没动，我再次选择让它向右一步，希望这一次能成功移动，并且我猜测移动的间隔应该是蛇的长度，按动的次数也应该是蛇的长度。我或许需要将它记录下来，如果按一次它因为前方有障碍而动不了，但前方需要移动的话，需要按两次或者以上，按照蛇的长度来计算要按几次。
- Example7. Thought: 我觉得我的猜测是正确的，小蛇的移动是根据手部的长度是否能达成这一条件进行前进，这对我之后的操作提供了很多帮助，也是游戏的通性。不过现在小蛇离苹果拿走只有一个格子，太过去了，所以后面还需要。再次往前走我们应该先走出道这个限制然后来到中间这个地方然后我们应该是绕一圈然后把这两道门选择开阔住然后使得这样才能让这个墙消失。那么我可以现在向左，尝试不触碰障碍迈进，这似乎能改变小蛇的操作，使其改变路数。
- Example8. Thought: 我观察到在出口管道里面，红苹果的前方还有一个阻挡物。那个阻挡物是一张带有浅褐和深褐色的老鼠皮，看起来随着红苹果的自然移动，它也在向着出口移动，但是对比旁边的方块框架显得很慢。目前这些都是我猜测的，我要看看推动这个老鼠皮要多少的力道。就在这时我刚好要按向右了，现在我按住 “D”键。
- Example9. Thought: 太好了，我的做法是正确的，但是我发现激光点发射出来的激光这个时候并没有发光，看来我刚刚的猜测是不太全面的，还有新的知识，需要我再次了解一下激光的规则，回忆起来，刚刚似乎这个红色激光点发射出来的激光，别上是黄色，但上面的并没有什么波动，我需要新的条件，才能发现它的规律，将上一步的最后一格步骤拿出来，我发现刚刚不仅是激光颜色改变了，重要的是上面的箭头也改变了方向，也就是说激光点跟着太阳光一样，会有方向改变，这应该会是个关键消息，那我需要思考一下。
- Example10. Thought: 我继续观察发光装置箭头方向和角度，我猜测离发射装置近的那个白方块，只能被移动到与发射装置相邻的中上方蓝色方块位置，那么此时下方的白方块只能位于最右边一列蓝色方块中的其中一个位置并与位于一条直线上的左下方的黑色圆圈重合，我只能在右下角和正下方的两个蓝色方块中选择，似乎，看起来右下角的这个方块的位置更能满足与两列黑色圆圈的距离的重合，但是到底是否正确的呢，那么我一定要去验证了。
- Example11. Thought: 我们第一关是一个四边形,这个四边形内部的红绳是交织在一起的,我们根据以上经验如果要挪动一个毛线团的话,没有办法挪动任何一个上方有绳子限制的毛线团。所以从解题思路上我们可以打破这四边形的限制方向，那我们就可以挪动上方的毛线团。
"""

_THOUGHT_EXAMPLES_EN = """- Example1. Thought: A number 2 appears in the first row, third column; the number 4 in the second column combines with the newly appeared number 4 in the fourth column to become 8. Notice that the number 8 in the second column is slightly lighter than the number 8 on the left, and the number 2 appears less deep than the number 8. I suspect that the depth of different colors represents the magnitude of values, with darker colors representing larger values. To verify this, I continue to press the left key to merge these two 8s into a larger number.
- Example2. Thought: Great! The number 2 in the first row, third column moved two spaces left to the first row, first column, and its color became much deeper than the original number 8. This proves my guess was correct! Indeed, only numbers with the same color depth can be merged, and after merging, the number will become twice the original and have a deeper color depth. Moreover! The 2 from the first row, third column moved two spaces left but didn't merge with the 2 in the first row, first column! From this, we can conclude that only consecutive identical cells can merge numbers. I press the down key, 16 can gradually merge to get 2048, but the process might be difficult. Operations like mine aren't achieved by step-by-step merging. I do this to better facilitate subsequent merging and obtain larger numbers.
- Example3. Thought: Starting over again. The down key didn't have much effect. The new cell still appeared in the third row, fourth column, indicating the down key doesn't have much effect right now. I wonder if certain layouts don't support operations in some directions. To verify this, I need to try different directions, so I'll press the left key and see.
- Example4. Thought: Oh, I get it now, choosing the same operation in the same position won't cause any changes. Unless we choose different directions! After clicking the up key, the numbers in rows 3 and 4 all moved up one space, and their original positions were refreshed with new numbers, 4 and 2 respectively. Similarly, the number 2 in the third row, fourth column didn't move but also refreshed with a new cell. Now that I understand all this, I'll try operating the left key.
- Example5. Thought: Through my persistent efforts and careful observation of selected strategies, I successfully achieved victory. This verifies my previous hypothesis that movement keys only change when my head moves to an area containing numbers, while the snake's body moving to number-containing areas doesn't affect movement keys.
- Example6. Thought: The snake still hasn't moved. I choose to make it go right one more step, hoping this time it can move successfully. I suspect the movement interval should be the snake's length, and the number of button presses should also match the snake's length. I might need to record this - if pressing once doesn't work due to obstacles ahead, but forward movement is needed, it requires two or more presses, calculated based on the snake's length.
- Example7. Thought: I think my guess is correct - the snake's movement is based on whether the hand length can meet this condition to advance, which helps a lot with my later operations and is a common game mechanic. However, now the snake is only one square away from getting the apple, which is too close, so we still need more. Moving forward again, we should first get out of this restriction then come to the middle area, then we should go around in a circle and choose to open up these two doors, making the wall disappear this way. So I can now go left, trying to advance without touching obstacles, which seems to change the snake's operation, altering its path.
- Example8. Thought: I observe that inside the exit pipe, there's an obstacle in front of the red apple. That obstacle is a piece of mouse skin with light and dark brown colors, which seems to move toward the exit along with the red apple's natural movement, but appears slow compared to the block frame beside it. These are all my guesses for now, I want to see how much force it takes to push this mouse skin. Just as I'm about to press right, I now hold down the "D" key.
- Example9. Thought: Great, my approach was correct, but I notice the laser point's emitted laser isn't glowing right now. It seems my earlier guess wasn't comprehensive enough - there's new knowledge I need to understand about the laser rules. Thinking back, it seems the laser emitted from this red laser point was yellow on the side, but there wasn't any fluctuation above. I need new conditions to discover its pattern. Looking at the last grid step from before, I notice not only did the laser color change, but importantly, the arrow above also changed direction, meaning the laser point changes direction like sunlight. This should be crucial information, so I need to think about it.
- Example10. Thought: I continue observing the light device's arrow direction and angle. I guess the white block near the emission device can only be moved to the blue block position adjacent to the emission device in the middle top. Then the white block below can only be in one of the positions in the rightmost column of blue blocks and overlap with the black circle in the lower left that's in a straight line. I can only choose between the blue blocks in the bottom right corner and directly below. It seems the block position in the bottom right corner better satisfies the overlapping distance with the two columns of black circles, but is it really correct? I definitely need to verify this.
- Example11. Thought: Our first level is a quadrilateral, and the red ropes inside this quadrilateral are intertwined. Based on our previous experience, if we want to move a ball of yarn, there's no way to move any ball of yarn that has rope restrictions above it. So from a solution perspective, we can break the quadrilateral's restrictive direction, then we can move the upper ball of yarn.
"""

_SYSTEM_PROMPT_V1_0 = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(start_box='(x1,y1)')
left_double(start_box='(x1,y1)')
right_single(start_box='(x1,y1)')
drag(start_box='(x1,y1)', end_box='(x3,y3)')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
scroll(start_box='(x1,y1)', direction='down or up or right or left') # Show more information on the `direction` side.
wait() # Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.

## Note
- Use {language_str} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

_SYSTEM_PROMPT_V1_5 = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() # Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


## Note
- Use {language_str} in `Thought` part.
- {use_case_note}

## User Instruction
"""

_SYSTEM_PROMPT_POKI = """
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() # Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


## Note
- Use Chinese in `Thought` part.
- Compose a step-by-step approach in the `Thought` part, specifying your next action and its focus.

## User Instruction
"""

_SYSTEM_PROMPT_DOUBAO_15_15B = """
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='[x1, y1, x2, y2]')
left_double(start_box='[x1, y1, x2, y2]')
right_single(start_box='[x1, y1, x2, y2]')
drag(start_box='[x1, y1, x2, y2]', end_box='[x3, y3, x4, y4]')
hotkey(key='')
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
scroll(start_box='[x1, y1, x2, y2]', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language_str} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

_SYSTEM_PROMPT_DOUBAO_15_20B = """
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
press(key='ctrl') # Presses and holds down ONE key (e.g., ctrl). Use this action in combination with release(). You can perform other actions between press and release. For example, click elements while holding the ctrl key.
release(key='ctrl') # Releases the key previously pressed. All actions between press and release will execute with the key held down. Note: Ensure all keys you pressed are released by the end of the step.
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
wait() # Sleep for 5s and take a screenshot to check for any changes.
call_user() # Call the user when the task is unsolvable, or when you need the user's help. Then, user will see and answer your question in `user_resp`.
finished(content='xxx') # Submit the task with an report to the user. Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language_str} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- You may stumble upon new rules or features while playing the game or executing GUI tasks for the first time. Make sure to record them in your `Thought` and utilize them later.
- Your thought style should follow the style of thought Examples.
- You can provide multiple actions in one step, separated by "\\n\\n".
- Ensure all keys you pressed are released by the end of the step.

## Thought Examples
{thought_examples}

## Output Examples
Thought: {output_example_thought}
Action: click(point='<point>10 20</point>')

## User Instruction
"""

# --- Mapping for Prompt Generation ---
_SYSTEM_PROMPT_MAP = {
    UITarsPromptVersion.V1_0: _SYSTEM_PROMPT_V1_0,
    UITarsPromptVersion.V1_5: _SYSTEM_PROMPT_V1_5,
    UITarsPromptVersion.POKI: _SYSTEM_PROMPT_POKI,
    UITarsPromptVersion.DOUBAO_15_15B: _SYSTEM_PROMPT_DOUBAO_15_15B,
    UITarsPromptVersion.DOUBAO_15_20B: _SYSTEM_PROMPT_DOUBAO_15_20B,
}

_USE_CASE_NOTES = {
    (UITarsUseCase.NORMAL, UITarsLanguage.EN): 'Generate a well-defined and practical strategy in the `Thought` section, summarizing your next move and its objective.',
    (UITarsUseCase.POKI, UITarsLanguage.EN): 'Compose a step-by-step approach in the `Thought` part, specifying your next action and its focus.',
    (UITarsUseCase.NORMAL, UITarsLanguage.ZH): 'Generate a well-defined and practical strategy in the `Thought` section, summarizing your next move and its objective.', # Assuming ZH notes map similarly
    (UITarsUseCase.POKI, UITarsLanguage.ZH): 'Compose a step-by-step approach in the `Thought` part, specifying your next action and its focus.',
}

_THOUGHT_EXAMPLES_MAP = {
    UITarsLanguage.ZH: _THOUGHT_EXAMPLES_ZH,
    UITarsLanguage.EN: _THOUGHT_EXAMPLES_EN,
}


def _get_system_prompt_text(
    prompt_version: UITarsPromptVersion,
    language: UITarsLanguage,
    use_case: UITarsUseCase
) -> str:
    """Generates the system prompt text based on version, language, and use case."""
    template = _SYSTEM_PROMPT_MAP.get(prompt_version)
    if template is None:
        print(f"Warning: Unknown UI-tars prompt version: {prompt_version}. Using default V1_0.")
        template = _SYSTEM_PROMPT_V1_0 # Fallback

    language_str = "Chinese" if language == UITarsLanguage.ZH else "English"

    placeholders: Dict[str, str] = {
        "language_str": language_str,
    }

    if prompt_version == UITarsPromptVersion.V1_5:
         use_case_note = _USE_CASE_NOTES.get((use_case, language), _USE_CASE_NOTES[(UITarsUseCase.NORMAL, UITarsLanguage.EN)]) # Fallback
         placeholders["use_case_note"] = use_case_note

    if prompt_version == UITarsPromptVersion.DOUBAO_15_20B:
        thought_examples = _THOUGHT_EXAMPLES_MAP.get(language, _THOUGHT_EXAMPLES_EN) # Fallback
        output_example_thought = '在这里输出你的中文思考，你的思考样式应该参考上面的Thought Examples...' if language == UITarsLanguage.ZH else 'Write your thoughts here in English, your thinking style should follow the Thought Examples above...'
        placeholders["thought_examples"] = thought_examples
        placeholders["output_example_thought"] = output_example_thought


    def replace_placeholder(match):
        placeholder_name = match.group(1)
        return placeholders.get(placeholder_name, match.group(0))

    prompt_text = re.sub(r'\{(\w+)\}', replace_placeholder, template)

    return prompt_text.strip()


# --- Simplified Conversation Class for VLM ONLY ---

@dataclasses.dataclass
class Conversation:
    """
    A class that keeps VLM conversation history and generates prompts
    in the format expected by UI-tars-like models.
    This version removes the old message structure and related methods.
    """
    # History for VLM prompt generation: stores previous raw model outputs
    # Each entry is the raw text output string from a completed model turn
    previous_model_outputs: List[str] = dataclasses.field(default_factory=list)

    # Parameters for VLM system prompt generation
    language: UITarsLanguage = UITarsLanguage.EN
    use_case: UITarsUseCase = UITarsUseCase.NORMAL
    model_ver_ts: UITarsPromptVersion = UITarsPromptVersion.DOUBAO_15_20B

    # Old attributes removed: system, roles, messages, offset, sep_style, sep, sep2, version, skip_next

    # --- Methods for VLM Prompt Generation ---

    def append_model_output(self, raw_output_text: str):
        """Appends the raw text output from a model turn to the history."""
        self.previous_model_outputs.append(raw_output_text.strip())

    def get_vlm_prompt(self, current_instruction: str) -> str:
        """
        Generates the full text prompt for the VLM based on history
        and the current user instruction.

        Args:
            current_instruction: The instruction text for the current turn.

        Returns:
            The complete prompt string.
        """
        # 1. Generate System Prompt
        prompt_parts = [_get_system_prompt_text(
            prompt_version=self.model_ver_ts,
            language=self.language,
            use_case=self.use_case,
        )]

        # 2. Append Previous Model Outputs (History)
        if self.previous_model_outputs:
            prompt_parts.append("\n\n")
            prompt_parts.extend(self.previous_model_outputs)

        # 3. Append User Instruction Marker and Current Instruction
        prompt_parts.append("\n\n## User Instruction")
        prompt_parts.append(current_instruction.strip())

        return "\n".join(prompt_parts).strip()

    # --- Removed Old Methods and Attributes ---
    # get_prompt, get_images, to_gradio_chatbot, copy, dict, from_dict (from the old structure) are removed.
    # If you need saving/loading, you'll need to implement a new dict/from_dict
    # based on the new attributes (previous_model_outputs, language, use_case, model_ver_ts).

    def copy(self):
        """Creates a deep copy of the conversation (new structure)."""
        return Conversation(
            previous_model_outputs=self.previous_model_outputs.copy(),
            language=self.language,
            use_case=self.use_case,
            model_ver_ts=self.model_ver_ts,
        )

    def dict(self):
        """Returns a dictionary representation (new structure)."""
        return {
            "previous_model_outputs": self.previous_model_outputs,
            "language": self.language.value,
            "use_case": self.use_case.value,
            "model_ver_ts": self.model_ver_ts.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates a Conversation instance from a dictionary (new structure)."""
        data_copy = data.copy()

        # Handle Enum conversion
        if 'language' in data_copy and isinstance(data_copy['language'], str):
             try:
                  data_copy['language'] = UITarsLanguage(data_copy['language'])
             except ValueError:
                  print(f"Warning: Unknown language '{data_copy['language']}' in dict. Using default.")
                  data_copy['language'] = UITarsLanguage.EN

        if 'use_case' in data_copy and isinstance(data_copy['use_case'], str):
             try:
                  data_copy['use_case'] = UITarsUseCase(data_copy['use_case'])
             except ValueError:
                  print(f"Warning: Unknown use_case '{data_copy['use_case']}' in dict. Using default.")
                  data_copy['use_case'] = UITarsUseCase.NORMAL

        if 'model_ver_ts' in data_copy and isinstance(data_copy['model_ver_ts'], str):
             try:
                  data_copy['model_ver_ts'] = UITarsPromptVersion(data_copy['model_ver_ts'])
             except ValueError:
                  print(f"Warning: Unknown model_ver_ts '{data_copy['model_ver_ts']}' in dict. Using default.")
                  data_copy['model_ver_ts'] = UITarsPromptVersion.DOUBAO_15_20B

        # Ensure all required fields for the dataclass are present, even if not in the dict
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data_copy.items() if k in valid_keys}

        # Provide defaults for missing fields if necessary (dataclass default_factory handles lists)
        # filtered_data.setdefault('previous_model_outputs', []) # Not needed with default_factory
        filtered_data.setdefault('language', UITarsLanguage.EN)
        filtered_data.setdefault('use_case', UITarsUseCase.NORMAL)
        filtered_data.setdefault('model_ver_ts', UITarsPromptVersion.DOUBAO_15_20B)


        return cls(**filtered_data)


# --- Consolidated Template Dictionary ---

# Remove old templates, only keep UI-tars specific ones
conv_templates = {
    # Add new UI-tars templates
    "uitars_v1_0_en_normal": Conversation(language=UITarsLanguage.EN, use_case=UITarsUseCase.NORMAL, model_ver_ts=UITarsPromptVersion.V1_0),
    "uitars_v1_5_en_normal": Conversation(language=UITarsLanguage.EN, use_case=UITarsUseCase.NORMAL, model_ver_ts=UITarsPromptVersion.V1_5),
    "uitars_v1_5_zh_poki": Conversation(language=UITarsLanguage.ZH, use_case=UITarsUseCase.POKI, model_ver_ts=UITarsPromptVersion.POKI),
    "uitars_doubao_15b_en_normal": Conversation(language=UITarsLanguage.EN, use_case=UITarsUseCase.NORMAL, model_ver_ts=UITarsPromptVersion.DOUBAO_15_15B),
    "uitars_doubao_20b_zh_normal": Conversation(language=UITarsLanguage.ZH, use_case=UITarsUseCase.NORMAL, model_ver_ts=UITarsPromptVersion.DOUBAO_15_20B),
    "uitars_doubao_20b_en_normal": Conversation(language=UITarsLanguage.EN, use_case=UITarsUseCase.NORMAL, model_ver_ts=UITarsPromptVersion.DOUBAO_15_20B),

    # Set a convenient default alias
    "uitars": Conversation(language=UITarsLanguage.EN, use_case=UITarsUseCase.NORMAL, model_ver_ts=UITarsPromptVersion.DOUBAO_15_20B), # Default alias
    "uitars_poki": Conversation(language=UITarsLanguage.ZH, use_case=UITarsUseCase.POKI, model_ver_ts=UITarsPromptVersion.POKI),
}

# Define a default conversation instance for convenience if needed, but using templates is better
# default_conversation = conv_templates["uitars"].copy() # Example default instance


if __name__ == "__main__":
    # Example Usage of the new VLM prompt generation
    print("--- Testing new VLM prompt generation ---")
    # Get a copy of a template configuration
    vlm_conv = conv_templates["uitars_doubao_20b_en_normal"].copy()

    # Simulate a turn: User Instruction -> Model Output
    initial_instruction = "Open the browser."
    # get_vlm_prompt is called BEFORE the model responds
    print("\n--- Prompt for Turn 1 ---")
    print(vlm_conv.get_vlm_prompt(current_instruction=initial_instruction))

    # Simulate receiving model output for Turn 1
    model_output_turn1 = """Thought: I will click the Firefox icon.
Action:
click(point='<point>100 200</point>')"""

    # Append the model output to history AFTER it's generated
    vlm_conv.append_model_output(model_output_turn1)

    # Simulate the next turn: new User Instruction
    next_instruction = "Navigate to example.com"

    print("\n--- Prompt for Turn 2 ---")
    print(vlm_conv.get_vlm_prompt(current_instruction=next_instruction))

    # Simulate receiving model output for Turn 2
    model_output_turn2 = """Thought: I need to type the URL in the address bar and press enter.
Action:
type(content='example.com\\n')""" # Note: raw output including escaped newline

    vlm_conv.append_model_output(model_output_turn2)

    # Simulate the next turn: new User Instruction
    final_instruction = "Scroll down the page."

    print("\n--- Prompt for Turn 3 ---")
    print(vlm_conv.get_vlm_prompt(current_instruction=final_instruction))

    # Example of saving/loading state (new structure)
    print("\n--- Saving and Loading ---")
    saved_state = vlm_conv.dict()
    import json
    print("Saved state:", json.dumps(saved_state, indent=2))

    loaded_conv = Conversation.from_dict(saved_state)
    print("Loaded conversation:", loaded_conv)
    print("Loaded previous model outputs:", loaded_conv.previous_model_outputs)
    # Test generating prompt from loaded state
    print("\n--- Prompt from Loaded Conversation (Turn 3) ---")
    print(loaded_conv.get_vlm_prompt(current_instruction=final_instruction))

