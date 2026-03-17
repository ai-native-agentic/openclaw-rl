import base64
import json
import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from agents.utils.qwen_vl_utils import smart_resize
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.processing_utils import process_vision_info as slime_process_vision_info

logger = None


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


def process_image(image_bytes: bytes) -> str:
    """
    Process an image for Qwen VL models (thinking variant).
    Uses a tighter resize cap consistent with the thinking DUN agent.
    """
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        max_pixels=16 * 16 * 4 * 12800,
    )

    image = image.resize((resized_width, resized_height))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()

    return base64.b64encode(processed_bytes).decode("utf-8")


class Qwen3VLAgentLocal:
    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "qwen3-vl",
        max_steps: int = 100,
        max_image_history_length: int = 3,
        max_tokens: int = 32768,
        top_p: float = 0.9,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        coordinate_type: str = "relative",  # "relative" | "absolute" | "qwen25"
        example_result_dir: Optional[str] = None,
        add_thought_prefix: bool = False,
        **_unused_kwargs: Any,
    ):
        self.platform = platform
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_image_history_length = max(1, int(max_image_history_length))
        self.coordinate_type = coordinate_type
        self.example_result_dir = example_result_dir or os.getcwd()

        assert action_space in ["pyautogui"], "Invalid action space"
        assert observation_type in ["screenshot"], "Invalid observation type"
        assert coordinate_type in ["relative", "absolute", "qwen25"], "Invalid coordinate type"

        # 对齐 OpenCUAAgentLocal：显式保存 policy/reward 轨迹
        self.actions: List[str] = []
        self.responses: List[str] = []
        self.screenshots: List[str] = []  # 存 processed image base64（用于下一步回放给模型）

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.qwen3vl_agent_local")

        self.actions = []
        self.responses = []
        self.screenshots = []

    def _scale_scroll_for_windows(self, code: str, factor: int = 50) -> str:
        """与 OpenCUAAgentLocal 对齐：Windows 上 scroll scale 不同"""
        if self.platform.lower() != "windows":
            return code
        # 很粗暴但够用：pyautogui.scroll(<int>)
        # 只按右括号前的整数乘 factor
        import re

        pattern_pos = re.compile(r"(pyautogui\.scroll\()\s*([-+]?\d+)\s*\)")
        return pattern_pos.sub(lambda m: f"{m.group(1)}{int(m.group(2)) * factor})", code)

    def get_tool_spec(
        self,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        description_prompt_lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            (
                f"* The screen's resolution is {processed_width}x{processed_height}."
                if self.coordinate_type in ("absolute", "qwen25") and processed_width and processed_height
                else "* The screen's resolution is 1000x1000."
            ),
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
        description_prompt = "\n".join(description_prompt_lines)
        action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
        """
        return {
            "type": "function",
            "function": {
                "name_for_human": "computer_use",
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "properties": {
                        "action": {
                            "description": action_description_prompt,
                            "enum": [
                                "key",
                                "type",
                                "mouse_move",
                                "left_click",
                                "left_click_drag",
                                "right_click",
                                "middle_click",
                                "double_click",
                            "triple_click",
                                "scroll",
                            "hscroll",
                                "wait",
                                "terminate",
                            ],
                            "type": "string",
                        },
                        "keys": {"description": "Required only by `action=key`.", "type": "array"},
                        "text": {"description": "Required only by `action=type`.", "type": "string"},
                        "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                        "pixels": {"description": "The amount of scrolling.", "type": "number"},
                        "time": {"description": "The seconds to wait.", "type": "number"},
                        "status": {
                            "description": "The status of the task.",
                            "type": "string",
                            "enum": ["success", "failure"],
                        },
                    },
                    "required": ["action"],
                    "type": "object",
                },
                "args_format": "Format the arguments as a JSON object.",
            },
        }

    def get_system_prompt(
        self,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> str:
        tools_def = self.get_tool_spec(processed_width=processed_width, processed_height=processed_height)
        return (
            """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""
            + json.dumps(tools_def)
            + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""
        )

    def build_train_system_message(self) -> Dict[str, Any]:
        return {"role": "system", "content": self.get_system_prompt()}

    def build_instruction_prompt(self, instruction: str, actions_text: List[str]) -> str:
        prev = "\n".join([f"Step {i + 1}: {a}" for i, a in enumerate(actions_text)]) if actions_text else "None"
        return (
            "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Previous actions:\n{prev}"
        )

    @staticmethod
    def _extract_multimodal(messages: List[Dict[str, Any]], processor: Any) -> Dict[str, Any]:
        if not processor:
            return {}
        return slime_process_vision_info(messages, processor) or {}

    async def generate_with_sglang(
        self,
        *,
        args: Any,
        state: GenerateState,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        sampling_seed: int | None = None,
        tool_spec: Dict[str, Any] | None = None,
    ) -> Tuple[str, str]:
        tokenizer = state.tokenizer
        processor = state.processor
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=[tool_spec or self.get_tool_spec()],
        )
        current_sampling_params = dict(sampling_params)
        if sampling_seed is not None:
            current_sampling_params["sampling_seed"] = int(sampling_seed)

        payload: Dict[str, Any] = {"sampling_params": current_sampling_params, "return_logprob": True}
        image_data: List[str] = []
        if processor:
            multimodal_inputs = self._extract_multimodal(messages, processor)
            proc_kwargs: Dict[str, Any] = {"text": [prompt_text], "return_tensors": "pt", **multimodal_inputs}
            images = multimodal_inputs.get("images") or []
            if images:
                image_data = [encode_image_for_rollout_engine(img) for img in images]
            proc_out = processor(**proc_kwargs)
            input_ids = proc_out["input_ids"][0].tolist()
        else:
            input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        payload["input_ids"] = input_ids
        if image_data:
            payload["image_data"] = image_data

        output = await post(url, payload)
        finish_type = output["meta_info"]["finish_reason"]["type"]
        if "output_token_logprobs" in output["meta_info"]:
            output_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            response = tokenizer.decode(output_tokens)
        else:
            response = output.get("text", "")
        return response, finish_type

    def build_train_data(
        self,
        *,
        args: Any,
        state: GenerateState,
        train_messages: List[Dict[str, Any]],
        tool_spec: Dict[str, Any] | None = None,
    ) -> Tuple[List[int], List[int], Dict[str, Any] | None]:
        tokenizer = state.tokenizer
        processor = state.processor
        text_prompt = tokenizer.apply_chat_template(
            train_messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=[tool_spec or self.get_tool_spec()],
        )
        if processor:
            multimodal_inputs = self._extract_multimodal(train_messages, processor)
            kwargs: Dict[str, Any] = {"text": [text_prompt], "return_tensors": "pt", **multimodal_inputs}
            proc_out = processor(**kwargs)
            input_ids = proc_out["input_ids"][0].tolist()
            mm_train = {k: v for k, v in proc_out.items() if k not in ["input_ids", "attention_mask"]} or None
        else:
            input_ids = tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
            mm_train = None

        mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=getattr(args, "loss_mask_type", "qwen3"))
        _, loss_mask = mask_generator.get_loss_mask_with_multimodal_alignment(
            train_messages, input_ids, tools=[tool_spec or self.get_tool_spec()]
        )
        return input_ids, loss_mask, mm_train

    def record_policy_turn(self, *, action_text: str, response: str, screenshot_bytes: bytes) -> None:
        self.actions.append(action_text)
        self.responses.append(response)
        self.screenshots.append(process_image(screenshot_bytes))

    def build_policy_messages(self, instruction: str, obs: Dict) -> Dict[str, Any]:
        """
        Build policy messages exactly like OSWorld qwen3vl_agent_local.predict.
        """
        step_index = len(self.actions)
        screenshot_bytes: bytes = obs["screenshot"]

        img0 = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = img0.size

        processed_image_b64 = process_image(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_image_b64)))
        processed_width, processed_height = processed_img.size

        system_prompt = self.get_system_prompt(processed_width=processed_width, processed_height=processed_height)
        tool_spec = self.get_tool_spec(processed_width=processed_width, processed_height=processed_height)
        instruction_prompt = self.build_instruction_prompt(instruction=instruction, actions_text=self.actions)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        image_traj: List[str] = []

        start_i = max(0, step_index - self.max_image_history_length + 1)
        for i in range(start_i, step_index):
            img_url = f"data:image/png;base64,{self.screenshots[i]}"
            messages.append(
                {"role": "user", "content": [{"type": "image", "image": img_url}]}
            )
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": self.responses[i]}]}
            )
            image_traj.append(os.path.join(self.example_result_dir, f"step_{i}.png"))

        curr_img_url = f"data:image/png;base64,{processed_image_b64}"
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": curr_img_url},
                    {"type": "text", "text": instruction_prompt},
                ],
            }
        )
        image_traj.append(os.path.join(self.example_result_dir, f"step_{step_index}.png"))
        return {
            "messages": messages,
            "image_traj": image_traj,
            "step_index": step_index,
            "processed_image_b64": processed_image_b64,
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": processed_width,
            "processed_height": processed_height,
            "system_prompt": system_prompt,
            "tool_spec": tool_spec,
        }

    def parse_response(
        self,
        response: str,
        original_width: int,
        original_height: int,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        与 OpenCUAAgentLocal 对齐：返回 (action_text, [pyautogui_code...], other_info)
        """
        low_level_instruction = ""
        pyautogui_code: List[str] = []
        other: Dict[str, Any] = {"raw_response": response, "tool_calls": []}

        if response is None or not response.strip():
            return low_level_instruction, pyautogui_code, other

        def adjust_coordinates(x: float, y: float) -> Tuple[int, int]:
            # absolute / qwen25：按 processed->original 缩放；relative：按 0..999 网格缩放
            if self.coordinate_type in ("absolute", "qwen25"):
                if processed_width and processed_height:
                    x_scale = original_width / processed_width
                    y_scale = original_height / processed_height
                    return int(x * x_scale), int(y * y_scale)
                return int(x), int(y)

            # relative (0..999)
            x_scale = original_width / 999
            y_scale = original_height / 999
            return int(x * x_scale), int(y * y_scale)

        def process_tool_call(json_str: str) -> None:
            try:
                tool_call = json.loads(json_str)
                other["tool_calls"].append(tool_call)

                if tool_call.get("name") != "computer_use":
                    return
                args = tool_call.get("arguments", {})
                action = args.get("action")

                # --- mouse actions ---
                if action == "left_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        pyautogui_code.append(f"pyautogui.click({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.click()")

                elif action == "right_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        pyautogui_code.append(f"pyautogui.rightClick({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.rightClick()")

                elif action == "middle_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        pyautogui_code.append(f"pyautogui.middleClick({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.middleClick()")

                elif action == "double_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        pyautogui_code.append(f"pyautogui.doubleClick({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.doubleClick()")

                elif action == "mouse_move":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        pyautogui_code.append(f"pyautogui.moveTo({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.moveTo(0, 0)")

                elif action == "left_click_drag":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        duration = args.get("duration", 0.5)
                        pyautogui_code.append(f"pyautogui.dragTo({adj_x}, {adj_y}, duration={duration})")
                    else:
                        pyautogui_code.append("pyautogui.dragTo(0, 0)")

                # --- keyboard ---
                elif action == "type":
                    text = args.get("text", "")
                    # 简单转义
                    text = str(text).replace("\\", "\\\\").replace("'", "\\'")
                    pyautogui_code.append(f"pyautogui.typewrite('{text}')")

                elif action == "key":
                    keys = args.get("keys", [])
                    if not isinstance(keys, list):
                        keys = [keys]
                    keys = [str(k).strip() for k in keys if k is not None]
                    keys_str = ", ".join([f"'{k}'" for k in keys])
                    if len(keys) > 1:
                        pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
                    elif len(keys) == 1:
                        pyautogui_code.append(f"pyautogui.press({keys_str})")

                # --- scroll / wait / terminate ---
                elif action == "scroll":
                    pixels = args.get("pixels", 0)
                    pyautogui_code.append(f"pyautogui.scroll({int(pixels)})")

                elif action == "wait":
                    pyautogui_code.append("WAIT")

                elif action == "terminate":
                    status = (args.get("status") or "success").lower()
                    pyautogui_code.append("DONE" if status == "success" else "FAIL")

            except Exception as e:
                logger.error(f"Failed to parse tool call: {e}")

        # ---- parse response text ----
        lines = response.split("\n")
        inside_tool_call = False
        current_tool_call: List[str] = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            if line.lower().startswith("action:"):
                if not low_level_instruction:
                    low_level_instruction = line.split(":", 1)[-1].strip()
                continue

            if line.startswith("<tool_call>"):
                inside_tool_call = True
                continue

            if line.startswith("</tool_call>"):
                inside_tool_call = False
                if current_tool_call:
                    process_tool_call("\n".join(current_tool_call))
                    current_tool_call = []
                continue

            if inside_tool_call:
                current_tool_call.append(line)
                continue

            # 容忍一行 JSON
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if "name" in obj and "arguments" in obj:
                        process_tool_call(line)
                except Exception:
                    pass

        if current_tool_call:
            process_tool_call("\n".join(current_tool_call))

        if not low_level_instruction and pyautogui_code:
            low_level_instruction = "Execute the tool call"

        other["action"] = low_level_instruction
        other["code"] = pyautogui_code
        return low_level_instruction, pyautogui_code, other

