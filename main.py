from __future__ import annotations

from datetime import datetime
import json
from typing import Any

import ollama

from controller import SimulationController


MODEL = "gpt-oss:20b"
MAX_STEPS = 30
MODEL_RETRY_LIMIT = 3

active_controller: SimulationController | None = None
active_user_task = ""


def _controller() -> SimulationController:
    if active_controller is None:
        raise RuntimeError("Simulation controller is not initialized.")
    return active_controller


def move_to_pose(x: float, y: float, z: float, rotz: float) -> str:
    """
    Move the robot grasp target to a Cartesian world pose.

    Args:
        x: target x position in meters.
        y: target y position in meters.
        z: target z position in meters.
        rotz: end-effector rotation about world z in radians. Use 0.0 as the default downward grasp orientation.
    """
    return _controller().move_to_pose(x, y, z, rotz)


def move_relative(dx: float, dy: float, dz: float, drotz: float = 0.0) -> str:
    """
    Move the robot grasp target by a relative offset from its current pose.

    Args:
        dx: relative x motion in meters.
        dy: relative y motion in meters.
        dz: relative z motion in meters.
        drotz: relative rotation about world z in radians.
    """
    return _controller().move_relative(dx, dy, dz, drotz)


def move_to_object_pose(object_name: str, dx: float, dy: float, dz: float, rotz: float = 0.0) -> str:
    """
    Move the grasp target to an object-relative pose.

    Args:
        object_name: name of the reference object, such as cube1 or cube2.
        dx: target x offset from the object's center in meters.
        dy: target y offset from the object's center in meters.
        dz: target z offset from the object's center in meters.
        rotz: end-effector rotation about world z in radians.
    """
    return _controller().move_to_object_pose(object_name, dx, dy, dz, rotz)


def open_gripper() -> str:
    """Open the robot gripper and report the measured outcome."""
    return _controller().open_gripper()


def close_gripper() -> str:
    """Close the robot gripper and report contacts, width, and verified held object."""
    return _controller().close_gripper()


def wait_until_settled() -> str:
    """Advance physics until scene objects are stable, then return the structured world state."""
    return _controller().wait_until_settled()


def observe() -> str:
    """Return the latest structured robot, object, contact, and relation state."""
    return _controller().observe()


def approach_object(object_name: str, clearance: float = 0.08) -> str:
    """
    Move above an object's center using a generic clearance.

    Args:
        object_name: name of the target object.
        clearance: vertical distance above the object's top surface in meters.
    """
    return _controller().approach_object(object_name, clearance)


def grasp_object(object_name: str, grasp_axis: str = "y") -> str:
    """
    Perform a generic object grasp: approach, descend to the object center height, close, lift, and verify.

    Args:
        object_name: name of the object to grasp.
        grasp_axis: gripper closing axis, "y" by default or "x" for a 90 degree rotated grasp.
    """
    return _controller().grasp_object(object_name, grasp_axis)


def place_object_at_pose(x: float, y: float, z: float, rotz: float = 0.0) -> str:
    """
    Place the currently held object at a target object-center pose.

    Args:
        x: desired held-object center x position in meters.
        y: desired held-object center y position in meters.
        z: desired held-object center z position in meters.
        rotz: end-effector rotation about world z in radians.
    """
    return _controller().place_object_at_pose(x, y, z, rotz)


def done(reason: str = "") -> str:
    """
    Request task completion. The controller accepts this only when inferred task predicates are satisfied.

    Args:
        reason: short explanation of why the observed world state satisfies the user's task.
    """
    return _controller().done(active_user_task, reason)


TOOLS = [
    move_to_pose,
    move_relative,
    move_to_object_pose,
    open_gripper,
    close_gripper,
    wait_until_settled,
    observe,
    approach_object,
    grasp_object,
    place_object_at_pose,
    done,
]

AVAILABLE_FUNCTIONS = {tool.__name__: tool for tool in TOOLS}

FINAL_INSTRUCTION_LINE = (
    "Analyze the latest structured world state and decide whether the task is complete. "
    "If the task is complete, call done(reason). If not, call exactly the best next generic tool."
)

SYSTEM = """
You control a simulated Franka Panda robot arm with a gripper through generic tools.
Your role is planner and critic: choose useful actions, then update your beliefs from the observed outcome.

Core invariant:
- Never assume an action succeeded because it was commanded.
- Trust the latest world state, object deltas, contacts, held_object, and relation predicates over your prior plan.
- Gripper width is only finger opening; it is not proof of holding an object.
- If a grasp fails, recover: observe, open, reposition lower/centered, retry with a different generic action, or choose another safe approach.

Geometry facts:
- end_effector_position is the Panda grasp target between the fingers, in world coordinates.
- Object pose is the object center. Each object also reports size, top_z, and bottom_z.
- Spatial relation convention: right_of(a, b) means object a has a larger world x coordinate than object b by a safe margin.
- For a side grasp, approach above the object, then put the grasp target near the object center height before closing.
- place_object_at_pose(x, y, z) treats x/y/z as the desired center pose of the currently held object.
- Use object top_z + held_object half height when you want an object resting on top of another object.

Available generic tools:
- move_to_pose(x, y, z, rotz): precise Cartesian movement.
- move_relative(dx, dy, dz, drotz): small local corrections.
- move_to_object_pose(object_name, dx, dy, dz, rotz): object-relative movement.
- open_gripper(): open fingers and report outcome.
- close_gripper(): close fingers and report contacts/held-object evidence.
- wait_until_settled(): let physics settle and return world state.
- observe(): return current structured world state.
- approach_object(object_name, clearance): generic above-object approach.
- grasp_object(object_name, grasp_axis): generic approach-descend-close-lift-verify skill.
- place_object_at_pose(x, y, z, rotz): generic placement of the currently held object.
- done(reason): request completion; it can be rejected if observed predicates do not satisfy the task.

Do not rely on task-specific shortcuts. Reason from object names, dimensions, poses, contacts, and predicates.
When possible, write a short expected effect in content before a tool call.
"""


def _tool_calls_to_dict(tool_calls: list[Any]) -> list[dict[str, Any]]:
    calls = []
    for call in tool_calls:
        calls.append(
            {
                "name": call.function.name,
                "arguments": call.function.arguments or {},
            }
        )
    return calls


def _parse_done_result(result: str) -> bool:
    try:
        return bool(json.loads(result).get("accepted", False))
    except json.JSONDecodeError:
        return False


def run_llm_task(
    controller: SimulationController,
    user_task: str,
    max_steps: int = MAX_STEPS,
    print_trace: bool = True,
) -> dict[str, Any]:
    global active_controller, active_user_task
    active_controller = controller
    active_user_task = user_task

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Task: {user_task}\n"
                f"Initial structured world state:\n{controller.observe()}\n"
                f"{FINAL_INSTRUCTION_LINE}"
            ),
        },
    ]
    transcript: list[dict[str, Any]] = []
    completed = False

    for step in range(max_steps):
        response = None
        last_model_error = None
        for attempt in range(MODEL_RETRY_LIMIT):
            try:
                response = ollama.chat(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    options={"temperature": 0.0},
                    keep_alive="30m",
                )
                break
            except Exception as exc:
                last_model_error = exc
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "The previous model response failed before any tool could be executed: "
                            f"{exc}. Retry with exactly one valid tool call. Tool arguments must be "
                            "well-formed JSON with no trailing quote, comma, or extra text."
                        ),
                    }
                )

        if response is None:
            transcript.append(
                {
                    "step": step,
                    "model_error": repr(last_model_error),
                    "retry_limit": MODEL_RETRY_LIMIT,
                }
            )
            break

        content = (getattr(response.message, "content", "") or "").strip()
        thinking = getattr(response.message, "thinking", None)
        tool_calls = getattr(response.message, "tool_calls", None) or []
        call_dicts = _tool_calls_to_dict(tool_calls)

        if print_trace:
            print("Thinking:", thinking)
            print("Content :", repr(content))
            print("Tool calls:", tool_calls, "\n")

        transcript.append(
            {
                "step": step,
                "thinking": thinking,
                "content": content,
                "tool_calls": call_dicts,
            }
        )

        messages.append(response.message)

        if not tool_calls:
            reminder = (
                "Invalid response. You must call one generic tool. "
                "If the task is complete, call done(reason)."
            )
            messages.append({"role": "system", "content": reminder})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Task: {user_task}\n"
                        f"Latest structured world state:\n{controller.observe()}\n"
                        "Retry with one generic tool call now."
                    ),
                }
            )
            continue

        stop_chain = False
        for call in tool_calls:
            fn_name = call.function.name
            fn_args = call.function.arguments or {}
            fn = AVAILABLE_FUNCTIONS.get(fn_name)

            if fn is None:
                result = f"ERROR: unknown tool {fn_name}"
            else:
                try:
                    result = fn(**fn_args)
                except Exception as exc:
                    result = f"ERROR executing {fn_name}({fn_args}): {exc}"

            if print_trace:
                print("Tool result:", result, "\n")

            transcript.append(
                {
                    "step": step,
                    "tool_name": fn_name,
                    "tool_args": fn_args,
                    "tool_result": result,
                }
            )
            messages.append({"role": "tool", "tool_name": fn_name, "content": str(result)})

            if fn_name == "done":
                stop_chain = _parse_done_result(str(result))
                completed = stop_chain
                if not stop_chain:
                    messages.append(
                        {
                            "role": "system",
                            "content": "done() was rejected by the predicate checker. Continue from the latest observed state.",
                        }
                    )
                break

        if stop_chain:
            break

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Updated structured world state:\n{controller.observe()}\n"
                    f"Task: {user_task}\n"
                    f"{FINAL_INSTRUCTION_LINE}"
                ),
            }
        )

    return {
        "task": user_task,
        "completed": completed,
        "steps": len([item for item in transcript if "tool_calls" in item]),
        "final_state": controller.observe(),
        "transcript": transcript,
    }


def main() -> None:
    controller = SimulationController(gui=True, sleep=True)
    try:
        while True:
            now = datetime.now()
            print(f"\n=== {now} ===")
            user_task = input("Enter your task for the robot (or 'exit' to exit): ")
            if user_task.lower() == "exit":
                break
            result = run_llm_task(controller, user_task, print_trace=True)
            if not result["completed"]:
                print(f"Task stopped after {MAX_STEPS} steps without accepted completion.")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
