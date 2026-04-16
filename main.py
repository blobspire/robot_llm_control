from datetime import datetime
from operator import concat
import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from robot import Panda

import ollama

# Parameters
control_dt = 1. / 240.

# Create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                cameraYaw=40.0,
                                cameraPitch=-30.0, 
                                cameraTargetPosition=[0.5, 0.0, 0.2])

# Load the objects
urdfRootPath = pybullet_data.getDataPath()
plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
# Randomize cube position to ensure genuine reasoning
cube1 = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.6 + np.random.uniform(-0.05, 0.05), -0.2 + np.random.uniform(-0.05, 0.05), 0.05])
cube2 = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.4 + np.random.uniform(-0.05, 0.05), -0.3 + np.random.uniform(-0.05, 0.05), 0.05])

# Load the robot
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)

# Tool configuration
def move_to_pose(x: float, y: float, z: float, rotz: float) -> str:
    """
    Move robot end-effector to a Cartesian pose.

    Args:
        x: target x (meters)
        y: target y (meters)
        z: target z (meters)
        rotz: target rotation about z (radians). Use 0.0 for gripper facing down.
    Returns:
        A short status string.
    """

    # Could clamp bounds for safety

    for _ in range(800):
        panda.move_to_pose(ee_position=[x, y, z], ee_rotz=rotz, positionGain=0.01)
        p.stepSimulation()
        time.sleep(control_dt)

    return f"moved_to_pose({x:.3f}, {y:.3f}, {z:.3f}, {rotz:.3f})"

def open_gripper() -> str:
    """Open the robot gripper."""
    for _ in range(300):
        panda.open_gripper()
        p.stepSimulation()
        time.sleep(control_dt)
    return "gripper_opened"

def close_gripper() -> str:
    """Close the robot gripper."""
    for _ in range(300):
        panda.close_gripper()
        p.stepSimulation()
        time.sleep(control_dt)
    return "gripper_closed"

def done(reason: str = "") -> str:
    return f"done: {reason}"

available_functions = {
    "move_to_pose": move_to_pose,
    "open_gripper": open_gripper,
    "close_gripper": close_gripper,
    "done": done
}

def describe_state() -> str:
    s = panda.get_state()
    ee = s["ee-position"]
    return (
        f"End-Effector Position: ({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}); "
        f"Gripper State: {s['gripper-state']}; Gripper Width: {s['gripper-width']:.4f}; {state_description}"
    )

def describe_env() -> str:
    cube1_pos, _ = p.getBasePositionAndOrientation(cube1)
    cube2_pos, _ = p.getBasePositionAndOrientation(cube2)
    return (
        f"The center of Cube1 is at position: ({cube1_pos[0]:.4f}, {cube1_pos[1]:.4f}, {cube1_pos[2]:.4f}); "
        f"The center of Cube2 is at position: ({cube2_pos[0]:.4f}, {cube2_pos[1]:.4f}, {cube2_pos[2]:.4f}); "
        f"{env_description}"
    )

state_description = "" # Not needed yet
env_description = "Cube dimensions: 0.05m x 0.05m x 0.05m."
final_instruction_line = "Now, analyze the latest environment and robot states and determine if the task has been completed within a reasonable margin of error. If you think the task has been completed, call done(). If you think that the task is not complete yet, execute the best next move towards completing the task using the tools."

# System prompt for tool use
# Concat the following chunks together to enable final_instruction_line insertion without picking up the ToolCall braces
SYSTEM = ""
SYSTEM_1 = f"""
You control a simulation Panda robot arm with a gripper by calling the available tools.
You were given a task by the user. You have since executed a series of actions. Check if you have completed the user's task within a reasonable margin of error. If you think the task has been completed based on the current state of the robot and environment, call done(). If you think that the task is not complete yet, execute the best next move towards completing the task using the tools.

Available tools:
- move_to_pose(x, y, z, rotz): move the end-effector to a Cartesian pose
- open_gripper(): open the robot gripper
- close_gripper(): close the robot gripper
- done(): indicate task completion

Ensure a margin of safety to avoid unintended collisions. Grab the center of the block to maximize grasp success. 
Consider that when you move to pose, the parameters that you provide are the executed location of the end-effector. Therefore, the offset between the end-effector location and the cube center location must be considered to avoid attempting to move the cube into the space of another cube.

<Start of example>
State: End-Effector Position: (0.5545, 0.0002, 0.5195); Gripper State: open; Gripper Width: 0.0800; {state_description}
Env: The center of Cube1 is at position: (0.6268, -0.2203, 0.0250); The center of Cube2 is at position: (0.4229, -0.3189, 0.0250); {env_description}
Task: pick up cube1
{final_instruction_line}
"""
SYSTEM_2 = """
content: ''
tool_calls: ToolCall(function=Function(name='move_to_pose', arguments={'x': 0.6268, 'y': -0.2203, 'z': 0.025, 'rotz': 0}))
<End of example>
"""
SYSTEM = concat(SYSTEM_1, SYSTEM_2)

# Including few shot examples for picking up cubes and stacking cubes drastically improves success rate but defeats some of the purpose of the LLM reasoning.
# ToolCall(function=Function(name='move_to_pose', arguments={'x': 0.4312, 'y': -0.3368, 'z': 0.075, 'rotz': 0})), ToolCall(function=Function(name='open_gripper', arguments={})), ToolCall(function=Function(name='done', arguments={}))
# After picking up a cube, you should raise it at least 10cm above the table before moving to another location because each cube is 5cm tall and you want to avoid unwanted collisions.
# Therefore, the difference between the end-effector position and each of the cube sides at grasp-time must be considered to avoid attempting to move the cube into the space of another cube.

### 
# A potentially beneficial prompt technique could be somthing like:
# You were given the task XYZ. You have executed a series of action. Check if you completed the task. If so, call done(). If not, execute the best next move towards completing the task using the tools.
#
# Though right now, it has succesfully terminated after stacking two blocks. This prompt is just likely a bit more robust.
###

MODEL = "gpt-oss:20b"

# let the scene initialize
for i in range (200):
    p.stepSimulation()
    time.sleep(control_dt)

terminate = False

while not terminate: # Execute commands for robot
    now = datetime.now()
    print(f"\n=== {now} ===")
    user_task = input("Enter your task for the robot (or 'exit' to exit): ")
    if user_task.lower() == 'exit':
        terminate = True
        break

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Observations:\nState: {describe_state()}\nEnv: {describe_env()}\nTask: {user_task}\n{final_instruction_line}"}
    ]

    MAX_STEPS = 20
    for step in range(MAX_STEPS):
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=[move_to_pose, open_gripper, close_gripper, done],
            options={
                "temperature": 0.0,
            },
            keep_alive="30m",
        )

        content = (getattr(response.message, "content", "") or "").strip()
        tool_calls = getattr(response.message, "tool_calls", None) or []

        print("Thinking:", getattr(response.message, "thinking", None))
        print("Content :", repr(content))
        print("Tool calls:", tool_calls, "\n")

        messages.append(response.message)

        # If model does not call tools but does provide content, remind of tool use and force retry. It often details the plan but doesn't call tools. 
        if not tool_calls:
            messages.append({
                "role": "system",
                "content": (
                    "Invalid response. You must call tools to control the robot and complete the task. If the task is complete, call done()."
                )
            })
            messages.append({
                "role": "user",
                "content": (
                    f"Task: {user_task}\n"
                    f"State: {describe_state()}\n"
                    f"Env: {describe_env()}\n"
                    "Retry the best next move using the tools now.\n"
                )
            })
            continue

        stop_chain = False

        for call in tool_calls:
            fn_name = call.function.name
            fn_args = call.function.arguments or {}
            fn = available_functions.get(fn_name)

            if fn is None:
                messages.append({
                    "role": "tool",
                    "tool_name": fn_name,
                    "content": f"ERROR: unknown tool {fn_name}",
                })
                continue
            
            if fn_name == "done":
                stop_chain = True
                messages.append({
                "role": "tool",
                "tool_name": "done",
                "content": "ok",
                })
                break

            try:
                result = fn(**fn_args)
            except Exception as e:
                result = f"ERROR executing {fn_name}({fn_args}): {e}"

            messages.append({
                "role": "tool",
                "tool_name": fn_name,
                "content": str(result),
            })

        if stop_chain:
            break

        messages.append({
        "role": "user",
        "content": "Updated observations:\n"
                f"State: {describe_state()}\n"
                f"Env: {describe_env()}\n"
                f"Task: {user_task}\n"
                f"{final_instruction_line}"
        })


# Notes from class:

# Control robot with LLM commands via Ollama
# Ignore joints 8 and 9
# Joints 10 and 11 are the gripper fingers
# Revolute = radians and prismatic = meters
# We're using this Panda robot this year

# My notes:

# TODO I should switch the command to be an "original command" so that the model knows I issued it at first rather than after the 10th command. This will likely help it from not restarting the task when it has
# accomplished it. Also, add reasoning to "examine if further actions are required. does the current state accomplish the user's command? if so, your job is done so call done()." 
# Also, add some extra reasoning for unintended collisions. I could instruct to lift to certain height but if I add more cubes / other objects, it will need to be dynamic.

# TODO likely include description of env / size / what the model needs to internpret the state, directly with the state. Otherwise, might forget since way at beginning.
# TODO look at messages thread and see if the prompt location / subsequent makes sense or if it's causing the model to think it's a new task, causing the interference / start over.