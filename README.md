# LLM-Controlled Robot Arm

This project controls a simulated Franka Panda robot arm with a local large
language model. The user gives the robot a task in natural language, the
program observes the current robot and environment state, and the LLM chooses
the next robot action by calling tools. The loop repeats until the model decides
the task is complete.

The simulation runs in PyBullet and uses Ollama for local LLM inference.

## What It Does

- Opens a PyBullet simulation with a Panda robot arm, a table, and two small
  cubes.
- Randomizes the cube positions slightly at startup so the model must reason
  from the current environment instead of memorizing fixed coordinates.
- Sends the robot state, environment state, and user task to a local LLM.
- Exposes robot-control functions as tools the LLM can call directly.
- Executes the selected tool call in simulation, updates the observations, and
  asks the model for the next action.
- Stops when the model calls `done()`, or after a maximum number of tool-use
  steps.

Example tasks:

```text
pick up cube1
move cube2 to the right of cube1
stack cube1 on cube2
open the gripper
move above cube1
```

## Project Structure

```text
.
├── main.py             # Simulation setup, LLM loop, tool definitions, user prompt loop
├── robot.py            # Panda robot wrapper around PyBullet controls and state access
├── test_ollama.py      # Ollama tool-calling test to confirm your Ollama is properly set up
└── README.md
```

## Requirements

- Python 3.10 or newer
- PyBullet
- NumPy
- Ollama
- A local Ollama model that supports tool calling

Currently, the best performing model is gpt-oss 20b. To try another model, change `MODEL` in `main.py`.

```python
MODEL = "gpt-oss:20b"
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install pybullet numpy ollama
```

Install and start Ollama if it is not already running, then pull the model used
by the program:

```bash
ollama pull gpt-oss:20b
```

You can verify that Ollama tool calling is working with:

```bash
python test_ollama.py
```

## Running the Robot Controller

Start the simulator and interactive command loop:

```bash
python main.py
```

A PyBullet GUI window should open with the Panda arm, table, and cubes. In the
terminal, enter a task for the robot:

```text
Enter your task for the robot (or 'exit' to exit): stack cube1 on cube2
```

For each task, the program sends the latest observations to the model and prints
the model's response, including any tool calls. The robot then executes those
tool calls in simulation.

Type `exit` at the prompt to close the command loop.

## Control Loop

The core loop in `main.py` works like this:

1. Read a natural-language task from the user.
2. Build an observation containing:
   - End-effector position
   - Gripper state and width
   - Cube positions
   - Cube dimensions
3. Ask the local LLM to choose the best next action.
4. Require the model to call one of the available tools.
5. Execute the tool in PyBullet.
6. Append the tool result to the conversation.
7. Send updated observations back to the model.
8. Repeat until the model calls `done()` or reaches `MAX_STEPS`.

## Available Tools

The LLM can control the robot with these tools:

### `move_to_pose(x, y, z, rotz)`

Moves the end-effector to a Cartesian pose in world coordinates.

- `x`: target x position in meters
- `y`: target y position in meters
- `z`: target z position in meters
- `rotz`: end-effector rotation about the z axis in radians

The prompt tells the model to use `rotz = 0.0` for a downward-facing gripper.

### `open_gripper()`

Opens the gripper fingers.

### `close_gripper()`

Closes the gripper fingers.

### `done(reason="")`

Signals that the task is complete.

## Simulation Details

The scene contains:

- A ground plane
- A table
- A fixed-base Franka Panda robot arm
- Two small cubes loaded from PyBullet's built-in URDF assets

Each cube is approximately:

```text
0.05 m x 0.05 m x 0.05 m
```

The cube centers are read from PyBullet at every step and included in the model
prompt. This allows the model to adapt to randomized cube positions.

## Prompting Strategy

The system prompt tells the model that it controls the robot through tools and
must decide whether the user task is complete. If the task is not complete, the
model should call the best next tool.

The prompt also includes an example tool call to ensure that the model has the proper tool-calling syntax.

If the model returns text without a tool call, the program appends a corrective
message and retries:

```text
Invalid response. You must call tools to control the robot and complete the task.
If the task is complete, call done().
```

## Important Parameters

In `main.py`:

```python
control_dt = 1. / 240.
MAX_STEPS = 20
MODEL = "gpt-oss:20b"
```

- `control_dt` controls the simulation step timing.
- `MAX_STEPS` limits how many model/tool iterations a single user task can run.
- `MODEL` selects the Ollama model.

The movement and gripper tools currently execute fixed numbers of simulation
steps:

```python
move_to_pose: 800 steps
open_gripper: 300 steps
close_gripper: 300 steps
```

## Troubleshooting

### PyBullet GUI does not open

Make sure you are running the program in an environment with GUI access. The
program connects with:

```python
p.connect(p.GUI)
```

Headless environments may need to use `p.DIRECT` instead, but that will remove
the interactive visualization.

### Ollama connection errors

Confirm that Ollama is installed and running:

```bash
ollama list
```

Then confirm the configured model exists locally:

```bash
ollama pull gpt-oss:20b
```

## Future Work
The next major milestone is transitioning from known cube coordinates to vision. This adds considerable complexity but will better represent real-world applications.
