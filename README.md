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
- Sends a structured world state, contact summary, relation predicates, and user
  task to a local LLM.
- Exposes generic robot-control and manipulation tools to the LLM. These are
  reusable primitives, not task-specific shortcuts.
- Executes the selected tool call in simulation, verifies the observed outcome,
  reports object deltas and contacts, and asks the model for the next action.
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
├── controller.py       # Generic robot tools, safeguards, and action verification
├── world_model.py      # Structured world state, contacts, object deltas, relations
├── benchmark.py        # Optional LLM benchmark runner with JSONL transcripts
├── robot.py            # Panda robot wrapper around PyBullet controls and state access
├── test_ollama.py      # Ollama tool-calling test to confirm your Ollama is properly set up
├── tests/              # Headless PyBullet tests for grasp/place/world-model behavior
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

The core loop in `main.py` now follows a planner/executor/critic pattern:

1. Read a natural-language task from the user.
2. Build a structured world state containing robot state, object geometry,
   contacts, held-object estimate, relation predicates, and the last action
   result.
3. Ask the local LLM to choose the best next generic action.
4. Execute the tool in PyBullet with workspace and low-transfer safeguards.
5. Report the verified outcome, including object deltas, contacts, warnings, and
   whether a held object was actually detected.
6. Send updated observations back to the model.
7. Repeat until `done(reason)` is accepted by the predicate checker or
   `MAX_STEPS` is reached.

## Available Tools

The LLM can control the robot with these generic tools:

### `move_to_pose(x, y, z, rotz)`

Moves the end-effector to a Cartesian pose in world coordinates.

- `x`: target x position in meters
- `y`: target y position in meters
- `z`: target z position in meters
- `rotz`: end-effector rotation about the z axis in radians

The prompt tells the model to use `rotz = 0.0` for a downward-facing gripper.

### `move_relative(dx, dy, dz, drotz=0.0)`

Moves the end-effector by a relative offset from the current pose.

### `move_to_object_pose(object_name, dx, dy, dz, rotz=0.0)`

Moves the end-effector to a pose expressed as an offset from an object's center.
This keeps the LLM's geometric reasoning explicit while reducing arithmetic
errors.

### `open_gripper()`

Opens the gripper fingers.

### `close_gripper()`

Closes the gripper fingers.

The returned action result reports whether a held object was verified. Gripper
width alone is not treated as proof that a grasp succeeded.

### `wait_until_settled()`

Advances the simulation until objects are stable, then returns the latest world
state.

### `observe()`

Returns the latest structured world state without moving the robot.

### `approach_object(object_name, clearance=0.08)`

Moves above an object's center using a generic clearance above the object's top
surface.

### `grasp_object(object_name, grasp_axis="y")`

Runs a generic manipulation primitive: approach, descend to the object's center
height, close, lift, and verify that the object moved with the gripper.

This is not a task-specific shortcut. The LLM still decides which object to
grasp and what task-level relation it is trying to achieve.

### `place_object_at_pose(x, y, z, rotz=0.0)`

Places the currently held object at a desired object-center pose, releases it,
waits for physics to settle, and verifies the result.

### `done(reason="")`

Requests task completion. The controller accepts this only when inferred
predicate checks for known tabletop relations pass. For example, `stack cube1 on
cube2` requires the observed `on(cube1, cube2)` relation to be true.

The project intentionally does **not** expose task-specific tools such as
`stack_cubes()` or hard-coded stack examples.

## World Model

The LLM receives structured observations with:

- Object center pose, size, velocity, `top_z`, `bottom_z`, and stability.
- End-effector pose, gripper width, and a note that finger width is not proof of
  holding an object.
- Robot/object contacts, including whether contact is on a gripper finger.
- Estimated `held_object`, based on contact plus object pose relative to the
  grasp target.
- Relation predicates including `held`, `stable`, `near`, `touching`, `above`,
  `on`, and `right_of`.
- Last action result with success/failure, warnings, and per-object pose deltas.

`right_of(a, b)` uses the positive world x axis: object `a` must have a larger
x coordinate than object `b` by a safe margin.

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

The system prompt tells the model that it controls the robot through generic
tools and must act as both planner and critic. It emphasizes:

- Trust observed state over intended actions.
- Recover after failed grasps instead of assuming success.
- Use object geometry and relation predicates for task completion.
- Treat `place_object_at_pose` targets as held-object center poses.

If the model returns text without a tool call, the program appends a corrective
message and retries:

```text
Invalid response. You must call one generic tool. If the task is complete,
call done(reason).
```

## Tests and Benchmarks

Run the headless PyBullet tests:

```bash
python -m unittest tests.test_controller
```

These tests verify object geometry, relation predicates, failed high grasps,
held-object estimation, generic grasping, and generic placement.

Run optional LLM benchmarks and save JSONL transcripts:

```bash
python benchmark.py --runs 20
```

Add `--gui` to watch benchmark runs in PyBullet. The benchmark tasks include
pick-up, relative movement, stacking, near-but-not-touching, and return-to-start
style tasks.

## Important Parameters

In `main.py` and `controller.py`:

```python
MAX_STEPS = 30
MODEL = "gpt-oss:20b"
CONTROL_DT = 1. / 240.
WORKSPACE_BOUNDS = {"x": (0.20, 0.85), "y": (-0.60, 0.35), "z": (0.015, 0.65)}
```

- `MAX_STEPS` limits how many model/tool iterations a single user task can run.
- `MODEL` selects the Ollama model.
- `CONTROL_DT` controls the simulation step timing.
- `WORKSPACE_BOUNDS` rejects unsafe or unreachable poses before execution.

The low-level movement and gripper commands still execute fixed simulation
steps internally. The generic skills combine these low-level movements with
verification:

```python
move_to_pose: 800 steps
open_gripper: 300 steps
close_gripper: 400 steps
grasp_object: approach + descend + close + lift + verify
place_object_at_pose: lift + transfer + lower + release + settle + verify
```

## Troubleshooting

### PyBullet GUI does not open

Make sure you are running the program in an environment with GUI access. The
program connects with:

```python
p.connect(p.GUI)
```

Headless environments can use `p.DIRECT` instead as this removes
the visualization.

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
The next major milestone will be transitioning from known cube coordinates to vision. This will add considerable complexity but will better represent real-world applications.
