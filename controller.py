from __future__ import annotations

from dataclasses import asdict
import math
import os
import re
import time
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data

from robot import Panda
from world_model import ActionResult, WorldModel, json_dumps, vector_distance


CONTROL_DT = 1.0 / 240.0
CUBE_SIZE = (0.05, 0.05, 0.05)
TABLE_TOP_Z = 0.0
WORKSPACE_BOUNDS = {
    "x": (0.20, 0.85),
    "y": (-0.60, 0.35),
    "z": (0.015, 0.65),
}


class SimulationController:
    def __init__(self, gui: bool = True, sleep: bool | None = None, seed: int | None = None):
        self.gui = gui
        self.sleep = gui if sleep is None else sleep
        self.rng = np.random.default_rng(seed)
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=40.0,
                cameraPitch=-30.0,
                cameraTargetPosition=[0.5, 0.0, 0.2],
            )

        self.urdf_root_path = pybullet_data.getDataPath()
        self.plane = p.loadURDF(
            os.path.join(self.urdf_root_path, "plane.urdf"),
            basePosition=[0, 0, -0.625],
        )
        self.table = p.loadURDF(
            os.path.join(self.urdf_root_path, "table/table.urdf"),
            basePosition=[0.5, 0, -0.625],
        )

        self.objects = {
            "cube1": self._load_cube([0.6, -0.2, 0.05]),
            "cube2": self._load_cube([0.4, -0.3, 0.05]),
        }
        self.object_sizes = {name: CUBE_SIZE for name in self.objects}

        self.joint_start_positions = [
            0.0,
            0.0,
            0.0,
            -2 * np.pi / 4,
            0.0,
            np.pi / 2,
            np.pi / 4,
            0.0,
            0.0,
            0.04,
            0.04,
        ]
        self.panda = Panda(
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            jointStartPositions=self.joint_start_positions,
        )
        self.world = WorldModel(
            robot=self.panda,
            objects=self.objects,
            object_sizes=self.object_sizes,
            table_top_z=TABLE_TOP_Z,
        )
        self._step(240)
        self.initial_state = self.world.snapshot_dict(include_last_action=False)

    def close(self) -> None:
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)

    def _load_cube(self, base: list[float]) -> int:
        randomized = [
            base[0] + float(self.rng.uniform(-0.05, 0.05)),
            base[1] + float(self.rng.uniform(-0.05, 0.05)),
            base[2],
        ]
        return p.loadURDF(
            os.path.join(self.urdf_root_path, "cube_small.urdf"),
            basePosition=randomized,
        )

    def _step(self, steps: int) -> None:
        for _ in range(steps):
            p.stepSimulation()
            if self.sleep:
                time.sleep(CONTROL_DT)

    def _move_arm(self, x: float, y: float, z: float, rotz: float, steps: int = 800) -> None:
        for _ in range(steps):
            self.panda.move_to_pose(ee_position=[x, y, z], ee_rotz=rotz, positionGain=0.01)
            p.stepSimulation()
            if self.sleep:
                time.sleep(CONTROL_DT)

    def _open_gripper(self, steps: int = 300) -> None:
        for _ in range(steps):
            self.panda.open_gripper()
            p.stepSimulation()
            if self.sleep:
                time.sleep(CONTROL_DT)

    def _close_gripper(self, steps: int = 400) -> None:
        for _ in range(steps):
            self.panda.close_gripper()
            p.stepSimulation()
            if self.sleep:
                time.sleep(CONTROL_DT)

    def _validate_pose(
        self,
        x: float,
        y: float,
        z: float,
        before_state: dict[str, Any],
        allow_low_horizontal: bool = False,
    ) -> tuple[bool, list[str]]:
        warnings: list[str] = []
        for axis, value in (("x", x), ("y", y), ("z", z)):
            lower, upper = WORKSPACE_BOUNDS[axis]
            if value < lower or value > upper:
                warnings.append(
                    f"Rejected pose: {axis}={value:.3f} is outside workspace [{lower:.3f}, {upper:.3f}]."
                )
                return False, warnings

        current = before_state["robot_state"]["end_effector_position"]
        horizontal_motion = math.sqrt((x - current[0]) ** 2 + (y - current[1]) ** 2)
        max_top = max(obj["top_z"] for obj in before_state["objects"].values())
        safe_transfer_z = max_top + 0.06
        if (
            not allow_low_horizontal
            and horizontal_motion > 0.08
            and current[2] < safe_transfer_z
            and z < safe_transfer_z
        ):
            warnings.append(
                "Rejected low horizontal transfer near objects. Lift above "
                f"z={safe_transfer_z:.3f} before moving laterally."
            )
            return False, warnings

        return True, warnings

    def _finish_action(
        self,
        name: str,
        args: dict[str, Any],
        before_state: dict[str, Any],
        success: bool,
        warnings: list[str],
        message: str = "",
    ) -> str:
        after_state = self.world.snapshot_dict(include_last_action=False)
        deltas = self.world.object_deltas(before_state, after_state)
        result = ActionResult(
            name=name,
            args=args,
            success=success,
            before_state=before_state,
            after_state=after_state,
            object_deltas=deltas,
            contacts=after_state["contacts"],
            warnings=warnings,
            message=message,
        )
        result_dict = asdict(result)
        self.world.last_action_result = {
            "name": name,
            "args": args,
            "success": success,
            "object_deltas": deltas,
            "held_object_after": after_state["held_object"],
            "relations_after": after_state["relations"],
            "warnings": warnings,
            "message": message,
        }
        return json_dumps(result_dict)

    def observe(self) -> str:
        return self.world.snapshot_json(include_last_action=True)

    def wait_until_settled(self, max_steps: int = 720) -> str:
        for _ in range(max_steps):
            self._step(1)
            state = self.world.snapshot(include_last_action=True)
            if all(obj.stable for obj in state.objects.values()):
                break
        return self.observe()

    def move_to_pose(self, x: float, y: float, z: float, rotz: float) -> str:
        before = self.world.snapshot_dict(include_last_action=False)
        args = {"x": x, "y": y, "z": z, "rotz": rotz}
        valid, warnings = self._validate_pose(x, y, z, before)
        if not valid:
            return self._finish_action("move_to_pose", args, before, False, warnings)

        self._move_arm(x, y, z, rotz)
        after = self.world.snapshot_dict(include_last_action=False)
        ee = after["robot_state"]["end_effector_position"]
        err = vector_distance(tuple(ee), (x, y, z))
        success = err <= 0.018
        if not success:
            warnings.append(f"End effector stopped {err:.4f}m from target.")
        return self._finish_action(
            "move_to_pose",
            args,
            before,
            success,
            warnings,
            f"target_error={err:.4f}m",
        )

    def move_relative(self, dx: float, dy: float, dz: float, drotz: float = 0.0) -> str:
        state = self.world.snapshot_dict(include_last_action=False)
        ee = state["robot_state"]["end_effector_position"]
        rotz = state["robot_state"]["end_effector_euler"][2] + drotz
        return self.move_to_pose(ee[0] + dx, ee[1] + dy, ee[2] + dz, rotz)

    def move_to_object_pose(
        self,
        object_name: str,
        dx: float,
        dy: float,
        dz: float,
        rotz: float = 0.0,
    ) -> str:
        before = self.world.snapshot_dict(include_last_action=False)
        args = {"object_name": object_name, "dx": dx, "dy": dy, "dz": dz, "rotz": rotz}
        obj = before["objects"].get(object_name)
        if obj is None:
            return self._finish_action(
                "move_to_object_pose",
                args,
                before,
                False,
                [f"Unknown object '{object_name}'. Available objects: {sorted(self.objects)}."],
            )

        target = (
            obj["pose"][0] + dx,
            obj["pose"][1] + dy,
            obj["pose"][2] + dz,
        )
        valid, warnings = self._validate_pose(*target, before_state=before)
        if not valid:
            return self._finish_action("move_to_object_pose", args, before, False, warnings)

        self._move_arm(target[0], target[1], target[2], rotz)
        after = self.world.snapshot_dict(include_last_action=False)
        ee = after["robot_state"]["end_effector_position"]
        err = vector_distance(tuple(ee), target)
        success = err <= 0.018
        if not success:
            warnings.append(f"End effector stopped {err:.4f}m from object-relative target.")
        return self._finish_action(
            "move_to_object_pose",
            args,
            before,
            success,
            warnings,
            f"target={target}, target_error={err:.4f}m",
        )

    def open_gripper(self) -> str:
        before = self.world.snapshot_dict(include_last_action=False)
        self._open_gripper()
        after = self.world.snapshot_dict(include_last_action=False)
        width = after["robot_state"]["gripper_width"]
        success = width > 0.06
        warnings: list[str] = []
        if not success:
            warnings.append(f"Gripper width is {width:.4f}m after opening.")
        return self._finish_action(
            "open_gripper",
            {},
            before,
            success,
            warnings,
            f"gripper_width={width:.4f}m",
        )

    def close_gripper(self) -> str:
        before = self.world.snapshot_dict(include_last_action=False)
        self._close_gripper()
        after = self.world.snapshot_dict(include_last_action=False)
        width = after["robot_state"]["gripper_width"]
        warnings: list[str] = []
        if after["held_object"] is None:
            warnings.append(
                "No held object verified. Gripper width alone is not treated as grasp success."
            )
        return self._finish_action(
            "close_gripper",
            {},
            before,
            True,
            warnings,
            f"gripper_width={width:.4f}m, held_object={after['held_object']}",
        )

    def approach_object(self, object_name: str, clearance: float = 0.08) -> str:
        before = self.world.snapshot_dict(include_last_action=False)
        args = {"object_name": object_name, "clearance": clearance}
        obj = before["objects"].get(object_name)
        if obj is None:
            return self._finish_action(
                "approach_object",
                args,
                before,
                False,
                [f"Unknown object '{object_name}'. Available objects: {sorted(self.objects)}."],
            )
        target_z = obj["top_z"] + max(clearance, 0.03)
        valid, warnings = self._validate_pose(obj["pose"][0], obj["pose"][1], target_z, before)
        if not valid:
            return self._finish_action("approach_object", args, before, False, warnings)
        self._move_arm(obj["pose"][0], obj["pose"][1], target_z, 0.0)
        after = self.world.snapshot_dict(include_last_action=False)
        ee = after["robot_state"]["end_effector_position"]
        err = vector_distance(tuple(ee), (obj["pose"][0], obj["pose"][1], target_z))
        success = err <= 0.018
        if not success:
            warnings.append(f"End effector stopped {err:.4f}m from approach target.")
        return self._finish_action(
            "approach_object",
            args,
            before,
            success,
            warnings,
            f"approach_target_z={target_z:.4f}, target_error={err:.4f}m",
        )

    def grasp_object(self, object_name: str, grasp_axis: str = "y") -> str:
        before = self.world.snapshot_dict(include_last_action=False)
        args = {"object_name": object_name, "grasp_axis": grasp_axis}
        obj = before["objects"].get(object_name)
        if obj is None:
            return self._finish_action(
                "grasp_object",
                args,
                before,
                False,
                [f"Unknown object '{object_name}'. Available objects: {sorted(self.objects)}."],
            )

        rotz = 0.0 if grasp_axis.lower() == "y" else math.pi / 2.0
        approach_z = obj["top_z"] + 0.08
        grasp_z = obj["pose"][2]
        lift_z = max(obj["top_z"] + 0.12, grasp_z + 0.12, 0.14)
        warnings: list[str] = []
        for target in (
            (obj["pose"][0], obj["pose"][1], approach_z),
            (obj["pose"][0], obj["pose"][1], grasp_z),
            (obj["pose"][0], obj["pose"][1], lift_z),
        ):
            valid, pose_warnings = self._validate_pose(*target, before_state=before, allow_low_horizontal=True)
            warnings.extend(pose_warnings)
            if not valid:
                return self._finish_action("grasp_object", args, before, False, warnings)

        self._open_gripper(steps=180)
        self._move_arm(obj["pose"][0], obj["pose"][1], approach_z, rotz)
        self._move_arm(obj["pose"][0], obj["pose"][1], grasp_z, rotz)
        self._close_gripper(steps=500)
        self._move_arm(obj["pose"][0], obj["pose"][1], lift_z, rotz)
        self._step(120)

        after = self.world.snapshot_dict(include_last_action=False)
        held = after["held_object"]
        delta = self.world.object_deltas(before, after)[object_name]
        held_target = held == object_name
        lifted = delta["dz"] > 0.04
        near_ee = (
            vector_distance(
                tuple(after["robot_state"]["end_effector_position"]),
                tuple(after["objects"][object_name]["pose"]),
            )
            <= max(obj["size"]) * 1.5
        )
        success = held_target and lifted and near_ee
        if not success:
            warnings.append(
                "Grasp verification failed. The object must move upward with the gripper "
                "and remain near the grasp target after lift."
            )
        return self._finish_action(
            "grasp_object",
            args,
            before,
            success,
            warnings,
            f"held_object={held}, lifted_dz={delta['dz']:.4f}m",
        )

    def place_object_at_pose(self, x: float, y: float, z: float, rotz: float = 0.0) -> str:
        before = self.world.snapshot_dict(include_last_action=False)
        args = {"x": x, "y": y, "z": z, "rotz": rotz}
        held = before["held_object"]
        if held is None:
            return self._finish_action(
                "place_object_at_pose",
                args,
                before,
                False,
                ["Cannot place because no held object is verified."],
            )

        held_size = before["objects"][held]["size"]
        if z - held_size[2] / 2.0 < TABLE_TOP_Z - 0.005:
            return self._finish_action(
                "place_object_at_pose",
                args,
                before,
                False,
                ["Rejected place target: held object's bottom would be below the table."],
            )

        safe_z = max(
            before["robot_state"]["end_effector_position"][2],
            z + 0.10,
            max(obj["top_z"] for obj in before["objects"].values()) + 0.10,
        )
        warnings: list[str] = []
        for target in (
            (before["robot_state"]["end_effector_position"][0], before["robot_state"]["end_effector_position"][1], safe_z),
            (x, y, safe_z),
            (x, y, z),
            (x, y, safe_z),
        ):
            valid, pose_warnings = self._validate_pose(*target, before_state=before, allow_low_horizontal=True)
            warnings.extend(pose_warnings)
            if not valid:
                return self._finish_action("place_object_at_pose", args, before, False, warnings)

        self._move_arm(
            before["robot_state"]["end_effector_position"][0],
            before["robot_state"]["end_effector_position"][1],
            safe_z,
            rotz,
        )
        self._move_arm(x, y, safe_z, rotz)
        self._move_arm(x, y, z, rotz)
        self._open_gripper(steps=400)
        self._step(240)
        self._move_arm(x, y, safe_z, rotz)
        self._step(240)

        after = self.world.snapshot_dict(include_last_action=False)
        placed_pose = tuple(after["objects"][held]["pose"])
        target = (x, y, z)
        position_error = vector_distance(placed_pose, target)
        success = after["held_object"] is None and position_error <= 0.04
        if not success:
            warnings.append(
                f"Place verification failed: released={after['held_object'] is None}, "
                f"object_position_error={position_error:.4f}m."
            )
        return self._finish_action(
            "place_object_at_pose",
            args,
            before,
            success,
            warnings,
            f"placed_object={held}, object_position_error={position_error:.4f}m",
        )

    def done(self, user_task: str, reason: str = "") -> str:
        state = self.world.snapshot_dict(include_last_action=True)
        accepted, checks, warnings = self._validate_task_complete(user_task, state)
        return json_dumps(
            {
                "accepted": accepted,
                "reason": reason,
                "checks": checks,
                "warnings": warnings,
                "world_state": state,
            }
        )

    def _validate_task_complete(
        self,
        user_task: str,
        state: dict[str, Any],
    ) -> tuple[bool, list[dict[str, Any]], list[str]]:
        task = user_task.lower()
        names = list(self.objects)
        checks: list[dict[str, Any]] = []
        warnings: list[str] = []

        for source in names:
            for target in names:
                if source == target:
                    continue
                source_re = re.escape(source.lower())
                target_re = re.escape(target.lower())
                on_pattern = re.compile(
                    rf"(stack|put|place|move).*\b{source_re}\b.*\b(on|onto|on top of)\b.*\b{target_re}\b"
                )
                plain_on_pattern = re.compile(rf"\b{source_re}\b.*\b(on|onto|on top of)\b.*\b{target_re}\b")
                if on_pattern.search(task) or plain_on_pattern.search(task):
                    key = f"{source}_to_{target}"
                    passed = bool(state["relations"]["on"].get(key, False))
                    checks.append({"predicate": "on", "source": source, "target": target, "passed": passed})

                near_pattern = re.compile(rf"\b{source_re}\b.*\bnear\b.*\b{target_re}\b")
                if near_pattern.search(task):
                    key = f"{source}_to_{target}"
                    near = bool(state["relations"]["near"].get(key, False))
                    touching = bool(state["relations"]["touching"].get(key, False))
                    require_not_touching = "not touching" in task or "without touching" in task
                    passed = near and (not require_not_touching or not touching)
                    checks.append(
                        {
                            "predicate": "near",
                            "source": source,
                            "target": target,
                            "not_touching_required": require_not_touching,
                            "passed": passed,
                        }
                    )

                right_pattern = re.compile(rf"\b{source_re}\b.*\bright of\b.*\b{target_re}\b")
                if right_pattern.search(task):
                    key = f"{source}_to_{target}"
                    passed = bool(state["relations"]["right_of"].get(key, False))
                    checks.append({"predicate": "right_of", "source": source, "target": target, "passed": passed})

        for name in names:
            name_re = re.escape(name.lower())
            if re.search(rf"(pick up|grasp|hold).*\b{name_re}\b", task):
                passed = bool(state["relations"]["held"].get(name, False))
                checks.append({"predicate": "held", "source": name, "passed": passed})

        if not checks:
            warnings.append(
                "No deterministic completion predicate was inferred for this task. "
                "Accepting done based on the model's stated reason."
            )
            return True, checks, warnings

        return all(check["passed"] for check in checks), checks, warnings
