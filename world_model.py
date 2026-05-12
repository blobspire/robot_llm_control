from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any

import pybullet as p


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass
class SceneObject:
    name: str
    body_id: int
    size: Vector3
    pose: Vector3
    orientation: Quaternion
    velocity: Vector3
    angular_velocity: Vector3
    top_z: float
    bottom_z: float
    stable: bool


@dataclass
class WorldState:
    robot_state: dict[str, Any]
    objects: dict[str, SceneObject]
    contacts: list[dict[str, Any]]
    held_object: str | None
    relations: dict[str, Any]
    last_action_result: dict[str, Any] | None = None


@dataclass
class ActionResult:
    name: str
    args: dict[str, Any]
    success: bool
    before_state: dict[str, Any]
    after_state: dict[str, Any]
    object_deltas: dict[str, dict[str, float]]
    contacts: list[dict[str, Any]]
    warnings: list[str]
    message: str = ""


def vector_distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def xy_distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


class WorldModel:
    def __init__(
        self,
        robot,
        objects: dict[str, int],
        object_sizes: dict[str, Vector3],
        table_top_z: float = 0.0,
    ):
        self.robot = robot
        self.objects = objects
        self.object_sizes = object_sizes
        self.table_top_z = table_top_z
        self.last_action_result: dict[str, Any] | None = None

    def snapshot(self, include_last_action: bool = True) -> WorldState:
        objects = {name: self.scene_object(name) for name in self.objects}
        robot_state = self.robot_state()
        contacts = self.contact_summary()
        held_object = self.estimate_held_object(robot_state, objects, contacts)
        relations = self.compute_relations(objects, held_object)
        return WorldState(
            robot_state=robot_state,
            objects=objects,
            contacts=contacts,
            held_object=held_object,
            relations=relations,
            last_action_result=self.last_action_result if include_last_action else None,
        )

    def snapshot_dict(self, include_last_action: bool = True) -> dict[str, Any]:
        return asdict(self.snapshot(include_last_action=include_last_action))

    def snapshot_json(self, include_last_action: bool = True) -> str:
        return json_dumps(self.snapshot_dict(include_last_action=include_last_action))

    def scene_object(self, name: str) -> SceneObject:
        body_id = self.objects[name]
        size = self.object_sizes[name]
        pos, orn = p.getBasePositionAndOrientation(body_id)
        velocity, angular_velocity = p.getBaseVelocity(body_id)
        linear_speed = math.sqrt(sum(v * v for v in velocity))
        angular_speed = math.sqrt(sum(v * v for v in angular_velocity))
        return SceneObject(
            name=name,
            body_id=body_id,
            size=tuple(float(v) for v in size),
            pose=tuple(float(v) for v in pos),
            orientation=tuple(float(v) for v in orn),
            velocity=tuple(float(v) for v in velocity),
            angular_velocity=tuple(float(v) for v in angular_velocity),
            top_z=float(pos[2] + size[2] / 2.0),
            bottom_z=float(pos[2] - size[2] / 2.0),
            stable=linear_speed < 0.02 and angular_speed < 0.08,
        )

    def robot_state(self) -> dict[str, Any]:
        state = self.robot.get_state()
        ee = state["ee-position"]
        euler = state["ee-euler"]
        return {
            "end_effector_position": tuple(float(v) for v in ee),
            "end_effector_euler": tuple(float(v) for v in euler),
            "gripper_width": float(state["gripper-width"]),
            "finger_width_state": state["gripper-state"],
            "note": "finger_width_state describes finger opening only; it is not proof that an object is held.",
        }

    def contact_summary(self) -> list[dict[str, Any]]:
        contacts: list[dict[str, Any]] = []
        for object_name, body_id in self.objects.items():
            for contact in p.getContactPoints(bodyA=self.robot.panda, bodyB=body_id):
                contacts.append(
                    {
                        "object": object_name,
                        "robot_link": int(contact[3]),
                        "object_link": int(contact[4]),
                        "position_on_robot": tuple(float(v) for v in contact[5]),
                        "position_on_object": tuple(float(v) for v in contact[6]),
                        "contact_distance": float(contact[8]),
                        "normal_force": float(contact[9]),
                        "finger_contact": int(contact[3]) in self.robot.FINGER_LINK_INDICES,
                    }
                )
        return contacts

    def estimate_held_object(
        self,
        robot_state: dict[str, Any],
        objects: dict[str, SceneObject],
        contacts: list[dict[str, Any]],
    ) -> str | None:
        if robot_state["gripper_width"] > 0.065:
            return None

        ee = robot_state["end_effector_position"]
        candidates: list[tuple[float, str]] = []
        for name, obj in objects.items():
            object_contacts = [
                contact for contact in contacts
                if contact["object"] == name and contact["finger_contact"]
            ]
            near_grasp_target = vector_distance(ee, obj.pose) <= max(obj.size) * 1.25
            lifted_from_support = obj.bottom_z > self.table_top_z + 0.015
            if object_contacts and near_grasp_target and (lifted_from_support or obj.stable is False):
                candidates.append((vector_distance(ee, obj.pose), name))

        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def compute_relations(
        self,
        objects: dict[str, SceneObject],
        held_object: str | None,
    ) -> dict[str, Any]:
        relations: dict[str, Any] = {
            "held": {name: name == held_object for name in objects},
            "stable": {name: obj.stable for name, obj in objects.items()},
            "on": {},
            "above": {},
            "near": {},
            "touching": {},
            "right_of": {},
        }

        names = list(objects)
        for a_name in names:
            for b_name in names:
                if a_name == b_name:
                    continue
                a = objects[a_name]
                b = objects[b_name]
                key = f"{a_name}_to_{b_name}"
                pair_contacts = p.getContactPoints(bodyA=a.body_id, bodyB=b.body_id)
                xy_dist = xy_distance(a.pose, b.pose)
                xy_tol = max(min(a.size[0], b.size[0]) * 0.75, 0.035)
                height_tol = max(a.size[2], b.size[2]) * 0.35
                bottom_on_top = abs(a.bottom_z - b.top_z) <= height_tol
                relations["touching"][key] = len(pair_contacts) > 0
                relations["near"][key] = xy_dist <= 0.12
                relations["above"][key] = xy_dist <= xy_tol and a.pose[2] > b.pose[2]
                relations["on"][key] = (
                    xy_dist <= xy_tol
                    and bottom_on_top
                    and a.stable
                    and b.stable
                    and held_object != a_name
                )
                relations["right_of"][key] = a.pose[0] > b.pose[0] + max(a.size[0], b.size[0]) * 0.75

        return relations

    def object_deltas(
        self,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        deltas: dict[str, dict[str, float]] = {}
        for name, before_obj in before_state["objects"].items():
            after_obj = after_state["objects"][name]
            before_pose = tuple(before_obj["pose"])
            after_pose = tuple(after_obj["pose"])
            deltas[name] = {
                "distance": vector_distance(before_pose, after_pose),
                "dx": after_pose[0] - before_pose[0],
                "dy": after_pose[1] - before_pose[1],
                "dz": after_pose[2] - before_pose[2],
            }
        return deltas
