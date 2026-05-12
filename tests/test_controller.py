import json
import unittest

from controller import SimulationController


class ControllerTestCase(unittest.TestCase):
    def make_controller(self, seed=0):
        return SimulationController(gui=False, sleep=False, seed=seed)

    def test_scene_object_geometry(self):
        controller = self.make_controller()
        try:
            state = json.loads(controller.observe())
            cube = state["objects"]["cube1"]
            self.assertEqual(cube["size"], [0.05, 0.05, 0.05])
            self.assertAlmostEqual(cube["top_z"] - cube["bottom_z"], 0.05, places=4)
            self.assertAlmostEqual(cube["bottom_z"], 0.0, delta=0.002)
        finally:
            controller.close()

    def test_gripper_width_alone_does_not_imply_held_object(self):
        controller = self.make_controller()
        try:
            state = json.loads(controller.observe())
            cube = state["objects"]["cube1"]
            controller.move_to_pose(cube["pose"][0], cube["pose"][1], cube["top_z"] + 0.08, 0.0)
            controller.close_gripper()
            after = json.loads(controller.observe())
            self.assertIsNone(after["held_object"])
            self.assertLess(after["robot_state"]["gripper_width"], 0.02)
        finally:
            controller.close()

    def test_high_grasp_fails_without_claiming_held_object(self):
        controller = self.make_controller()
        try:
            state = json.loads(controller.observe())
            cube = state["objects"]["cube1"]
            controller.open_gripper()
            controller.move_to_pose(cube["pose"][0], cube["pose"][1], 0.07, 0.0)
            controller.close_gripper()
            controller.move_to_pose(cube["pose"][0], cube["pose"][1], 0.16, 0.0)
            after = json.loads(controller.observe())
            self.assertIsNone(after["held_object"])
            self.assertLess(abs(after["objects"]["cube1"]["pose"][2] - cube["pose"][2]), 0.01)
        finally:
            controller.close()

    def test_generic_grasp_object_succeeds_across_randomized_positions(self):
        for seed in range(5):
            controller = self.make_controller(seed=seed)
            try:
                result = json.loads(controller.grasp_object("cube1"))
                self.assertTrue(result["success"], msg=result["warnings"])
                after = json.loads(controller.observe())
                self.assertEqual(after["held_object"], "cube1")
                self.assertGreater(after["objects"]["cube1"]["bottom_z"], 0.08)
            finally:
                controller.close()

    def test_generic_place_establishes_on_relation(self):
        controller = self.make_controller()
        try:
            grasp = json.loads(controller.grasp_object("cube1"))
            self.assertTrue(grasp["success"], msg=grasp["warnings"])
            state = json.loads(controller.observe())
            cube1 = state["objects"]["cube1"]
            cube2 = state["objects"]["cube2"]
            target_z = cube2["top_z"] + cube1["size"][2] / 2.0
            place = json.loads(
                controller.place_object_at_pose(
                    cube2["pose"][0],
                    cube2["pose"][1],
                    target_z,
                    0.0,
                )
            )
            self.assertTrue(place["success"], msg=place["warnings"])
            controller.wait_until_settled()
            after = json.loads(controller.observe())
            self.assertTrue(after["relations"]["on"]["cube1_to_cube2"])
        finally:
            controller.close()

    def test_right_of_uses_positive_x_axis(self):
        controller = self.make_controller(seed=12)
        try:
            state = json.loads(controller.observe())
            cube1 = state["objects"]["cube1"]
            cube2 = state["objects"]["cube2"]
            self.assertLess(cube2["pose"][0], cube1["pose"][0])
            self.assertFalse(state["relations"]["right_of"]["cube2_to_cube1"])

            result = json.loads(controller.grasp_object("cube2"))
            self.assertTrue(result["success"], msg=result["warnings"])
            target_x = cube1["pose"][0] + cube1["size"][0] + 0.02
            place = json.loads(controller.place_object_at_pose(target_x, cube1["pose"][1], cube2["pose"][2], 0.0))
            self.assertTrue(place["success"], msg=place["warnings"])
            after = json.loads(controller.observe())
            self.assertTrue(after["relations"]["right_of"]["cube2_to_cube1"])
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
