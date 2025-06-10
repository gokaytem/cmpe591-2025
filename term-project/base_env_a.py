import collections
import numpy as np
import mujoco
import mujoco_viewer
from env_shared import create_tabletop_scene

IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success'])


class BaseEnvA:
    def __init__(self, render_mode="gui", pick_pos=None, place_pos=None):
        self._gripper_idx = 6
        self._gripper_norm = 0.721
        self._render_mode = render_mode
        self.viewer = None
        self._n_joints = 7
        self._init_position = [-np.pi/2, -np.pi/2, np.pi/2, -2.07, 0, 0, 0]
        self._joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
            "ur5e/robotiq_2f85/right_driver_joint"
        ]
        self.pick_pos = pick_pos
        self.place_pos = place_pos
        self.reset()
        self._joint_qpos_idxs = [self.model.joint(x).qposadr for x in self._joint_names]
        self._ee_site = "ur5e/robotiq_2f85/gripper_site"

    def reset(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "data"):
            del self.data
        if self.viewer is not None:
            if self._render_mode == "offscreen":
                del self.viewer
            else:
                self.viewer.close()

        scene = self._create_scene()
        xml_string = scene.to_xml_string()
        assets = scene.get_assets()
        self.model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
        self.data = mujoco.MjData(self.model)
        if self._render_mode == "gui":
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.cam.fixedcamid = 0
            self.viewer.cam.type = 2
            self.viewer._render_every_frame = False
            self.viewer._run_speed = 100
        elif self._render_mode == "offscreen":
            self.viewer = mujoco.Renderer(self.model, 128, 128)

        self.data.ctrl[:] = self._init_position
        mujoco.mj_step(self.model, self.data, nstep=2000)
        self.data.ctrl[4] = -np.pi/2
        mujoco.mj_step(self.model, self.data, nstep=2000)
        self._t = 0

    def _create_scene(self):
        return create_tabletop_scene(
            object_color=[0.2, 0.2, 1.0, 1.0],         # blue
            container_color=[0.2, 1.0, 0.2, 1.0],      # green
            pick_pos=self.pick_pos,
            place_pos=self.place_pos,
            table_color=[0.7, 0.7, 0.7, 1.0],          # gray (default)
            wall_color=[1.0, 0.2, 0.2, 1.0],            # red walls
            groundplane_builtin="flat",              # flat ground plane
        )

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self._render_mode == "gui":
            self.viewer.render()