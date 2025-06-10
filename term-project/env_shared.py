import collections
import numpy as np
from dm_control import mjcf
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from copy import deepcopy

IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success'])


def create_tabletop_scene(
    object_color=None,
    container_color=None,
    pick_pos=None,
    place_pos=None,
    table_color=None,
    wall_color=None,
    groundplane_rgb1=None,
    groundplane_rgb2=None,
    groundplane_builtin=None,
    table_texture=None,
    table_texture_path=None,
    floor_texture=None,
    floor_texture_path=None
):
    scene = create_empty_scene(
        groundplane_rgb1=groundplane_rgb1,
        groundplane_rgb2=groundplane_rgb2,
        groundplane_builtin=groundplane_builtin,
        floor_texture=floor_texture,
        floor_texture_path=floor_texture_path
    )
    add_camera_to_scene(scene, "topdown", [0.73, 0., 2.3], [0.68, 0, 0])
    add_camera_to_scene(scene, "frontface", [2.5, 0., 2.0], [-1.5, 0, 0])
    create_base(scene, [0, 0, 0.5], 0.5)

    # Create the tabletop with a box and legs
    # If a texture is specified, create the texture and material
    table_material = None
    if table_texture == "image" and table_texture_path is not None:
        image_tex = scene.asset.add(
            "texture",
            name="image",
            type="2d",
            file=table_texture_path,
            width="512",
            height="512"
        )
        image_mat = scene.asset.add(
            "material",
            name="image_mat",
            texture=image_tex,
            texrepeat=[2, 2],
            texuniform=True,
            reflectance=0.1
        )
        table_material = "image_mat"

    create_object(
        scene, "box", [0.7, 0, 1], [1, 0, 0, 0], [0.5, 0.5, 0.02],
        table_color if table_color is not None else [0.7, 0.7, 0.7, 1.0],
        friction=[0.2, 0.005, 0.0001],
        name="table", static=True,
        material=table_material
    )
    create_object(scene, "box", [0.7, 0, 0.5], [1, 0, 0, 0],
                  [0.05, 0.05, 0.5], [0.9, 0.9, 0.9, 1.0],
                  name="table_leg", static=True)
    wall_rgba = wall_color if wall_color is not None else [0.3, 0.3, 1.0, 1.0]
    create_object(scene, "capsule", [0.7, 0.5, 1.04], [0, 0.7071068, 0, 0.7071068],
                  [0.02, 0.5], wall_rgba, name="right_wall", static=True)
    create_object(scene, "capsule", [0.7, -0.5, 1.04], [0, 0.7071068, 0, 0.7071068],
                  [0.02, 0.5], wall_rgba, name="left_wall", static=True)
    create_object(scene, "capsule", [0.2, 0., 1.04], [0.7071068, 0.7071068, 0, 0],
                  [0.02, 0.5], wall_rgba, name="top_wall", static=True)
    create_object(scene, "capsule", [1.2, 0., 1.04], [0.7071068, 0.7071068, 0, 0],
                  [0.02, 0.5], wall_rgba, name="bottom_wall", static=True)
    scene.find("site", "attachment_site").attach(create_ur5e_robotiq85f())

    # Add a movable box at the pick position for pick-and-place.
    pick_box_pos = pick_pos if pick_pos is not None else [0.7, 0.0, 1.02]
    create_object(
        scene,
        "box",
        pick_box_pos,                # position
        [1, 0, 0, 0],                # orientation (quaternion)
        [0.025, 0.025, 0.025],       # size
        object_color if object_color is not None else [1.0, 0.2, 0.2, 1.0],  # color (rgba)
        friction=[2.0, 0.2, 0.01],   # increased friction values
        name="pick_box",
        static=False
    )

    # Add a small open container at the place position
    container_box_pos = place_pos if place_pos is not None else [0.7, 0.2, 1.02]
    create_box(
        scene,
        pos=container_box_pos,        # place position
        quat=[1, 0, 0, 0],
        size=[0.05, 0.05, 0.03],    # inner size (x, y, z)
        width=0.01,                 # wall thickness
        rgba=container_color if container_color is not None else [0.2, 0.2, 1.0, 1.0],   # blue color
        friction=[1.0, 0.01, 0.001],
        lid_type=None,               # no lid
        name="place_container",
        static=True
    )
    return scene


def create_empty_scene(
    groundplane_rgb1=None,
    groundplane_rgb2=None,
    groundplane_builtin=None,
    floor_texture=None,
    floor_texture_path=None
):
    root = mjcf.RootElement()
    root.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    root.visual.headlight.ambient = [0.1, 0.1, 0.1]
    root.visual.headlight.specular = [0.0, 0.0, 0.0]
    root.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
    getattr(root.visual, "global").azimuth = 120
    getattr(root.visual, "global").elevation = -20

    # Set defaults if not provided
    rgb1 = groundplane_rgb1 if groundplane_rgb1 is not None else [0.2, 0.3, 0.4]
    rgb2 = groundplane_rgb2 if groundplane_rgb2 is not None else [0.1, 0.2, 0.3]
    builtin = groundplane_builtin if groundplane_builtin is not None else "checker"

    root.asset.add("texture", type="skybox", builtin="gradient", rgb1=[0.3, 0.5, 0.7],
                   rgb2=[0, 0, 0], width="512", height="3072")

    # Create the ground plane texture and material
    # If a texture is specified, create the texture and material
    if floor_texture == "image" and floor_texture_path is not None:
        floor_tex = root.asset.add(
            "texture",
            type="2d",
            name="floor_img",
            file=floor_texture_path,
            width="512",
            height="512"
        )
        floor_mat = root.asset.add(
            "material",
            name="floor_img_mat",
            texture=floor_tex,
            texrepeat=[5, 5],
            texuniform=True,
            reflectance=0.2
        )
    # If no texture is specified, create a default ground plane texture and material
    else:
        groundplane = root.asset.add(
            "texture",
            type="2d",
            name="groundplane",
            builtin=builtin,
            mark="edge",
            rgb1=rgb1,
            rgb2=rgb2,
            markrgb=[0.8, 0.8, 0.8],
            width="300",
            height="300"
        )
        floor_mat = root.asset.add(
            "material",
            name="groundplane",
            texture=groundplane,
            texrepeat=[5, 5],
            texuniform=True,
            reflectance=0.2
        )

    # If a texture is specified, use the image material; otherwise, use the default ground plane material
    if floor_texture == "image" and floor_texture_path is not None:
        root.worldbody.add("geom", type="plane", material=floor_mat, size=[0, 0, 0.05])
    else:
        root.worldbody.add("geom", type="plane", material=floor_mat, size=[0, 0, 0.05])
    return root


def create_ur5e_robotiq85f():
    robot = mjcf.from_path("mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
    gripper = mjcf.from_path("mujoco_menagerie/robotiq_2f85/2f85.xml")
    gripper.worldbody.add("site", name="gripper_site", pos=[0, 0, 0.15], size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 0])
    robot.find("site", "attachment_site").attach(gripper)
    return robot


def create_object(root, obj_type, pos, quat, size, rgba, friction=[0.5, 0.005, 0.0001], density=1000,
                  name=None, static=False, color=None, material=None):
    # If color is provided, override rgba
    final_rgba = color if color is not None else rgba
    body = root.worldbody.add("body", pos=pos, quat=quat, name=name)
    if not static:
        body.add("joint", type="free")
    geom_kwargs = dict(type=obj_type, size=size, rgba=final_rgba, friction=friction, name=name, density=density)
    if material is not None:
        geom_kwargs["material"] = material
    body.add("geom", **geom_kwargs)
    return root


def create_box(root, pos, quat, size, width, rgba, friction=[0.5, 0.005, 0.0001],
               lid_type="slide", name=None, static=False, color=None):
    # If color is provided, override rgba
    final_rgba = color if color is not None else rgba
    base = root.worldbody.add("body", pos=pos, quat=quat, name=name)
    if not static:
        base.add("joint", type="free")
    base.add("geom", type="box", size=[size[0] + width, size[1] + width, width/2],
             rgba=final_rgba, friction=friction, pos=[0, 0, -(size[2]+width/2)], mass=0.025)
    base.add("geom", type="box", size=[width/2, size[1] + width, size[2]],
             rgba=final_rgba, friction=friction, pos=[size[0]+width/2, 0, 0], mass=0.025)
    base.add("geom", type="box", size=[width/2, size[1] + width, size[2]],
             rgba=final_rgba, friction=friction, pos=[-(size[0]+width/2), 0, 0], mass=0.025)
    base.add("geom", type="box", size=[size[0], width/2, size[2]],
             rgba=final_rgba, friction=friction, pos=[0, size[1]+width/2, 0], mass=0.025)
    base.add("geom", type="box", size=[size[0], width/2, size[2]],
             rgba=final_rgba, friction=friction, pos=[0, -(size[1]+width/2), 0], mass=0.025)
    lid = base.add("body", pos=[0, 0, size[2] + width/2])
    if lid_type == "slide":
        lid.add("joint", type="slide", axis=[0, 1, 0], range=[-2*size[1], 2*size[1]], damping=0.1)
        lid.add("geom", type="box", size=[size[0]+width, size[1]+width, width/2],
                rgba=[final_rgba[0]*0.8, final_rgba[1]*0.8, final_rgba[2]*0.8, 1], friction=friction, pos=[0, 0, 0],
                mass=0.025)
        lid.add("geom", type="cylinder", size=[0.005, 0.01], rgba=[0.8, 0.8, 0.8, 1.0],
                pos=[0, -0.02, 0.01+width/2], mass=0.025)
        lid.add("geom", type="cylinder", size=[0.005, 0.01], rgba=[0.8, 0.8, 0.8, 1.0],
                pos=[0, 0.02, 0.01+width/2], mass=0.025)
        lid.add("geom", type="capsule", size=[0.005, 0.02], rgba=[0.8, 0.8, 0.8, 1.0],
                friction=[1., 0.005, 0.0001],
                pos=[0, 0, 0.02+width/2], quat=[0.7071068, 0.7071068, 0, 0], mass=0.025)
    if lid_type == "hinge":
        lid.add("joint", type="hinge", axis=[1, 0, 0], range=[0, np.pi], pos=[0, -(size[1]+width/2), 0])
        lid.add("geom", type="box", size=[size[0]+width, size[1]+width, width/2],
                rgba=[final_rgba[0]*0.8, final_rgba[1]*0.8, final_rgba[2]*0.8, 1], friction=friction, pos=[0, 0, 0],
                mass=0.025)
        lid.add("geom", type="box", size=[0.0075, 0.0075, 0.0075], rgba=[0.8, 0.8, 0.8, 1.0],
                friction=[1., 0.005, 0.0001],
                pos=[0, size[1]-0.0075, 0.0075+width/2], mass=0.025)

    return root


def create_visual(root, obj_type, pos, quat, size, rgba, name=None):
    body = root.worldbody.add("body", pos=pos, quat=quat, name=name)
    body.add("site", type=obj_type, size=size, rgba=rgba, name=name)
    return root


def create_base(root, position, height, rgba=[0.5, 0.5, 0.5, 1.0]):
    body = root.worldbody.add("body", pos=position, name="groundbase")
    body.add("geom", type="cylinder", size=[0.1, height], rgba=rgba, name="groundbase")
    body.add("site", pos=[0, 0, height], name="attachment_site")
    return root


def add_camera_to_scene(root, name, position, target):
    target_dummy = root.worldbody.add("body", pos=target)
    root.worldbody.add("camera", name=name, mode="targetbody", pos=position, target=target_dummy)
    return root


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             point1[0], point1[1], point1[2],
                             point2[0], point2[1], point2[2])


# modified from https://github.com/deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
def qpos_from_site_pose(model,
                        data,
                        site_name,
                        target_pos=None,
                        target_quat=None,
                        joint_names=None,
                        tol=1e-14,
                        rot_weight=1.0,
                        regularization_threshold=0.1,
                        regularization_strength=3e-2,
                        max_update_norm=2.0,
                        progress_thresh=20.0,
                        max_steps=20,
                        inplace=False):
    dtype = data.qpos.dtype

    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif target_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError("At least one of `target_pos` or `target_quat` must be specified.")

    update_nv = np.zeros(model.nv, dtype=dtype)

    if target_quat is not None:
        site_xquat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)

    if not inplace:
        data = deepcopy(data)

    mujoco.mj_fwdPosition(model, data)
    site_id = model.site(site_name).id

    # todo: check they are indeed updated in place

    if joint_names is None:
        dof_indices = slice(None)
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        dof_indices = [model.joint(name).id for name in joint_names]
    else:
        raise ValueError(f"`joint_names` must be either None, a list, a tuple, or a numpy array; "
                         f"got {type(joint_names)}.")

    success = False
    for steps in range(max_steps):
        err_norm = 0.0

        if target_pos is not None:
            # translational error.
            err_pos[:] = target_pos - data.site(site_name).xpos
            err_norm += np.linalg.norm(err_pos)
        if target_quat is not None:
            # rotational error.
            mujoco.mju_mat2Quat(site_xquat, data.site(site_name).xmat)
            mujoco.mju_negQuat(neg_site_xquat, site_xquat)
            mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            success = True
            break
        else:
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
            jac_joints = jac[:, dof_indices]
            reg_strength = regularization_strength if err_norm > regularization_threshold else 0.0
            update_joints = nullspace_method(jac_joints, err, regularization_strength=reg_strength)
            update_norm = np.linalg.norm(update_joints)

        progress_criterion = err_norm / update_norm
        if progress_criterion > progress_thresh:
            break

        if update_norm > max_update_norm:
            update_joints *= max_update_norm / update_norm

        update_nv[dof_indices] = update_joints
        mujoco.mj_integratePos(model, data.qpos, update_nv, 1)
        mujoco.mj_fwdPosition(model, data)

        if not inplace:
            qpos = data.qpos.copy()
        else:
            qpos = data.qpos

    return IKResult(qpos, err_norm, steps, success)


# modified from https://github.com/deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]


def pick_and_place(env, pick_pos, place_pos, height_above=0.1, grasp_height=0.02, rotation=[0, 180, 0]):
    """
    Executes a pick and place routine.
    Moves the end-effector to the pick position, grasps the object, moves it to the place position, and releases it.
    Returns True if the object is in the container after the operation, else False.
    """
    # Move above pick
    above_pick = [pick_pos[0], pick_pos[1], pick_pos[2] + height_above]
    set_ee_in_cartesian(env, above_pick, rotation=rotation)
    # Move down to grasp
    grasp_pick = [pick_pos[0], pick_pos[1], pick_pos[2] + grasp_height]
    set_ee_in_cartesian(env, grasp_pick, rotation=rotation)
    # Close gripper
    close_gripper(env)
    # Move up with object
    set_ee_in_cartesian(env, above_pick, rotation=rotation)
    # Move above place
    above_place = [place_pos[0], place_pos[1], place_pos[2] + height_above]
    set_ee_in_cartesian(env, above_place, rotation=rotation)
    # Move down to place
    place_drop = [place_pos[0], place_pos[1], place_pos[2] + grasp_height]
    set_ee_in_cartesian(env, place_drop, rotation=rotation)
    # Open gripper
    open_gripper(env)
    # Move up after placing
    set_ee_in_cartesian(env, above_place, rotation=rotation)
    # Check if object is in container
    return is_object_in_container(env)


def pick_and_place_stepwise(env, pick_pos, place_pos, height_above=0.1, grasp_height=0.02, rotation=[0, 180, 0], step_size=0.01, step_delay=0.0):
    """
    Generator for step-by-step pick and place.
    At each yield, returns (observation, action).
    Action space:
        0: move left (-x)
        1: move right (+x)
        2: move up (+y)
        3: move down (-y)
        4: pick (close gripper)
        5: place (open gripper)
    """
    import time

    def move_to(env, target, rotation, step_size):
        curr_pos, _ = env._get_ee_pose()
        curr_pos = np.array(curr_pos)
        target = np.array(target)
        while np.linalg.norm(target - curr_pos) > 1e-4:
            delta = target - curr_pos
            # Only move in x or y
            if abs(delta[0]) >= abs(delta[1]):
                axis = 0  # x
                if delta[axis] < 0:
                    action = 0  # left (-x)
                else:
                    action = 1  # right (+x)
            else:
                axis = 1  # y
                if delta[axis] > 0:
                    action = 2  # up (+y)
                else:
                    action = 3  # down (-y)
            move = np.zeros(3)
            move[axis] = np.sign(delta[axis]) * min(step_size, abs(delta[axis]))
            next_pos = curr_pos + move
            next_pos[2] = target[2]  # always set z to target z for this step
            env._set_ee_in_cartesian(next_pos, rotation=rotation, n_splits=1)
            curr_pos = next_pos
            obs = None
            if step_delay > 0:
                time.sleep(step_delay)
            yield obs, action

    # Move above pick (x/y only)
    above_pick = [pick_pos[0], pick_pos[1], pick_pos[2] + height_above]
    yield from move_to(env, above_pick, rotation, step_size)
    # Move down in z to grasp
    curr_pos, _ = env._get_ee_pose()
    grasp_pick = [curr_pos[0], curr_pos[1], pick_pos[2] + grasp_height]
    set_ee_in_cartesian(env, grasp_pick, rotation=rotation, n_splits=1)
    obs = None
    if step_delay > 0:
        time.sleep(step_delay)
    # Pick (close gripper)
    close_gripper(env)
    if step_delay > 0:
        time.sleep(step_delay)
    yield obs, 4  # 4: pick
    # Move up in z
    above_pick = [curr_pos[0], curr_pos[1], pick_pos[2] + height_above]
    set_ee_in_cartesian(env, above_pick, rotation=rotation, n_splits=1)
    if step_delay > 0:
        time.sleep(step_delay)
    # Move above place (x/y only)
    above_place = [place_pos[0], place_pos[1], place_pos[2] + height_above]
    yield from move_to(env, above_place, rotation, step_size)
    # Move down in z to place
    curr_pos, _ = env._get_ee_pose()
    place_drop = [curr_pos[0], curr_pos[1], place_pos[2] + grasp_height]
    set_ee_in_cartesian(env, place_drop, rotation=rotation, n_splits=1)
    obs = None
    if step_delay > 0:
        time.sleep(step_delay)
    # Place (open gripper)
    open_gripper(env)
    if step_delay > 0:
        time.sleep(step_delay)
    yield obs, 5  # 5: place
    # Move up in z
    above_place = [curr_pos[0], curr_pos[1], place_pos[2] + height_above]
    set_ee_in_cartesian(env, above_place, rotation=rotation, n_splits=1)
    if step_delay > 0:
        time.sleep(step_delay)


def is_object_in_container(env, pick_box_name="pick_box", container_name="place_container", xy_thresh=0.05, z_thresh=0.05):
    """
    Returns True if the pick_box is inside the container (by x/y/z proximity).
    """
    # Use mujoco.mj_name2id to get body ids
    try:
        pick_box_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, pick_box_name)
        container_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, container_name)
    except Exception:
        return False
    if pick_box_id < 0 or container_id < 0:
        return False
    pick_box_pos = env.data.xpos[pick_box_id]
    container_pos = env.data.xpos[container_id]
    xy_dist = np.linalg.norm(np.array(pick_box_pos[:2]) - np.array(container_pos[:2]))
    z_dist = abs(pick_box_pos[2] - container_pos[2])
    return xy_dist < xy_thresh and z_dist < z_thresh


def step_env_with_action(env, action, step_size=0.02, rotation=[0,180,0], pick_z=None, place_z=None, height_above=0.1, grasp_height=0.02):
    """
    Take a step in the environment according to the discrete action.
    Actions:
        0: move left (-x)
        1: move right (+x)
        2: move up (+y)
        3: move down (-y)
        4: pick (move down to pick, close gripper, move up)
        5: place (move down to place, open gripper, move up)
    """
    curr_pos, _ = env._get_ee_pose()
    curr_pos = np.array(curr_pos)
    next_pos = curr_pos.copy()
    if action == 0:
        # move left (-x)
        next_pos[0] = curr_pos[0] - step_size
        set_ee_in_cartesian(env, next_pos, rotation=rotation, n_splits=1)
    elif action == 1:
        # move right (+x)
        next_pos[0] = curr_pos[0] + step_size
        set_ee_in_cartesian(env, next_pos, rotation=rotation, n_splits=1)
    elif action == 2:
        # move up (+y)
        next_pos[1] = curr_pos[1] + step_size
        set_ee_in_cartesian(env, next_pos, rotation=rotation, n_splits=1)
    elif action == 3:
        # move down (-y)
        next_pos[1] = curr_pos[1] - step_size
        set_ee_in_cartesian(env, next_pos, rotation=rotation, n_splits=1)
    elif action == 4:
        # pick: move down to pick height, close gripper, move up
        z_down = pick_z if pick_z is not None else curr_pos[2] - height_above + grasp_height
        above = curr_pos.copy()
        down = curr_pos.copy()
        down[2] = z_down
        set_ee_in_cartesian(env, down, rotation=rotation, n_splits=1)
        close_gripper(env)
        set_ee_in_cartesian(env, above, rotation=rotation, n_splits=1)
    elif action == 5:
        # place: move down to place height, open gripper, move up
        z_down = place_z if place_z is not None else curr_pos[2] - height_above + grasp_height
        above = curr_pos.copy()
        down = curr_pos.copy()
        down[2] = z_down
        set_ee_in_cartesian(env, down, rotation=rotation, n_splits=1)
        open_gripper(env)
        set_ee_in_cartesian(env, above, rotation=rotation, n_splits=1)
    # else: do nothing


def get_joint_position(env):
    position = np.zeros(env._n_joints)
    for idx in range(env._n_joints):
        position[idx] = env.data.qpos[env._joint_qpos_idxs[idx]]
        if idx == env._gripper_idx:
            position[idx] /= env._gripper_norm
    return position


def set_joint_position(env, position_dict, max_iters=2000, threshold=0.05):
    for idx in position_dict:
        if idx == env._gripper_idx:
            env.data.ctrl[idx] = position_dict[idx]*255
        else:
            env.data.ctrl[idx] = position_dict[idx]

    max_error = 100*threshold
    it = 0
    while max_error > threshold:
        it += 1
        env.step()
        max_error = 0
        current_position = get_joint_position(env)
        for idx in position_dict:
            error = abs(current_position[idx] - position_dict[idx])
            if error > max_error:
                max_error = error
        if it > max_iters:
            break


def get_ee_pose(env):
    ee_position = env.data.site(env._ee_site).xpos
    ee_rotation = env.data.site(env._ee_site).xmat
    ee_orientation = np.zeros(4)
    mujoco.mju_mat2Quat(ee_orientation, ee_rotation)
    return ee_position, ee_orientation


def set_ee_pose(env, position, rotation=None, orientation=None, max_iters=2000, threshold=0.01):
    if rotation is not None and orientation is not None:
        raise Exception("Only one of rotation or orientation can be set")
    quat = None
    if rotation is not None:
        quat = R.from_euler("xyz", rotation, degrees=True).as_quat()
    elif orientation is not None:
        quat = orientation
    qpos = qpos_from_site_pose(env.model, env.data, env._ee_site,
                               position, quat, joint_names=env._joint_names[:-1]).qpos
    qdict = {i: qpos[q_idx][0] for i, q_idx in enumerate(env._joint_qpos_idxs[:-1])}

    max_error = 100*threshold
    it = 0
    while max_error > threshold:
        it += 1
        env.step()
        max_error = 0
        curr_pos, curr_quat = get_ee_pose(env)
        max_error += np.linalg.norm(np.array(position) - curr_pos)

        if quat is not None:
            neg_quat = np.zeros(4)
            mujoco.mju_negQuat(neg_quat, curr_quat)
            error_quat = np.zeros(4)
            mujoco.mju_mulQuat(error_quat, quat, neg_quat)
            error_vel = np.zeros(3)
            mujoco.mju_quat2Vel(error_vel, error_quat, 1)
            max_error += np.linalg.norm(error_vel)
        for idx in qdict:
            env.data.ctrl[idx] = qpos[env._joint_qpos_idxs[idx]]
        if it > max_iters:
            break


def set_ee_in_cartesian(env, position, rotation=None, max_iters=2000, threshold=0.01, n_splits=30):
    ee_position, ee_orientation = get_ee_pose(env)
    position_traj = np.linspace(ee_position, position, n_splits+1)[1:]
    if rotation is not None:
        target_orientation = R.from_euler("xyz", rotation, degrees=True).as_quat()
        r = R.from_quat([ee_orientation, target_orientation])
        slerp = Slerp([0, 1], r)
        orientation_traj = slerp(np.linspace(0, 1, n_splits+1)[1:]).as_quat()
    else:
        orientation_traj = [ee_orientation]*n_splits

    follow_ee_trajectory(env, position_traj, orientation_traj,
                         max_iters=max_iters, threshold=threshold)


def follow_ee_trajectory(env, position_traj, orientation_traj, max_iters=2000, threshold=0.01):
    n_splits = len(position_traj)
    for position, orientation in zip(position_traj, orientation_traj):
        set_ee_pose(env, position, orientation=orientation,
                    max_iters=max_iters//n_splits, threshold=threshold)


def open_gripper(env):
    set_joint_position(env, {env._gripper_idx: 0.0})


def close_gripper(env):
    set_joint_position(env, {env._gripper_idx: 1.0})