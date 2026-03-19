"""G1 XHand standby controller environment configuration.

Trains a 29-DOF policy to maintain an upright standing pose while
tracking arbitrary body joint position commands.  Key design:

1. No velocity commands or locomotion rewards.
2. Flat ground with dark sky — no terrain generator.
3. BodyPoseCommand: joint-space targets for all 29 DOFs with
   OpenHomie-style exponential arm curriculum.
4. Joint tracking rewards (L1 + tanh fine-grained bonus).
5. Wrist FK target visualization (sphere markers show where
   the target joint configuration places the wrists).
6. XHand mass randomization for payload adaptation.

Observation layout matches locomotion for sim2sim compatibility:
  base_ang_vel(3) + projected_gravity(3) + body_pose_commands(29)
  + joint_pos(29) + joint_vel(29) + actions(29) = 122 dims.

At deployment, IK can be computed externally and fed as joint
commands — the policy just tracks whatever joint targets it receives.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from custom_standby.tasks.standby import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg as BaseActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg as BaseObservationsCfg,
    RewardsCfg,
)
from custom_standby.robots.g1 import G1_CYLINDER_CFG, G1_BODY_ACTION_SCALE

# All 29 G1 body joints (legs + waist + arms, no hands)
G1_BODY_JOINT_NAMES = [
    ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint",
    ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
]


@configclass
class G1StandbyObservationsCfg(BaseObservationsCfg):
    """Standby observations — joint-space body pose commands.

    Matches the locomotion policy layout for sim2sim compatibility.
    Policy obs: 3+3+29+29+29+29 = 122 dims.
    Critic obs: above + 3 (base_lin_vel) = 125 dims.
    """

    @configclass
    class StandbyPolicyCfg(ObsGroup):
        """Policy observations: IMU + body pose commands + 29 joint encoders."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        body_pose_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "body_targets"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=G1_BODY_JOINT_NAMES
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=G1_BODY_JOINT_NAMES
                ),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class StandbyCriticCfg(ObsGroup):
        """Critic observations: includes base_lin_vel as privileged info."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        body_pose_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "body_targets"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=G1_BODY_JOINT_NAMES
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=G1_BODY_JOINT_NAMES
                ),
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: StandbyPolicyCfg = StandbyPolicyCfg()
    critic: StandbyCriticCfg = StandbyCriticCfg()


@configclass
class StandbyActionsCfg(BaseActionsCfg):
    """Full 29-DOF G1 body (legs + waist + arms, no hands)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_BODY_JOINT_NAMES,
        scale=G1_BODY_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class G1StandbyRewards(RewardsCfg):
    """Standby reward set — joint-space pose tracking + balance.

    Balance comes from L2 penalties (upright, base_height, velocity)
    plus positive survival signals (alive, feet_contact).
    Arm tracking is the primary task signal via L1 error + tanh bonus.
    """

    # -- task: body pose tracking (L1 penalty + tanh bonus) --
    joint_target_tracking = RewTerm(
        func=mdp.body_target_tracking,
        weight=-2.0,
        params={
            "command_name": "body_targets",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=G1_BODY_JOINT_NAMES,
            ),
        },
    )
    joint_target_tracking_fine = RewTerm(
        func=mdp.body_target_tracking_tanh,
        weight=1.0,
        params={
            "command_name": "body_targets",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=G1_BODY_JOINT_NAMES,
            ),
            "std": 0.5,
        },
    )

    alive = RewTerm(func=mdp.is_alive, weight=2.0)

    # -- base stability penalties --
    upright = RewTerm(func=mdp.upward, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.78},
    )
    base_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_xy_vel = RewTerm(func=mdp.base_xy_vel_penalty, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)

    # -- joint regularization --
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*", ".*_knee_joint", ".*_ankle_.*",
                    "waist_.*", ".*_shoulder_.*", ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            ),
        },
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*", ".*_knee_joint", ".*_ankle_.*",
                    "waist_.*", ".*_shoulder_.*", ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            ),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.10)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*", ".*_knee_joint", ".*_ankle_.*",
                    "waist_.*", ".*_shoulder_.*", ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            ),
        },
    )
    energy_penalty = RewTerm(func=mdp.energy, weight=-2e-5)

    # -- feet --
    feet_contact = RewTerm(
        func=mdp.feet_contact_reward,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_roll.*"
            ),
        },
    )

    # -- contacts --
    undesired_contacts_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["(?!.*ankle.*|.*_hand_.*).*"],
            ),
        },
    )

    # -- disable stock rewards (replaced above) --
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = None
    dof_torques_l2 = None
    dof_acc_l2 = None
    action_rate_l2 = None
    feet_air_time = None
    undesired_contacts = None
    flat_orientation_l2 = None


@configclass
class G1StandbyEnvCfg(LocomotionVelocityRoughEnvCfg):
    """G1 XHand standby controller environment.

    Flat ground, dark sky, no locomotion rewards.  Joint-space body pose
    command with OpenHomie arm curriculum.  Wrist FK target visualization.
    XHand mass randomization for payload adaptation.
    """

    observations: G1StandbyObservationsCfg = G1StandbyObservationsCfg()
    actions: StandbyActionsCfg = StandbyActionsCfg()
    rewards: G1StandbyRewards = G1StandbyRewards()

    def __post_init__(self):
        """Finalize the environment configuration."""
        super().__post_init__()

        self.scene.num_envs = 4096
        self.scene.robot = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.height_scanner = None

        # Flat ground — no terrain generator
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Dark sky: uniform dome light with no HDR texture
        from isaaclab.assets import AssetBaseCfg
        self.scene.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                color=(0.75, 0.75, 0.75),
                intensity=3000.0,
            ),
        )

        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**24

        # =============================================================
        # Domain randomization
        # =============================================================

        # Friction
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.1, 1.6),
                "dynamic_friction_range": (0.1, 1.2),
                "restitution_range": (0.0, 0.5),
                "num_buckets": 64,
            },
        )

        # Payload on torso (-1 to 5 kg)
        self.events.add_base_mass.params["asset_cfg"].body_names = [
            "torso_link"
        ]
        self.events.add_base_mass.params["mass_distribution_params"] = (
            -1.0,
            5.0,
        )
        self.events.add_base_mass.params["operation"] = "add"

        # Wrist mass randomization — absolute range covers different hand
        # weights (lighter grippers to heavy xhands with payload).
        self.events.randomize_xhand_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=[".*_wrist_yaw_link"]
                ),
                "mass_distribution_params": (0.1, 3.0),
                "operation": "abs",
                "recompute_inertia": True,
            },
        )

        # CoM shift
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names="torso_link"
                ),
                "com_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.1, 0.1),
                    "z": (-0.1, 0.1),
                },
            },
        )

        # Joint calibration error
        self.events.add_joint_default_pos = EventTerm(
            func=mdp.randomize_joint_default_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*"]
                ),
                "pos_distribution_params": (-0.01, 0.01),
                "operation": "add",
            },
        )

        # Moderate external force on torso (reduced from ±5/±3 to ease
        # initial standing learning)
        self.events.base_external_force_torque.params[
            "asset_cfg"
        ].body_names = ["torso_link"]
        self.events.base_external_force_torque.params["force_range"] = (
            -2.0,
            2.0,
        )
        self.events.base_external_force_torque.params["torque_range"] = (
            -1.0,
            1.0,
        )

        # Reason: gentle pushes let the policy learn to stand first;
        # disturbance robustness can be increased via curriculum later.
        self.events.push_robot.interval_range_s = (5.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (-0.1, 0.1),
            "roll": (-0.15, 0.15),
            "pitch": (-0.15, 0.15),
            "yaw": (-0.3, 0.3),
        }

        # Disable wrist forces initially — they destabilize early training
        self.events.left_hand_force = None
        self.events.right_hand_force = None

        # Reset with small perturbations
        self.events.reset_robot_joints.params["position_range"] = (
            0.95,
            1.05,
        )
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
        }

        # =============================================================
        # Commands: joint-space body pose targets with arm curriculum
        # =============================================================
        self.commands.base_velocity = None
        self.commands.body_targets = mdp.BodyPoseCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0, 3.0),
            joint_names=G1_BODY_JOINT_NAMES,
            leg_offset_range=(-0.05, 0.05),
            waist_offset_range=(-0.2, 0.2),
            waist_yaw_offset_range=(-1.0, 1.0),
            arm_offset_range=(-0.8, 0.8),
            initial_ratio=0.0,
            interpolation_duration=1.0,
            debug_vis=True,
        )

        # =============================================================
        # Curriculum: survival-gated arm command progression
        # =============================================================
        self.curriculum.terrain_levels = None
        self.curriculum.arm_cmd_levels = CurrTerm(
            func=mdp.arm_cmd_levels,
            params={"step_size": 0.002, "min_mean_ep_buf": 400.0},
        )

        # =============================================================
        # Terminations
        # =============================================================
        self.terminations.base_contact.params[
            "sensor_cfg"
        ].body_names = "torso_link"
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": 0.8}
        )
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.2},
        )


@configclass
class G1StandbyEnvCfg_PLAY(G1StandbyEnvCfg):
    """Play configuration for the G1 standby controller task."""

    def __post_init__(self):
        """Reduce scene size and disable most randomization."""
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Reason: no curriculum runs during play, so set full arm range
        # directly so the policy is tested with diverse targets.
        self.commands.body_targets.initial_ratio = 1.0
