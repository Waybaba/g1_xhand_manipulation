# G1 XHand Standby Controller

IsaacLab environment for training a 29-DOF standby policy on the Unitree G1
with XHands. The policy maintains upright stance while tracking arbitrary
body joint position commands, with payload adaptation via mass randomization.

## Key Features

- **29-DOF action space** — legs + waist + arms (no hand joints)
- **FK command mode** — BodyPoseCommand generates random target joint positions
  with per-group offset ranges and OpenHomie arm curriculum
- **IK command mode** — IKPoseCommand uses DifferentialIK to resolve random
  wrist EE poses into joint positions
- **XHand mass randomization** — -1 to 3 kg on wrist links for payload adaptation
- **Dark sky scene** — flat ground with uniform dome lighting
- **EE visualization** — colored sphere markers at wrist target positions

## Quick Start

```bash
# Install the package
cd locomotion/g1_xhand_standby
conda activate mimic
pip install -e source/custom_standby

# Test training (16 envs, GUI)
cd locomotion/g1_xhand_standby
python scripts/rsl_rl/train.py --task Custom-Standby-G1-v0 --num_envs 16

# Full training (4096 envs, headless)
python scripts/rsl_rl/train.py --task Custom-Standby-G1-v0 --headless

# Export ONNX
python scripts/export_deploy.py --run <run_name>

# Deploy to MuJoCo
cd ~/Desktop/xhand_manip_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch g1_xhand_description mujoco.launch.py use_rl_policy:=true
```

## Observation Space (122 dims)

| Term | Dimensions | Description |
|------|-----------|-------------|
| base_ang_vel | 3 | IMU angular velocity |
| projected_gravity | 3 | Gravity in body frame |
| body_pose_commands | 29 | Target joint positions |
| joint_pos | 29 | Current joint positions (relative) |
| joint_vel | 29 | Current joint velocities (relative) |
| last_action | 29 | Previous policy output |
