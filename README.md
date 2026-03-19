# G1 XHand Standby Controller

IsaacLab environment for training a 29-DOF standby policy on the Unitree G1
with XHands. The policy maintains upright stance while tracking arbitrary
body joint position commands, with payload adaptation via mass randomization.

Standalone copy of this package (train/play only): [jefferzn2001/g1_xhand_manipulation](https://github.com/jefferzn2001/g1_xhand_manipulation).

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

# Full training with wandb logging (standby + waist_yaw, 4096 envs)
python scripts/rsl_rl/train.py --task Custom-Standby-G1-v0 --headless \
    --num_envs 4096 \
    --run_name standby_v10_waist_yaw \
    --logger wandb \
    --log_project_name g1_standby

# Play a checkpoint in Isaac Sim (16 envs; use the Play task variant)
python scripts/rsl_rl/play.py --task Custom-Standby-G1-Play-v0 --num_envs 16 \
    --load_run standby_v10_waist_yaw \
    --checkpoint model_1500.pt

# Export ONNX (latest checkpoint) → copies to xhand_manip_controller/config/policies/standby_policy.onnx
python scripts/export_deploy.py --run <run_name>
# Pin a checkpoint:  --checkpoint model_15000.pt

# Deploy to MuJoCo (this workspace only; not in the standalone GitHub repo)
cd ~/Desktop/xhand_manip_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch xhand_manip_controller mujoco.launch.py use_rl_policy:=true
```

## Friend: run `play.py` with only a `.pt` file (no training)

Your friend clones [g1_xhand_manipulation](https://github.com/jefferzn2001/g1_xhand_manipulation), installs the package, then **creates this folder layout** (names are arbitrary except they must match `--load_run` and the checkpoint filename):

**Project root** = folder that contains `scripts/` and `source/` (standalone repo) **or** `locomotion/g1_xhand_standby` in **this** workspace.

**Exact on-disk path for the weights:**

```text
<PROJECT_ROOT>/logs/rsl_rl/custom_g1_standby/<ANY_NEW_FOLDER_NAME>/model_<STEP>.pt
```

Example: pick a new run name `friend_standby` and copy Jeff’s file as `model_8000.pt`:

```bash
cd <PROJECT_ROOT>    # e.g. clone of g1_xhand_manipulation, OR locomotion/g1_xhand_standby here
pip install -e source/custom_standby

mkdir -p logs/rsl_rl/custom_g1_standby/friend_standby
cp /where/you/saved/model_8000.pt logs/rsl_rl/custom_g1_standby/friend_standby/

python scripts/rsl_rl/play.py --task Custom-Standby-G1-Play-v0 --num_envs 16 \
    --load_run friend_standby \
    --checkpoint model_8000.pt
```

- **`--load_run`** = the **folder name only** under `custom_g1_standby/` (here: `friend_standby`). You can rename it anything; just use the same string in `mkdir` and in `--load_run`.
- **`--checkpoint`** = the **filename only** inside that folder; it must match the file you copied (e.g. if the file is `model_8000.pt`, pass exactly `model_8000.pt`).
- Always run `play.py` with cwd = `<PROJECT_ROOT>` so `logs/rsl_rl/...` resolves.

## Checkpoints (`--load_run` / `--checkpoint`) — if you trained yourself

- **Root:** `<PROJECT_ROOT>/logs/rsl_rl/custom_g1_standby/` (`custom_g1_standby` is the RSL-RL `experiment_name`.)
- **Run folder:** your training `--run_name`, or a **timestamp** folder if you omitted `--run_name`.
- **Resolved path:** `logs/rsl_rl/custom_g1_standby/<load_run>/<checkpoint>` (e.g. `model_1500.pt`).

## Observation Space (122 dims)

| Term | Dimensions | Description |
|------|-----------|-------------|
| base_ang_vel | 3 | IMU angular velocity |
| projected_gravity | 3 | Gravity in body frame |
| body_pose_commands | 29 | Target joint positions |
| joint_pos | 29 | Current joint positions (relative) |
| joint_vel | 29 | Current joint velocities (relative) |
| last_action | 29 | Previous policy output |
