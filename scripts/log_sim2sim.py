"""Log joint states during MuJoCo sim2sim deployment.

Records /joint_states and prints diagnostic info every 0.5s.
Run alongside: ros2 launch xhand_manip_controller mujoco.launch.py use_rl_policy:=true

Usage:
    python scripts/log_sim2sim.py [--duration 10]
"""

import argparse
import os
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

TRAINING_DEFAULTS = {
    "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669, "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 1.28, "right_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2,
}


class SimLogger(Node):
    """ROS2 node that logs joint states for sim2sim debugging."""

    def __init__(self, duration: float):
        super().__init__("sim2sim_logger")
        self.sub = self.create_subscription(
            JointState, "/joint_states", self.cb, 10
        )
        self.start_time = time.time()
        self.duration = duration
        self.msg_count = 0
        self.last_print = 0.0
        self.first_msg = None
        self.get_logger().info(f"Logging for {duration}s. Waiting for /joint_states...")

    def cb(self, msg: JointState) -> None:
        """Process joint state message."""
        self.msg_count += 1
        elapsed = time.time() - self.start_time

        if self.first_msg is None:
            self.first_msg = msg
            self.get_logger().info(f"First message received! Joints: {len(msg.name)}")
            print("\n=== INITIAL JOINT POSITIONS ===")
            for i, name in enumerate(msg.name):
                pos = msg.position[i] if i < len(msg.position) else 0
                default = TRAINING_DEFAULTS.get(name, 0.0)
                diff = pos - default
                marker = " <-- MISMATCH" if abs(diff) > 0.05 and name in TRAINING_DEFAULTS else ""
                if name in LEG_JOINTS or abs(diff) > 0.05:
                    print(f"  {name:35s} pos={pos:8.4f}  default={default:8.4f}  diff={diff:+8.4f}{marker}")
            print()

        if elapsed - self.last_print >= 0.5:
            self.last_print = elapsed
            name_to_idx = {n: i for i, n in enumerate(msg.name)}

            leg_pos = []
            leg_vel = []
            for jn in LEG_JOINTS:
                idx = name_to_idx.get(jn)
                if idx is not None:
                    p = msg.position[idx] if idx < len(msg.position) else 0
                    v = msg.velocity[idx] if idx < len(msg.velocity) else 0
                    leg_pos.append(p)
                    leg_vel.append(v)

            max_vel = max(abs(v) for v in leg_vel) if leg_vel else 0
            max_pos_err = 0
            for jn in LEG_JOINTS:
                idx = name_to_idx.get(jn)
                if idx is not None:
                    p = msg.position[idx] if idx < len(msg.position) else 0
                    d = TRAINING_DEFAULTS.get(jn, 0.0)
                    max_pos_err = max(max_pos_err, abs(p - d))

            print(
                f"t={elapsed:5.1f}s | msgs={self.msg_count:5d} | "
                f"max_leg_vel={max_vel:7.3f} | "
                f"max_pos_err={max_pos_err:7.3f} | "
                f"hip_pitch_L={msg.position[name_to_idx.get('left_hip_pitch_joint', 0)]:+.3f} "
                f"knee_L={msg.position[name_to_idx.get('left_knee_joint', 0)]:+.3f} "
                f"ankle_L={msg.position[name_to_idx.get('left_ankle_pitch_joint', 0)]:+.3f}"
            )

        if elapsed > self.duration:
            print(f"\n=== FINAL JOINT POSITIONS (t={elapsed:.1f}s) ===")
            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            for jn in LEG_JOINTS:
                idx = name_to_idx.get(jn)
                if idx is not None:
                    p = msg.position[idx]
                    v = msg.velocity[idx] if idx < len(msg.velocity) else 0
                    d = TRAINING_DEFAULTS.get(jn, 0.0)
                    print(f"  {jn:35s} pos={p:+8.4f}  vel={v:+8.4f}  default={d:+8.4f}")
            print(f"\nTotal messages: {self.msg_count}")
            rclpy.shutdown()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    rclpy.init()
    node = SimLogger(args.duration)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()


if __name__ == "__main__":
    main()
