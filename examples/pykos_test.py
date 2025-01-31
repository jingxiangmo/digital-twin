"""Test PyKOS communication."""

import argparse
import asyncio
from dataclasses import dataclass, field

from pykos import KOS

from digital_twin.actor.pykos_bot import PyKOSActor
from digital_twin.puppet.mujoco_puppet import MujocoPuppet


@dataclass
class RobotConfigs:
    joint_mapping: dict[str, int] = field(default_factory=dict)
    signs: dict[str, float] = field(default_factory=dict)

@dataclass
class ZbotConfigs(RobotConfigs):
    joint_mapping: dict[str, int] = field(default_factory=lambda: {
        # Left arm
        "left_shoulder_yaw": 11,
        "left_shoulder_pitch": 12,
        "left_elbow_yaw": 13,
        "left_gripper": 14,
        # Right arm
        "right_shoulder_yaw": 21,
        "right_shoulder_pitch": 22,
        "right_elbow_yaw": 23,
        "right_gripper": 24,
        # Left leg
        "left_hip_yaw": 31,
        "left_hip_roll": 32,
        "left_hip_pitch": 33,
        "left_knee": 34,
        "left_ankle": 35,
        # Right leg
        "right_hip_yaw": 41,
        "right_hip_roll": 42,
        "right_hip_pitch": 43,
        "right_knee": 44,
        "right_ankle": 45
    })

    signs: dict[str, float] = field(default_factory=lambda: {
        # Left arm
        "left_shoulder_yaw": 1,
        "left_shoulder_pitch": 1,
        "left_elbow_yaw": -1,
        "left_gripper": 1,
        # Right arm
        "right_shoulder_yaw": 1,
        "right_shoulder_pitch": 1,
        "right_elbow_yaw": 1,
        "right_gripper": 1,
        # Left leg
        "left_hip_yaw": 1,
        "left_hip_roll": -1,
        "left_hip_pitch": 1,
        "left_knee": 1,
        "left_ankle": 1,
        # Right leg
        "right_hip_yaw": 1,
        "right_hip_roll": 1,
        "right_hip_pitch": 1,
        "right_knee": 1,
        "right_ankle": 1
    })

@dataclass
class KbotConfigs(RobotConfigs):
    joint_mapping: dict[str, int] = field(default_factory=lambda: {
        # Left arm
        "L_shoulder_y_03": 11,
        "L_shoulder_x_03": 12,
        "L_shoulder_z_02": 13,
        "L_elbow_02": 14,
        "L_wrist_02": 15,
        # Right arm
        "R_shoulder_y_03": 21,
        "R_shoulder_x_03": 22,
        "R_shoulder_z_02": 23,
        "R_elbow_02": 24,
        "R_wrist_02": 25,
        # Left leg
        "L_hip_y_04": 31,
        "L_hip_x_03": 32,
        "L_hip_z_03": 33,
        "L_knee_04": 34,
        "L_ankle_02": 35,
        # Right leg
        "R_hip_y_04": 41,
        "R_hip_x_03": 42,
        "R_hip_z_03": 43,
        "R_knee_04": 44,
        "R_ankle_02": 45
    })

    signs: dict[str, float] = field(default_factory=lambda: {
        # Left arm
        "L_shoulder_y_03": 1,
        "L_shoulder_x_03": 1,
        "L_shoulder_z_02": 1,
        "L_elbow_02": 1,
        "L_wrist_02": 1,
        # Right arm
        "R_shoulder_y_03": 1,
        "R_shoulder_x_03": 1,
        "R_shoulder_z_02": 1,
        "R_elbow_02": 1,
        "R_wrist_02": 1,
        # Left leg
        "L_hip_y_04": 1,
        "L_hip_x_03": 1,
        "L_hip_z_03": 1,
        "L_knee_04": -1,
        "L_ankle_02": 1,
        # Right leg
        "R_hip_y_04": 1,
        "R_hip_x_03": 1,
        "R_hip_z_03": 1,
        "R_knee_04": 1,
        "R_ankle_02": 1
    })

@dataclass
class KbotNakedConfigs(RobotConfigs):
    joint_mapping: dict[str, int] = field(default_factory=lambda: {
        # Left arm
        "left_shoulder_pitch_03": 11,
        "left_shoulder_roll_03": 12,
        "left_shoulder_yaw_02": 13,
        "left_elbow_02": 14,
        "left_wrist_02": 15,
        # Right arm
        "right_shoulder_pitch_03": 21,
        "right_shoulder_roll_03": 22,
        "right_shoulder_yaw_02": 23,
        "right_elbow_02": 24,
        "right_wrist_02": 25,
        # Left leg
        "left_hip_pitch_04": 31,
        "left_hip_roll_03": 32,
        "left_hip_yaw_03": 33,
        "left_knee_04": 34,
        "left_ankle_02": 35,
        # Right leg
        "right_hip_pitch_04": 41,
        "right_hip_roll_03": 42,
        "right_hip_yaw_03": 43,
        "right_knee_04": 44,
        "right_ankle_02": 45
    })

    signs: dict[str, float] = field(default_factory=lambda: {
        # Left arm
        "left_shoulder_pitch_03": 1,
        "left_shoulder_roll_03": -1,
        "left_shoulder_yaw_02": 1,
        "left_elbow_02": 1,
        "left_wrist_02": 1,
        # Right arm
        "right_shoulder_pitch_03": 1,
        "right_shoulder_roll_03": 1,
        "right_shoulder_yaw_02": -1,
        "right_elbow_02": 1,
        "right_wrist_02": 1,
        # Left leg
        "left_hip_pitch_04": 1,
        "left_hip_roll_03": 1,
        "left_hip_yaw_03": 1,
        "left_knee_04": -1,  # Note: has negative limit range
        "left_ankle_02": 1,
        # Right leg
        "right_hip_pitch_04": 1,
        "right_hip_roll_03": 1,
        "right_hip_yaw_03": 1,
        "right_knee_04": -1,
        "right_ankle_02": -1
    })

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_name", type=str, help="Name of the Mujoco model in the K-Scale API")
    parser.add_argument("--ip", type=str, default="192.168.42.1", help="IP address of the robot")
    args = parser.parse_args()

    configs: RobotConfigs
    match args.mjcf_name:
        case "zbot-v2":
            configs = ZbotConfigs()
        case "kbot-v1":
            configs = KbotConfigs()
        case _:
            raise ValueError(f"No configs for {args.mjcf_name}")

    async with KOS(ip=args.ip) as kos:
        actor = PyKOSActor(kos, configs.joint_mapping, kos_signs=configs.signs)
        await actor.offset_in_place()

        # Enable torque for all actuators (example)
        for i in range(3):
            for actuator_id in [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45]:
                await kos.actuator.configure_actuator(actuator_id=actuator_id, torque_enabled=True)
                await asyncio.sleep(0.1)

        # Initialize all actuators to position 0 (example)
        for actuator_id in actor.joint_ids:
            await kos.actuator.command_actuators([{"actuator_id": actuator_id, "position": 0}])
            await asyncio.sleep(0.1)

        puppet = MujocoPuppet(args.mjcf_name)

        # -------------------------------------
        # Matplotlib setup for live bar chart
        # -------------------------------------
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        plt.ion()  # Turn on interactive mode

        # We'll treat the "actor.joint_ids" as the X-axis labels. 
        # Or you could use the joint names if you prefer more descriptive labels.
        x_labels = [str(jid) for jid in actor.joint_ids]
        x_positions = np.arange(len(actor.joint_ids))

        fig, ax = plt.subplots()
        ax.set_title("Live Actuator Angles")
        ax.set_xlabel("Actuator IDs")
        ax.set_ylabel("Angle (degrees)")
        ax.set_ylim([-180, 180])  # Adjust as needed

        # Initialize the bar plot with zeros
        bars = ax.bar(x_positions, [0]*len(actor.joint_ids), tick_label=x_labels)

        # -------------------------------------
        # Main loop
        # -------------------------------------
        while True:
            # Get the latest joint angles (in radians)
            joint_angles_rad = await actor.get_joint_angles()

            # The dictionary keys are joint names; 
            # we can map them to actuator IDs via actor.joint_ids or the config's joint_mapping.
            # For each actuator ID, find the corresponding angle in radians, then convert to degrees.
            # If you have a direct mapping from ID -> name, do that here; otherwise, adapt as needed.

            angles_degrees = []
            for actuator_id in actor.joint_ids:
                # Find the joint name from the config's mapping
                # (reverse lookup from 'configs.joint_mapping' if needed)
                joint_name = None
                for k, v in configs.joint_mapping.items():
                    if v == actuator_id:
                        joint_name = k
                        break
                # If a joint name was found, get that angle; otherwise, 0
                if joint_name and joint_name in joint_angles_rad:
                    angles_degrees.append(math.degrees(joint_angles_rad[joint_name]))
                else:
                    angles_degrees.append(0.0)

            # Update the puppet (existing code)
            await puppet.set_joint_angles(joint_angles_rad)

            # Update the bar heights in the chart
            for bar, new_height in zip(bars, angles_degrees):
                bar.set_height(new_height)

            plt.draw()
            plt.pause(0.001)  # Allow matplotlib to update

            await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())
