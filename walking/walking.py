"""Test Mujoco puppet communication."""

import argparse
import asyncio

import colorlogging

from digital_twin.actor.sinusoid import SinusoidActor
from digital_twin.puppet.mujoco_puppet import MujocoPuppet


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-skip-root", action="store_true", help="Do not skip the root joint")
    args = parser.parse_args()

    colorlogging.configure()

    puppet = MujocoPuppet("zbot-v2")

    # joint_names = ['root', 'left_shoulder_yaw', 'left_shoulder_pitch', 'left_elbow', 'left_gripper', 'right_shoulder_yaw', 'right_shoulder_pitch', 'right_elbow', 'right_gripper', 'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle']
    joint_names = await puppet.get_joint_names()
    print(joint_names)

    if not args.no_skip_root:
        joint_names = joint_names[1:]

    actor = SinusoidActor(joint_names)

    try:
        while True:
            joint_angles = await actor.get_joint_angles()
            await puppet.set_joint_angles(joint_angles)
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
