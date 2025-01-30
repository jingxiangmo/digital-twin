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

    joint_names = await puppet.get_joint_names()
    if not args.no_skip_root and len(joint_names) > 1:
        joint_names = joint_names[1:]

    async def walk_forward(joint_list, time_step):
        """
        Example walking function that returns a dictionary of 
        {joint_name: angle} for a simple sine-based walking gait.
        """
        import math

        freq_hip_pitch = 2.0
        freq_knee = 2.0
        amp_hip_pitch = 0.4
        amp_knee = 0.8
        phase_offset = math.pi

        angles_dict = {}
        for joint_name in joint_list:
            if "hip_pitch" in joint_name:
                offset = phase_offset if "right" in joint_name else 0.0
                angle = amp_hip_pitch * math.sin(freq_hip_pitch * time_step + offset)
            elif "knee" in joint_name:
                offset = phase_offset if "right" in joint_name else 0.0
                angle = -amp_knee * math.sin(freq_knee * time_step + offset)
            else:
                angle = 0.0

            angles_dict[joint_name] = angle

        return angles_dict

    try:
        t = 0.0
        while True:
            # Create a dictionary mapping joint names to their angles
            joint_angles_dict = await walk_forward(joint_names, t)
            await puppet.set_joint_angles(joint_angles_dict)

            t += 0.01
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
