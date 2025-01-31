import argparse
import asyncio
import math
import time

from pykos import KOS
from digital_twin.puppet.mujoco_puppet import MujocoPuppet

class BipedController:
    """
    Same bipedal walking logic as before, except we remove references to MuJoCo
    and instead will call pykos to directly move actuators.
    """

    def __init__(self):
        # -----------
        # Gait params
        # -----------
        self.LEG_LENGTH = 180.0      # mm
        self.hip_forward_offset = 2.04
        self.nominal_leg_height = 170.0
        self.initial_leg_height = 180.0
        self.gait_phase = 0
        self.walking_enabled = True

        # -----------
        # Variables for cyclical stepping
        # -----------
        self.stance_foot_index = 0  # 0 or 1
        self.step_cycle_length = 48
        self.step_cycle_counter = 0
        self.lateral_foot_shift = 12
        self.max_foot_lift = 20
        self.double_support_fraction = 0.2
        self.current_foot_lift = 0.0

        self.forward_offset = [0.0, 0.0]
        self.accumulated_forward_offset = 0.0
        self.previous_stance_foot_offset = 0.0
        self.previous_swing_foot_offset = 0.0
        self.step_length = 20.0

        self.lateral_offset = 0.0
        self.dyi = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        # The joint angle arrays
        self.K0 = [0.0, 0.0]  # hip pitch
        self.K1 = [0.0, 0.0]  # hip roll
        self.H  = [0.0, 0.0]  # knee
        self.A0 = [0.0, 0.0]  # ankle pitch
        # A1 is omitted, only 1 DOF ankles

    def control_foot_position(self, x, y, h, side):
        """
        Compute joint angles given the desired foot position (x, y, h).
        """
        k = math.sqrt(x * x + (y * y + h * h))
        if k > self.LEG_LENGTH:
            k = self.LEG_LENGTH

        if abs(k) < 1e-8:
            alpha = 0.0
        else:
            alpha = math.asin(x / k)

        cval = k / self.LEG_LENGTH
        cval = max(min(cval, 1.0), -1.0)
        gamma = math.acos(cval)

        self.K0[side] = gamma + alpha      # hip pitch
        self.H[side]  = 2.0 * gamma        # knee
        self.A0[side] = gamma - alpha      # ankle pitch

        if abs(h) < 1e-8:
            hip_roll = 0.0
        else:
            hip_roll = math.atan(y / h)
        self.K1[side] = hip_roll

    def update_gait(self):
        """
        Update the internal state machine and foot positions each timestep.
        """
        if self.gait_phase == 0:
            # Ramping down from initial_leg_height to nominal_leg_height
            if self.initial_leg_height > self.nominal_leg_height + 0.1:
                self.initial_leg_height -= 1.0
            else:
                self.gait_phase = 10

            # Keep both feet together
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.initial_leg_height, 0)
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.initial_leg_height, 1)

        elif self.gait_phase == 10:
            # Idle
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.nominal_leg_height, 0)
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.nominal_leg_height, 1)
            if self.walking_enabled:
                self.step_length = 20.0
                self.gait_phase = 20

        elif self.gait_phase in [20, 30]:
            # Main stepping logic
            lateral_shift = self.lateral_foot_shift * math.sin(
                math.pi * self.step_cycle_counter / self.step_cycle_length
            )
            if self.stance_foot_index == 0:
                self.lateral_offset = lateral_shift
            else:
                self.lateral_offset = -lateral_shift

            half_cycle = self.step_cycle_length / 2.0
            if self.step_cycle_counter < half_cycle:
                fraction = self.step_cycle_counter / self.step_cycle_length
                self.forward_offset[self.stance_foot_index] = (
                    self.previous_stance_foot_offset * (1.0 - 2.0 * fraction)
                )
            else:
                fraction = 2.0 * self.step_cycle_counter / self.step_cycle_length - 1.0
                self.forward_offset[self.stance_foot_index] = -(
                    self.step_length - self.accumulated_forward_offset
                ) * fraction

            if self.gait_phase == 20:
                if self.step_cycle_counter < (self.double_support_fraction * self.step_cycle_length):
                    self.forward_offset[self.stance_foot_index ^ 1] = (
                        self.previous_swing_foot_offset
                        - (self.previous_stance_foot_offset
                           - self.forward_offset[self.stance_foot_index])
                    )
                else:
                    self.previous_swing_foot_offset = self.forward_offset[self.stance_foot_index ^ 1]
                    self.gait_phase = 30

            if self.gait_phase == 30:
                start_swing = int(self.double_support_fraction * self.step_cycle_length)
                denom = (1.0 - self.double_support_fraction) * self.step_cycle_length
                if denom < 1e-8:
                    denom = 1.0
                frac = (
                    -math.cos(math.pi * (self.step_cycle_counter - start_swing) / denom) + 1.0
                ) / 2.0
                self.forward_offset[self.stance_foot_index ^ 1] = (
                    self.previous_swing_foot_offset
                    + frac * (self.step_length - self.accumulated_forward_offset
                              - self.previous_swing_foot_offset)
                )

            i = int(self.double_support_fraction * self.step_cycle_length)
            if self.step_cycle_counter > i:
                self.current_foot_lift = self.max_foot_lift * math.sin(
                    math.pi * (self.step_cycle_counter - i) / (self.step_cycle_length - i)
                )
            else:
                self.current_foot_lift = 0.0

            if self.stance_foot_index == 0:
                # left foot = stance
                self.control_foot_position(
                    self.forward_offset[0] - self.hip_forward_offset,
                    -self.lateral_offset + 1.0,
                    self.nominal_leg_height,
                    0
                )
                self.control_foot_position(
                    self.forward_offset[1] - self.hip_forward_offset,
                    self.lateral_offset + 1.0,
                    self.nominal_leg_height - self.current_foot_lift,
                    1
                )
            else:
                # right foot = stance
                self.control_foot_position(
                    self.forward_offset[0] - self.hip_forward_offset,
                    -self.lateral_offset + 1.0,
                    self.nominal_leg_height - self.current_foot_lift,
                    0
                )
                self.control_foot_position(
                    self.forward_offset[1] - self.hip_forward_offset,
                    self.lateral_offset + 1.0,
                    self.nominal_leg_height,
                    1
                )

            if self.step_cycle_counter >= self.step_cycle_length:
                self.stance_foot_index ^= 1
                self.step_cycle_counter = 1
                self.accumulated_forward_offset = 0.0
                self.previous_stance_foot_offset = self.forward_offset[self.stance_foot_index]
                self.previous_swing_foot_offset = self.forward_offset[self.stance_foot_index ^ 1]
                self.current_foot_lift = 0.0
                self.gait_phase = 20
            else:
                self.step_cycle_counter += 1

    def get_joint_angles(self):
        """
        Return a dictionary with all the joint angles in radians. 
        For each joint name, we store the needed angle in radians.
        """
        angles = {}
        angles["left_hip_yaw"]  = 0.0
        angles["right_hip_yaw"] = 0.0

        angles["left_hip_roll"]  = self.K1[0]
        angles["left_hip_pitch"] = -self.K0[0]
        angles["left_knee"]      = self.H[0]
        angles["left_ankle"]     = self.A0[0]

        angles["right_hip_roll"]  = self.K1[1]
        angles["right_hip_pitch"] = self.K0[1]
        angles["right_knee"]      = -self.H[1]
        angles["right_ankle"]     = -self.A0[1]

        # Arms & others as placeholders:
        angles["left_shoulder_yaw"]   = 0.0
        angles["left_shoulder_pitch"] = 0.0
        angles["left_elbow"]          = 0.0
        angles["left_gripper"]        = 0.0

        angles["right_shoulder_yaw"]   = 0.0
        angles["right_shoulder_pitch"] = 0.0
        angles["right_elbow"]          = 0.0
        angles["right_gripper"]        = 0.0

        return angles


joint_to_actuator_id = {
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
    "right_ankle": 45,
}


def angles_to_pykos_commands(angles_dict):
    """
    Convert each angle (in radians) to a pykos command dictionary.
    Here we do a naive 1 rad -> ~57 deg. 
    In reality, you'll likely need gear ratio, zero offset, sign flips, etc.
    """
    cmds = []
    for joint_name, angle_radians in angles_dict.items():
        if joint_name not in joint_to_actuator_id:
            continue
        actuator_id = joint_to_actuator_id[joint_name]
        # Example: convert radians to degrees
        angle_degrees = math.degrees(angle_radians)
        if actuator_id in [32]:
            angle_degrees = -angle_degrees

        # You might need an offset or gear ratio here:
        # position_counts = angle_degrees * (some_gain)

        # For demonstration, let's just send angle_degrees as the position:
        cmds.append({
            "actuator_id": actuator_id,
            "position": angle_degrees
        })
    return cmds


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.42.1",
                        help="IP for the KOS device, default=192.168.42.1")
    parser.add_argument("--mjcf_name", type=str, default="zbot-v2",
                        help="Name of the Mujoco model in the K-Scale API (optional).")
    args = parser.parse_args()

    # Create our biped controller
    controller = BipedController()

    # If --mjcf_name is provided, set up a MujocoPuppet
    puppet = None
    if args.mjcf_name:
        puppet = MujocoPuppet(args.mjcf_name)

    # Connect to KOS
    async with KOS(ip=args.ip) as kos:
        for actuator_id in joint_to_actuator_id.values():
            print(f"Enabling torque for actuator {actuator_id}")
            await kos.actuator.configure_actuator(actuator_id=actuator_id, kp=40, kd=30, torque_enabled=True)

        # Optionally initialize to 0 position
        for actuator_id in joint_to_actuator_id.values():
            print(f"Setting actuator {actuator_id} to 0 position")
            await kos.actuator.command_actuators([{"actuator_id": actuator_id, "position": 0}])

        # Countdown before starting movement
        for i in range(5, 0, -1):
            print(f"Starting in {i}...")
            await asyncio.sleep(1)

        dt = 0.001

        # ↓ Here we add the counters to measure commands per second
        commands_sent = 0
        start_time = time.time()

        try:
            while True:
                if i >= 1000:
                    i = 0
                i += 1
                # 1) Update the gait state machine
                start_time_gait = time.perf_counter()
                controller.update_gait()
                end_time_gait = time.perf_counter()
                gait_update_duration = end_time_gait - start_time_gait
                print(f"Gait update took {gait_update_duration * 1000:.3f} ms")

                # 2) Retrieve angles from the BipedController
                angles_dict = controller.get_joint_angles()

                # 3) Convert angles to PyKOS commands and send to the real robot
                commands = angles_to_pykos_commands(angles_dict)
                if commands:
                    await kos.actuator.command_actuators(commands)

                # 4) Also send the same angles to MuJoCo puppet, if available
                # start_time_puppet = time.perf_counter()
                # if puppet is not None:
                #     await puppet.set_joint_angles(angles_dict)
                # end_time_puppet = time.perf_counter()
                # puppet_update_duration = end_time_puppet - start_time_puppet
                # print(f"Puppet update took {puppet_update_duration * 1000:.3f} ms")

                # Count how many loops (or 'commands sent') per second
                commands_sent += 1
                current_time = time.time()
                if current_time - start_time >= 1.0:
                    print(f"Commands per second (CPS): {commands_sent}")
                    commands_sent = 0
                    start_time = current_time

                await asyncio.sleep(dt + 0.00)
                # await asyncio.sleep(dt)

        except KeyboardInterrupt:
            print("\nShutting down gracefully.")
            # Optionally disable torque at the end if desired
            # for actuator_id in joint_to_actuator_id.values():
            #     await kos.actuator.configure_actuator(actuator_id=actuator_id, torque_enabled=False)


if __name__ == "__main__":
    asyncio.run(main())
