import argparse
import asyncio
import math
import time

from loguru import logger  # High-performance logging library

from pykos import KOS
from digital_twin.puppet.mujoco_puppet import MujocoPuppet


class BipedController:
    """
    Bipedal walking controller using a gait state machine with virtual balance.
    """
    def __init__(self, lateral_movement_enabled=True):
        self.lateral_movement_enabled = lateral_movement_enabled

        self.roll_offset = math.radians(-2)
        self.hip_pitch_offset = math.radians(1)

        # Gait parameters
        self.LEG_LENGTH = 180.0      # mm
        self.hip_forward_offset = 2.04
        self.nominal_leg_height = 170.0
        self.initial_leg_height = 180.0
        self.gait_phase = 0
        self.walking_enabled = True

        # Stepping variables
        self.stance_foot_index = 0
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
        self.step_length = 15.0

        self.lateral_offset = 0.0
        self.dyi = 0.0
        self.pitch = 0.0
        self.roll  = 0.0

        # Joint angle arrays
        self.K0 = [0.0, 0.0]  # hip pitch
        self.K1 = [0.0, 0.0]  # hip roll
        self.H  = [0.0, 0.0]  # knee
        self.A0 = [0.0, 0.0]  # ankle pitch

    def control_foot_position(self, x, y, h, side):
        """
        Compute joint angles given the desired foot position (x, y, h).
        """
        k = math.sqrt(x*x + (y*y + h*h))
        k = min(k, self.LEG_LENGTH)
        alpha = math.asin(x/k) if abs(k) > 1e-8 else 0.0

        cval = max(min(k/self.LEG_LENGTH, 1.0), -1.0)
        gamma = math.acos(cval)

        self.K0[side] = gamma + alpha      # hip pitch
        self.H[side]  = 2.0 * gamma          # knee
        self.A0[side] = gamma - alpha        # ankle pitch

        hip_roll = math.atan(y/h) if abs(h) > 1e-8 else 0.0
        self.K1[side] = hip_roll + self.roll_offset

    def virtual_balance_adjustment(self):
        """
        Compute a virtual CoM based on current foot positions and adjust lateral_offset.
        """
        left_foot_x  = self.forward_offset[0] - self.hip_forward_offset
        left_foot_y  = -self.lateral_offset + 1.0
        right_foot_x = self.forward_offset[1] - self.hip_forward_offset
        right_foot_y = self.lateral_offset + 1.0

        # Compute center-of-mass (CoM)
        com_x = (left_foot_x + right_foot_x) / 2.0
        com_y = (left_foot_y + right_foot_y) / 2.0

        desired_com_y = 1.0
        error_y = desired_com_y - com_y

        feedback_gain = 0.1
        adjustment = feedback_gain * error_y
        self.lateral_offset += adjustment

    def update_gait(self):
        """
        Update the internal gait state and compute new joint targets.
        """
        if self.gait_phase == 0:
            # Ramping down leg height
            if self.initial_leg_height > self.nominal_leg_height + 0.1:
                self.initial_leg_height -= 1.0
            else:
                self.gait_phase = 10

            # Both feet together
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.initial_leg_height, 0)
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.initial_leg_height, 1)

        elif self.gait_phase == 10:
            # Idle phase
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.nominal_leg_height, 0)
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.nominal_leg_height, 1)
            if self.walking_enabled:
                self.step_length = 20.0
                self.gait_phase = 20

        elif self.gait_phase in [20, 30]:
            if self.lateral_movement_enabled:
                lateral_shift = self.lateral_foot_shift * math.sin(
                    math.pi * self.step_cycle_counter / self.step_cycle_length
                )
                self.lateral_offset = lateral_shift if self.stance_foot_index == 0 else -lateral_shift
                self.virtual_balance_adjustment()
            else:
                self.lateral_offset = 0.0

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
                        - (self.previous_stance_foot_offset - self.forward_offset[self.stance_foot_index])
                    )
                else:
                    self.previous_swing_foot_offset = self.forward_offset[self.stance_foot_index ^ 1]
                    self.gait_phase = 30

            elif self.gait_phase == 30:
                start_swing = int(self.double_support_fraction * self.step_cycle_length)
                denom = (1.0 - self.double_support_fraction) * self.step_cycle_length or 1.0
                frac = (-math.cos(math.pi * (self.step_cycle_counter - start_swing) / denom) + 1.0) / 2.0
                self.forward_offset[self.stance_foot_index ^ 1] = (
                    self.previous_swing_foot_offset
                    + frac * (self.step_length - self.accumulated_forward_offset - self.previous_swing_foot_offset)
                )

            i = int(self.double_support_fraction * self.step_cycle_length)
            if self.step_cycle_counter > i:
                self.current_foot_lift = self.max_foot_lift * math.sin(
                    math.pi * (self.step_cycle_counter - i) / (self.step_cycle_length - i)
                )
            else:
                self.current_foot_lift = 0.0

            # Set foot positions based on the stance index.
            if self.stance_foot_index == 0:
                # left foot is stance
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
                # right foot is stance
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
        Return a dictionary mapping joint names to angles (radians).
        """
        angles = {
            "left_hip_yaw": 0.0,
            "right_hip_yaw": 0.0,
            "left_hip_roll": self.K1[0],
            "left_hip_pitch": -self.K0[0] - self.hip_pitch_offset,
            "left_knee": self.H[0],
            "left_ankle": self.A0[0],
            "right_hip_roll": self.K1[1],
            "right_hip_pitch": self.K0[1] + self.hip_pitch_offset,
            "right_knee": -self.H[1],
            "right_ankle": -self.A0[1],
            # Arms & additional joints (placeholders)
            "left_shoulder_yaw": 0.0,
            "left_shoulder_pitch": 0.0,
            "left_elbow": 0.0,
            "left_gripper": 0.0,
            "right_shoulder_yaw": 0.0,
            "right_shoulder_pitch": 0.0,
            "right_elbow": 0.0,
            "right_gripper": 0.0,
        }
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
    cmds = []
    for joint_name, angle_rad in angles_dict.items():
        if joint_name not in joint_to_actuator_id:
            continue
        actuator_id = joint_to_actuator_id[joint_name]
        angle_deg = math.degrees(angle_rad)
        # Sample special treatment for certain actuators:
        if actuator_id in [32]:
            angle_deg = -angle_deg
        cmds.append({
            "actuator_id": actuator_id,
            "position": angle_deg
        })
    return cmds


class RobotInterface:
    def __init__(self, ip):
        self.ip = ip

    async def __aenter__(self):
        self.kos = KOS(ip=self.ip)
        await self.kos.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self.kos.__aexit__(*args)

    async def configure_actuators(self):
        for actuator_id in joint_to_actuator_id.values():
            logger.info(f"Enabling torque for actuator {actuator_id}")
            await self.kos.actuator.configure_actuator(
                actuator_id=actuator_id, kp=32, kd=32, max_torque=80, torque_enabled=True
            )

    async def zero_actuators(self):
        for actuator_id in joint_to_actuator_id.values():
            logger.info(f"Setting actuator {actuator_id} to 0 position")
            await self.kos.actuator.command_actuators([{"actuator_id": actuator_id, "position": 0}])

    async def send_commands(self, commands):
        if commands:
            await self.kos.actuator.command_actuators(commands)


async def run_robot_mode(ip_address, controller, puppet):
    dt = 0.001

    async with RobotInterface(ip=ip_address) as robot:
        await robot.configure_actuators()
        await robot.zero_actuators()

        for i in range(5, 0, -1):
            logger.info(f"Starting in {i}...")
            await asyncio.sleep(1)

        commands_sent = 0
        start_time = time.time()
        i = 0
        try:
            while True:
                i = 0 if i >= 1000 else i + 1
                tic = time.perf_counter()
                controller.update_gait()
                toc = time.perf_counter()
                # logger.debug(f"Gait update took {(toc-tic)*1000:.3f} ms")

                angles = controller.get_joint_angles()
                commands = angles_to_pykos_commands(angles)
                await robot.send_commands(commands)

                # Update puppet if available
                if puppet is not None:
                    await puppet.set_joint_angles(angles)

                commands_sent += 1
                if time.time() - start_time >= 1.0:
                    logger.info(f"Commands per second (CPS): {commands_sent}")
                    commands_sent = 0
                    start_time = time.time()

                await asyncio.sleep(dt)
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully.")


async def run_simulation_mode(controller, puppet):
    dt = 0.001
    commands_sent = 0
    start_time = time.time()
    i = 0

    try:
        while True:
            i = 0 if i >= 1000 else i + 1
            controller.update_gait()
            angles = controller.get_joint_angles()
            if puppet is not None:
                await puppet.set_joint_angles(angles)
            commands_sent += 1
            if time.time() - start_time >= 1.0:
                # Uncomment below if you want to log simulation CPS
                # logger.info(f"Simulated Commands per second (CPS): {commands_sent}")
                commands_sent = 0
                start_time = time.time()
            await asyncio.sleep(dt)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pykos", action="store_true",
                        help="Use PyKOS to command actual actuators.")
    parser.add_argument("--mujoco", action="store_true",
                        help="Also send commands to Mujoco simulation.")

    args = parser.parse_args()

    ip_address = "192.168.42.1"
    mjcf_name = "zbot-v2"

    controller = BipedController(lateral_movement_enabled=True)

    puppet = MujocoPuppet(mjcf_name) if args.mujoco else None

    if args.pykos:
        await run_robot_mode(ip_address, controller, puppet)
    else:
        logger.info("Running in simulation-only mode (PyKOS commands are disabled).")
        await run_simulation_mode(controller, puppet)


if __name__ == "__main__":
    asyncio.run(main())
