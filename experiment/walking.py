import argparse
import asyncio
import math
import time

from loguru import logger

from pykos import KOS
from digital_twin.puppet.mujoco_puppet import MujocoPuppet


class BipedController:
    """
    Robotic bipedal walking controller using a gait state machine with virtual balance.
    """
    def __init__(self, enable_lateral_motion=True):
        self.enable_lateral_motion = enable_lateral_motion

        self.foot_roll_offset = math.radians(-2)
        self.hip_pitch_offset = math.radians(1)

        # Gait parameters (units in mm where applicable)
        self.leg_length_mm = 180.0
        self.hip_x_offset = 2.04
        self.nominal_leg_z = 170.0
        self.initial_leg_z = 180.0
        self.gait_state = 0
        self.enable_walking = True

        # Stepping variables
        self.stance_leg = 0
        self.step_cycle_ticks = 48
        self.step_cycle_count = 0
        self.foot_lateral_shift_mm = 12
        self.max_foot_clearance = 20
        self.double_support_ratio = 0.2
        self.current_foot_clearance = 0.0

        self.foot_x_offsets = [0.0, 0.0]
        self.accumulated_forward_displacement = 0.0
        self.prev_stance_foot_offset = 0.0
        self.prev_swing_foot_offset = 0.0
        self.step_length_mm = 15.0

        self.robot_lateral_offset = 0.0
        self.dynamic_inertia = 0.0  # (currently unused, could represent dynamic compensation)
        self.pitch = 0.0
        self.roll = 0.0

        # Joint angle arrays (in radians)
        self.hip_pitch_angles = [0.0, 0.0]  # hip pitch angles for left and right legs
        self.hip_roll_angles  = [0.0, 0.0]  # hip roll angles for left and right legs
        self.knee_angles      = [0.0, 0.0]  # knee angles for left and right legs
        self.ankle_angles     = [0.0, 0.0]  # ankle pitch angles for left and right legs

    def compute_leg_ik(self, x, y, z, leg_side):
        """
        Compute joint angles using inverse kinematics for the desired foot position (x, y, z).
        """
        distance = math.sqrt(x * x + (y * y + z * z))
        distance = min(distance, self.leg_length_mm)
        alpha = math.asin(x / distance) if abs(distance) > 1e-8 else 0.0

        ratio = max(min(distance / self.leg_length_mm, 1.0), -1.0)
        gamma = math.acos(ratio)

        self.hip_pitch_angles[leg_side] = gamma + alpha      # hip pitch
        self.knee_angles[leg_side]      = 2.0 * gamma          # knee
        self.ankle_angles[leg_side]     = gamma - alpha        # ankle pitch

        hip_roll = math.atan(y / z) if abs(z) > 1e-8 else 0.0
        self.hip_roll_angles[leg_side] = hip_roll + self.foot_roll_offset

    def apply_virtual_balance(self):
        """
        Compute a virtual center-of-mass (CoM) and update the lateral offset of the robot.
        """
        left_foot_x  = self.foot_x_offsets[0] - self.hip_x_offset
        left_foot_y  = -self.robot_lateral_offset + 1.0
        right_foot_x = self.foot_x_offsets[1] - self.hip_x_offset
        right_foot_y = self.robot_lateral_offset + 1.0

        # Compute center-of-mass (CoM)
        com_x = (left_foot_x + right_foot_x) / 2.0
        com_y = (left_foot_y + right_foot_y) / 2.0

        desired_com_y = 1.0
        error_y = desired_com_y - com_y

        feedback_gain = 0.1
        adjustment = feedback_gain * error_y
        self.robot_lateral_offset += adjustment

    def update_gait(self):
        """
        Update the current gait state and compute new joint targets.
        """
        if self.gait_state == 0:
            # Ramping down leg height to approach nominal stance
            if self.initial_leg_z > self.nominal_leg_z + 0.1:
                self.initial_leg_z -= 1.0
            else:
                self.gait_state = 10

            # Both feet are together in the initial position
            self.compute_leg_ik(-self.hip_x_offset, 0.0, self.initial_leg_z, 0)
            self.compute_leg_ik(-self.hip_x_offset, 0.0, self.initial_leg_z, 1)

        elif self.gait_state == 10:
            # Idle phase; maintain nominal leg height
            self.compute_leg_ik(-self.hip_x_offset, 0.0, self.nominal_leg_z, 0)
            self.compute_leg_ik(-self.hip_x_offset, 0.0, self.nominal_leg_z, 1)
            if self.enable_walking:
                self.step_length_mm = 20.0
                self.gait_state = 20

        elif self.gait_state in [20, 30]:
            if self.enable_lateral_motion:
                lateral_shift = self.foot_lateral_shift_mm * math.sin(
                    math.pi * self.step_cycle_count / self.step_cycle_ticks
                )
                self.robot_lateral_offset = lateral_shift if self.stance_leg == 0 else -lateral_shift
                self.apply_virtual_balance()
            else:
                self.robot_lateral_offset = 0.0

            half_cycle = self.step_cycle_ticks / 2.0
            if self.step_cycle_count < half_cycle:
                fraction = self.step_cycle_count / self.step_cycle_ticks
                self.foot_x_offsets[self.stance_leg] = (
                    self.prev_stance_foot_offset * (1.0 - 2.0 * fraction)
                )
            else:
                fraction = 2.0 * self.step_cycle_count / self.step_cycle_ticks - 1.0
                self.foot_x_offsets[self.stance_leg] = -(
                    self.step_length_mm - self.accumulated_forward_displacement
                ) * fraction

            if self.gait_state == 20:
                if self.step_cycle_count < (self.double_support_ratio * self.step_cycle_ticks):
                    self.foot_x_offsets[self.stance_leg ^ 1] = (
                        self.prev_swing_foot_offset
                        - (self.prev_stance_foot_offset - self.foot_x_offsets[self.stance_leg])
                    )
                else:
                    self.prev_swing_foot_offset = self.foot_x_offsets[self.stance_leg ^ 1]
                    self.gait_state = 30

            elif self.gait_state == 30:
                start_swing = int(self.double_support_ratio * self.step_cycle_ticks)
                denom = (1.0 - self.double_support_ratio) * self.step_cycle_ticks or 1.0
                frac = (-math.cos(math.pi * (self.step_cycle_count - start_swing) / denom) + 1.0) / 2.0
                self.foot_x_offsets[self.stance_leg ^ 1] = (
                    self.prev_swing_foot_offset
                    + frac * (self.step_length_mm - self.accumulated_forward_displacement - self.prev_swing_foot_offset)
                )

            transition_count = int(self.double_support_ratio * self.step_cycle_ticks)
            if self.step_cycle_count >= transition_count:
                self.current_foot_clearance = self.max_foot_clearance * math.sin(
                    math.pi * (self.step_cycle_count - transition_count) / (self.step_cycle_ticks - transition_count)
                )
            else:
                self.current_foot_clearance = 0.0

            # Set foot positions based on the active stance leg.
            if self.stance_leg == 0:
                # Left leg is the stance leg.
                self.compute_leg_ik(
                    self.foot_x_offsets[0] - self.hip_x_offset,
                    -self.robot_lateral_offset + 1.0,
                    self.nominal_leg_z,
                    0
                )
                self.compute_leg_ik(
                    self.foot_x_offsets[1] - self.hip_x_offset,
                    self.robot_lateral_offset + 1.0,
                    self.nominal_leg_z - self.current_foot_clearance,
                    1
                )
            else:
                # Right leg is the stance leg.
                self.compute_leg_ik(
                    self.foot_x_offsets[0] - self.hip_x_offset,
                    -self.robot_lateral_offset + 1.0,
                    self.nominal_leg_z - self.current_foot_clearance,
                    0
                )
                self.compute_leg_ik(
                    self.foot_x_offsets[1] - self.hip_x_offset,
                    self.robot_lateral_offset + 1.0,
                    self.nominal_leg_z,
                    1
                )

            if self.step_cycle_count >= self.step_cycle_ticks:
                self.stance_leg ^= 1
                self.step_cycle_count = 1
                self.accumulated_forward_displacement = 0.0
                self.prev_stance_foot_offset = self.foot_x_offsets[self.stance_leg]
                self.prev_swing_foot_offset = self.foot_x_offsets[self.stance_leg ^ 1]
                self.current_foot_clearance = 0.0
                self.gait_state = 20
            else:
                self.step_cycle_count += 1

    def get_joint_angles(self):
        """
        Return a dictionary mapping joint names to angles (radians).
        """
        angles = {
            "left_hip_yaw": 0.0,
            "right_hip_yaw": 0.0,
            "left_hip_roll": self.hip_roll_angles[0],
            "left_hip_pitch": -self.hip_pitch_angles[0] - self.hip_pitch_offset,
            "left_knee": self.knee_angles[0],
            "left_ankle": self.ankle_angles[0],
            "right_hip_roll": self.hip_roll_angles[1],
            "right_hip_pitch": self.hip_pitch_angles[1] + self.hip_pitch_offset,
            "right_knee": -self.knee_angles[1],
            "right_ankle": -self.ankle_angles[1],
            
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


JOINT_TO_ID = {
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

JOINT_OFFSET = {
    # Left arm
    "left_shoulder_yaw": 0.0,
    "left_shoulder_pitch": 0.0,
    "left_elbow_yaw": 0.0,
    "left_gripper": 0.0,
    # Right arm
    "right_shoulder_yaw": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_elbow_yaw": 0.0,
    "right_gripper": 0.0,
    # Left leg
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": 0.0,
    "left_knee": 0.0,
    "left_ankle": 0.0,
    # Right leg
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.0,
    "right_knee": 0.0,
    "right_ankle": 0.0,
}

def angles_to_pykos_commands(angles_dict):
    cmds = []
    for joint_name, angle_rad in angles_dict.items():
        if joint_name not in JOINT_TO_ID:
            continue
        actuator_id = JOINT_TO_ID[joint_name]
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
        for actuator_id in JOINT_TO_ID.values():
            logger.info(f"Enabling torque for actuator {actuator_id}")
            await self.kos.actuator.configure_actuator(
                actuator_id=actuator_id, kp=32, kd=32, max_torque=80, torque_enabled=True
            )

    async def zero_actuators(self):
        for actuator_id in JOINT_TO_ID.values():
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

    controller = BipedController(enable_lateral_motion=True)

    puppet = MujocoPuppet(mjcf_name) if args.mujoco else None

    if args.pykos:
        logger.info("Running in real mode...")
        await run_robot_mode(ip_address, controller, puppet)
    else:
        logger.info("Running in sim mode...")
        await run_simulation_mode(controller, puppet)


if __name__ == "__main__":
    asyncio.run(main())
