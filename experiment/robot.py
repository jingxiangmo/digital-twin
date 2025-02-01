import subprocess
import logging
import asyncio
from pykos import KOS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JOINT_TO_ACTUATOR_ID = {
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

class Robot:
    def __init__(self):
        pass

    async def initialize(self):
        logger.info("Initializing robot...")

        asyncio.run(self.check_connection())
        asyncio.run(self.setup_actuators())
        asyncio.run(self.go_initial_position())

        logger.info("Robot initialized.")
    
    async def check_connection(self) -> None:
        """Checks the connection to the robot."""
        logger.info("Checking connection to robot...")
        try:
            subprocess.run(
                ["ping", "-c", "1", "192.168.42.1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            logger.info("Successfully pinged robot.")
        except subprocess.CalledProcessError:
            logger.error("Could not ping robot at 192.168.42.1")
            raise ConnectionError("Robot connection failed.")
        
    async def setup_actuators(self) -> None:
        """Sets up the servos."""
        logger.info("Setting up servos...")
        try:
            async with KOS(ip="192.168.42.1") as kos:
                for actuator_id in JOINT_TO_ACTUATOR_ID.values():
                    await kos.actuator.configure_actuator(actuator_id=actuator_id, kp=32, kd=32, torque_enabled=True)
        except Exception as e:
            logger.error(f"Error setting up servos: {e}")
            raise e
    
    async def go_initial_position(self) -> None:
        """Goes to the initial position."""
        logger.info("Going to initial position...")
