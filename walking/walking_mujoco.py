import argparse
import asyncio
import math

from digital_twin.puppet.mujoco_puppet import MujocoPuppet

class BipedController:
    """
    References:
      - K0W[s] -> hip pitch
      - K1W[s] -> hip roll
      - HW[s]  -> knee
      - A0W[s] -> ankle pitch
      - A1W[s] -> ankle roll (unused here, since only 1 DOF at the ankle)

    The control_foot_position(...) function sets these angles for left/right legs.
    The update_gait() function calls control_foot_position(...) with the correct x, y, h
    to produce the desired stepping pattern.
    """

    def __init__(self):
        # -----------
        # Gait params
        # -----------
        self.LEG_LENGTH = 180.0      # mm, from the original #define LEG 180
        self.hip_forward_offset = 2.04  # forward offset for foot, from original "adjFR"
        self.nominal_leg_height = 170.0   # nominal "height" from hip to foot
        self.initial_leg_height = 180.0   # used for initial transitions
        self.gait_phase = 0              # gait state: 0-> stand-up, 10-> idle, 20-> stance, 30-> swing
        self.walking_enabled = True      # set this True to start walking from idle

        # -----------
        # Variables for cyclical stepping
        # -----------
        self.stance_foot_index = 0  # 0 or 1 (which foot is stance foot)
        self.step_cycle_length = 48
        self.step_cycle_counter = 0
        self.lateral_foot_shift = 12     # how far to shift left/right
        self.max_foot_lift = 20          # how high to lift foot
        self.double_support_fraction = 0.2 # fraction of cycle for double support
        self.current_foot_lift = 0.0

        # forward and lateral offsets
        self.forward_offset = [0.0, 0.0]  # forward_offset[0]: left foot forward offset, forward_offset[1]: right foot
        self.accumulated_forward_offset = 0.0
        self.previous_stance_foot_offset = 0.0
        self.previous_swing_foot_offset = 0.0
        self.step_length = 20.0  # step length (set when we start walking)

        self.lateral_offset = 0.0
        self.dyi = 0.0  # not actively used in this code snippet

        # here we skip any sensor-based "UVC" approach:
        self.pitch = 0.0
        self.roll = 0.0

        # The joint angle arrays for each foot (0 = left, 1 = right)
        self.K0 = [0.0, 0.0]  # hip pitch
        self.K1 = [0.0, 0.0]  # hip roll
        self.H  = [0.0, 0.0]  # knee
        self.A0 = [0.0, 0.0]  # ankle pitch
        # A1 is omitted, because the URDF only has 1 ankle roll joint

    def control_foot_position(self, x, y, h, side):
        """
        Replicates the 'footCont(...)' function in the C++ code, with improved naming.
        side = 0 -> left foot, side = 1 -> right foot.

        Inputs:
          x (mm): forward offset relative to some center
          y (mm): lateral offset (+ => foot out to the right, - => out to the left)
          h (mm): 'height' from hip axis to foot
        """
        # 1. Distance from "ankle roll axis" to "hip roll axis" in the 2D plane
        k = math.sqrt(x*x + (y*y + h*h))
        if k > self.LEG_LENGTH:
            k = self.LEG_LENGTH  # clamp to avoid domain errors in acos

        # 2. foot pitch angle (x' = asin(x/k))
        if abs(k) < 1e-8:
            alpha = 0.0
        else:
            alpha = math.asin(x / k)

        # 3. knee/hip geometry (k' = acos(k / LEG_LENGTH))
        cval = k / self.LEG_LENGTH
        cval = max(min(cval, 1.0), -1.0)
        gamma = math.acos(cval)

        # 4. Fill in:
        #    K0W[side] = gamma + alpha   (hip pitch)
        #    HW[side]  = gamma*2        (knee)
        #    A0W[side] = gamma - alpha  (ankle pitch)
        #    K1W[side] = atan(y/h)      (hip roll)
        self.K0[side] = gamma + alpha      # hip pitch
        self.H[side]  = 2.0 * gamma        # knee
        self.A0[side] = gamma - alpha      # ankle pitch

        if abs(h) < 1e-8:
            hip_roll = 0.0
        else:
            hip_roll = math.atan(y / h)         # hip roll

        self.K1[side] = hip_roll

    def update_gait(self):
        """
          gait_phase=0 => ramp from initial_leg_height -> nominal_leg_height
          gait_phase=10 => idle
          gait_phase=20,30 => stepping
        """
        # Quick 1-step approach: We do not use force sensors or body angles
        # so skip the advanced "UVC" steps. We'll just replicate the foot trajectory.

        if self.gait_phase == 0:
            # Move from initial_leg_height down to nominal_leg_height
            if self.initial_leg_height > self.nominal_leg_height + 0.1:
                self.initial_leg_height -= 1.0  # you can tune this
            else:
                self.gait_phase = 10

            # Keep both feet in the same place (like stand)
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.initial_leg_height, 0)
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.initial_leg_height, 1)

        elif self.gait_phase == 10:
            # Idle
            self.K0[0] = 0.0
            self.K0[1] = 0.0
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.nominal_leg_height, 0)
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.nominal_leg_height, 1)

            # If user wants walking, we proceed
            if self.walking_enabled:
                self.step_length = 20.0
                self.gait_phase = 20

        elif self.gait_phase in [20, 30]:
            # Main stepping logic (simplified from C++).
            # 1) Lateral shift with a sine wave
            lateral_shift = self.lateral_foot_shift * math.sin(
                math.pi * self.step_cycle_counter / self.step_cycle_length
            )
            if self.stance_foot_index == 0:
                # stance foot = left, swing foot = right
                self.lateral_offset = lateral_shift  # shift to the right
            else:
                # stance foot = right, swing foot = left
                self.lateral_offset = -lateral_shift  # shift to the left

            # 2) stance foot forward offset
            half_cycle = self.step_cycle_length / 2.0
            if self.step_cycle_counter < half_cycle:
                # first half of stance
                fraction = self.step_cycle_counter / self.step_cycle_length
                self.forward_offset[self.stance_foot_index] = (
                    self.previous_stance_foot_offset * (1.0 - 2.0 * fraction)
                )
            else:
                # second half
                fraction = 2.0 * self.step_cycle_counter / self.step_cycle_length - 1.0
                self.forward_offset[self.stance_foot_index] = -(
                    self.step_length - self.accumulated_forward_offset
                ) * fraction

            # 3) If gait_phase=20 => "both feet contact" portion, then switch to gait_phase=30 => "swing foot up"
            if self.gait_phase == 20:
                if self.step_cycle_counter < (self.double_support_fraction * self.step_cycle_length):
                    # We do a partial shift
                    self.forward_offset[self.stance_foot_index ^ 1] = (
                        self.previous_swing_foot_offset 
                        - (self.previous_stance_foot_offset - 
                           self.forward_offset[self.stance_foot_index])
                    )
                else:
                    # switch to 30
                    self.previous_swing_foot_offset = self.forward_offset[self.stance_foot_index ^ 1]
                    self.gait_phase = 30

            # 4) If gait_phase=30 => do a front swing of the other foot
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
                    + frac * (self.step_length - self.accumulated_forward_offset - 
                              self.previous_swing_foot_offset)
                )

            # 5) foot-lift
            i = int(self.double_support_fraction * self.step_cycle_length)
            if self.step_cycle_counter > i:
                # half-sine for foot-lift
                self.current_foot_lift = self.max_foot_lift * math.sin(
                    math.pi * (self.step_cycle_counter - i) / (self.step_cycle_length - i)
                )
            else:
                self.current_foot_lift = 0.0

            # 6) call control_foot_position() for left foot, right foot
            if self.stance_foot_index == 0:
                # left foot = stance, right foot = swing
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
                # right foot = stance, left foot = swing
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

            # 7) update cycle
            if self.step_cycle_counter >= self.step_cycle_length:
                self.stance_foot_index ^= 1  # toggle stance foot
                self.step_cycle_counter = 1
                self.accumulated_forward_offset = 0.0
                self.previous_stance_foot_offset = self.forward_offset[self.stance_foot_index]
                self.previous_swing_foot_offset  = self.forward_offset[self.stance_foot_index ^ 1]
                self.current_foot_lift = 0.0
                self.gait_phase = 20
            else:
                self.step_cycle_counter += 1

        # else any other mode, do nothing

    def get_joint_angles(self):
        """
        Return a dictionary with ALL the joint angles in radians

        The relevant leg joints in your URDF are:
           - [left|right]_hip_yaw
           - [left|right]_hip_roll
           - [left|right]_hip_pitch
           - [left|right]_knee
           - [left|right]_ankle
        We also set arms & grippers to 0 as placeholders.
        """
        angles = {}

        # We do not use hip_yaw in this code, so set to 0:
        angles["left_hip_yaw"]  = 0.0
        angles["right_hip_yaw"] = 0.0

        # From control_foot_position arrays:
        # (The sign conventions remain as in the original code.)
        angles["left_hip_roll"]  = self.K1[0]
        angles["left_hip_pitch"] = -self.K0[0]
        angles["left_knee"]      = self.H[0]
        angles["left_ankle"]     = self.A0[0]

        angles["right_hip_roll"]  = self.K1[1]
        angles["right_hip_pitch"] = self.K0[1]
        angles["right_knee"]      = -self.H[1]
        angles["right_ankle"]     = -self.A0[1]

        # You can keep arms at 0 or do some default
        angles["left_shoulder_yaw"]   = 0.0
        angles["left_shoulder_pitch"] = 0.0
        angles["left_elbow"]          = 0.0
        angles["left_gripper"]        = 0.0

        angles["right_shoulder_yaw"]   = 0.0
        angles["right_shoulder_pitch"] = 0.0
        angles["right_elbow"]          = 0.0
        angles["right_gripper"]        = 0.0

        return angles


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-skip-root", action="store_true", 
                        help="If set, do not skip the root joint in set_joint_angles().")
    args = parser.parse_args()


    # Create the MuJoCo puppet
    puppet = MujocoPuppet("zbot-v2")

    # Retrieve the full list of joints from the environment
    joint_names = await puppet.get_joint_names()

    # Often we skip index [0] if it's a 'root' floating joint
    if not args.no_skip_root and len(joint_names) > 1 and joint_names[0] == "root":
        joint_names = joint_names[1:]

    # Create our biped controller
    controller = BipedController()

    # A simple time stepping loop
    dt = 0.01
    try:
        while True:
            # 1) Update the gait state machine
            controller.update_gait()

            # 2) Retrieve the newly computed angles
            angles_dict = controller.get_joint_angles()

            # 3) Send angles down to MuJoCo
            #    (The puppet might filter unknown joint names internally.)
            await puppet.set_joint_angles(angles_dict)

            # Sleep for dt seconds
            await asyncio.sleep(dt)

    except KeyboardInterrupt:
        print("\nShutting down gracefully.")


if __name__ == "__main__":
    asyncio.run(main())
