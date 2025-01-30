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

    The footCont(...) function sets these angles for left/right legs.
    The walk() function calls footCont(...) with the correct x,y,h
    to produce the desired stepping pattern.
    """

    def __init__(self):
        # -----------
        # Gait params
        # -----------
        self.LEG = 180.0  # mm, from the original #define LEG 180
        self.adjFR = 2.04  # forward offset for foot, from original "adjFR"
        self.autoH = 170.0  # nominal "height" from hip to foot
        self.autoHs = 180.0  # used for initial transitions
        self.mode = 0       # gait state: 0-> stand-up, 10-> idle, 20-> stance, 30-> swing
        self.walkF = True   # set this True to start walking from idle

        # -----------
        # Variables for cyclical stepping
        # -----------
        self.jikuasi = 0  # 0 or 1 (which foot is stance foot)
        self.fwctEnd = 48
        self.fwct = 0
        self.swf = 12       # how far to shift left/right
        self.fhMax = 20     # how high to lift foot
        self.landRate = 0.2 # fraction of cycle for double support
        self.fh = 0.0

        # forward and lateral offsets
        self.dx = [0.0, 0.0]  # dx[0]: left foot forward offset, dx[1]: right foot
        self.dxi = 0.0        # integrated forward correction
        self.fwr0 = 0.0
        self.fwr1 = 0.0
        self.fw = 20.0  # step length (set when we start walking)

        self.dy = 0.0
        self.dyi = 0.0

        # here we skip any sensor-based "UVC" approach:
        self.pitch = 0.0
        self.roll = 0.0

        # The joint angle arrays for each foot (0 = left, 1 = right)
        self.K0 = [0.0, 0.0]  # hip pitch
        self.K1 = [0.0, 0.0]  # hip roll
        self.H  = [0.0, 0.0]  # knee
        self.A0 = [0.0, 0.0]  # ankle pitch
        # A1 is omitted, because your URDF only has 1 ankle joint

    def footCont(self, x, y, h, side):
        """
        Replicates the 'footCont(...)' function in the C++ code.
        side = 0 -> left foot, side = 1 -> right foot.

        Inputs:
          x (mm): forward offset relative to some center
          y (mm): lateral offset (+ => foot out to the right, - => out to the left)
          h (mm): 'height' from hip axis to foot
        """
        # 1. Distance from “ankle roll axis” to “hip roll axis” in the 2D plane
        #    C++ does: k = sqrt(x^2 + (y^2 + h^2))
        k = math.sqrt(x*x + (y*y + h*h))
        if k > self.LEG:
            k = self.LEG  # clamp to avoid domain errors in acos

        # 2. foot pitch angle
        #    x' = asin(x/k)
        if abs(k) < 1e-8:
            alpha = 0.0
        else:
            alpha = math.asin(x / k)

        # 3. knee/hip geometry
        #    k' = acos(k / LEG)
        cval = k / self.LEG
        if cval > 1.0:
            cval = 1.0
        elif cval < -1.0:
            cval = -1.0
        gamma = math.acos(cval)

        # 4. Fill in:
        #    K0W[side] = gamma + alpha   (hip pitch)
        #    HW[side]  = gamma*2        (knee)
        #    A0W[side] = gamma - alpha  (ankle pitch)
        #    K1W[side] = atan(y/h)      (hip roll)
        #    A1W[side] = -K1W[side]     (unused; no ankle roll in your URDF)
        self.K0[side] = gamma + alpha      # hip pitch
        self.H[side]  = 2.0 * gamma        # knee
        self.A0[side] = gamma - alpha      # ankle pitch

        if abs(h) < 1e-8:
            hr = 0.0
        else:
            hr = math.atan(y / h)         # hip roll

        # "left_hip_roll" or "right_hip_roll" = K1W
        # sign changes can be tested if your foot inverts
        self.K1[side] = hr

    def walk(self):
        """
          mode=0 => ramp from autoHs -> autoH
          mode=10 => idle
          mode=20,30 => stepping
        """
        # Quick 1-step approach: We do not use force sensors or body angles
        # so skip the advanced "UVC" steps. We'll just replicate the foot
        # trajectory.

        if self.mode == 0:
            # Move from autoHs down to autoH
            if self.autoHs > self.autoH + 0.1:
                self.autoHs -= 1.0  # you can tune this
            else:
                self.mode = 10

            # Keep both feet in the same place (like stand)
            self.footCont(-self.adjFR, 0.0, self.autoHs, 0)
            self.footCont(-self.adjFR, 0.0, self.autoHs, 1)

        elif self.mode == 10:
            # Idle
            self.K0[0] = 0.0
            self.K0[1] = 0.0
            self.footCont(-self.adjFR, 0.0, self.autoH, 0)
            self.footCont(-self.adjFR, 0.0, self.autoH, 1)

            # If user wants walking, we proceed
            if self.walkF:
                self.fw = 20.0
                self.mode = 20

        elif self.mode in [20, 30]:
            # Main stepping logic (simplified from C++).
            # 1) Lateral shift with a sine wave
            k = self.swf * math.sin(math.pi * self.fwct / self.fwctEnd)
            if self.jikuasi == 0:
                # stance foot = left, swing foot = right
                self.dy = k  # shift to the right
            else:
                # stance foot = right, swing foot = left
                self.dy = -k  # shift to the left

            # 2) stance foot forward offset
            #    if fwct < fwctEnd/2 => one formula,
            #    else => another formula
            half_cycle = self.fwctEnd / 2.0
            if self.fwct < half_cycle:
                # first half of stance
                self.dx[self.jikuasi] = self.fwr0 * (1.0 - 2.0 * (self.fwct/self.fwctEnd))
            else:
                # second half
                self.dx[self.jikuasi] = -(self.fw - self.dxi)*(2.0*self.fwct/self.fwctEnd - 1.0)

            # 3) If mode=20 => "both feet contact" portion, then switch to mode=30 => "swing foot up"
            if self.mode == 20:
                if self.fwct < (self.landRate * self.fwctEnd):
                    # we do a partial shift
                    self.dx[self.jikuasi ^ 1] = self.fwr1 - (self.fwr0 - self.dx[self.jikuasi])
                else:
                    # switch to 30
                    self.fwr1 = self.dx[self.jikuasi ^ 1]
                    self.mode = 30

            # 4) If mode=30 => do a front swing of the other foot
            if self.mode == 30:
                start_swing = int(self.landRate * self.fwctEnd)
                denom = (1.0 - self.landRate) * self.fwctEnd
                if denom < 1e-8:
                    denom = 1.0
                # fraction from 0 to 1
                frac = (
                    -math.cos(
                        math.pi * (self.fwct - start_swing) / denom
                    ) + 1.0
                ) / 2.0
                self.dx[self.jikuasi ^ 1] = self.fwr1 + frac * (self.fw - self.dxi - self.fwr1)

            # 5) foot-lift
            i = int(self.landRate * self.fwctEnd)
            if self.fwct > i:
                # half-sine for foot-lift
                self.fh = self.fhMax * math.sin(
                    math.pi * (self.fwct - i) / (self.fwctEnd - i)
                )
            else:
                self.fh = 0.0

            # 6) call footCont() for left foot, right foot
            #    depending on which foot is stance. Example from c++:
            if self.jikuasi == 0:
                # left foot = stance, right foot = swing
                self.footCont(self.dx[0] - self.adjFR, -self.dy + 1.0, self.autoH, 0)
                self.footCont(self.dx[1] - self.adjFR,  self.dy + 1.0, self.autoH - self.fh, 1)
            else:
                # right foot = stance, left foot = swing
                self.footCont(self.dx[0] - self.adjFR, -self.dy + 1.0, self.autoH - self.fh, 0)
                self.footCont(self.dx[1] - self.adjFR,  self.dy + 1.0, self.autoH, 1)

            # 7) update cycle
            if self.fwct >= self.fwctEnd:
                self.jikuasi ^= 1  # toggle stance foot
                self.fwct = 1
                self.dxi = 0.0
                self.fwr0 = self.dx[self.jikuasi]
                self.fwr1 = self.dx[self.jikuasi ^ 1]
                self.fh = 0.0
                self.mode = 20

            else:
                self.fwct += 1

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

        # From footCont arrays:
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
            controller.walk()

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
