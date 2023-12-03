"""
File: ArmRobot.py

Description:This module defines the ArmRobot class, which represents an arm robot with multiple joints. 
It provides functionality to add frames to the arm, move individual joints, update the Denavit-Hartenberg 
(DH) table, and perform forward and inverse kinematics calculations.


Author: Ryan Barry
Date Created: October 5, 2023
"""

from ArmRobotKinematics import PRISMATIC, REVOLUTE, FIXED_TRANSLATION, FIXED_ROTATION, ArmRobotKinematics
from math import pi, degrees
import serial


LINK1_LENGTH = 11.25*2.54/100
LINK2_LENGTH = 11.5*2.54/100
WRIST_LENGTH = 1.5*2.54/100
class ArmRobot(ArmRobotKinematics):
    def __init__(self):
        super().__init__()
        '''
        Additional initialization code specific to the arm robot:
        This is where you will define the configuration of your robot with the addFrame method
        '''
        self.link1 = self.addFrame(joint_type=REVOLUTE, a=LINK1_LENGTH)
        self.link2 = self.addFrame(joint_type=REVOLUTE, a=LINK2_LENGTH)
        self.link3 = self.addFrame(joint_type=REVOLUTE, a=WRIST_LENGTH, min_lim=-pi/2, max_lim=pi/2)

        self.link1.pin = 29
        self.link2.pin = 30
        self.link3.pin = 31


        self.ser = serial.Serial('COM6', 115200)


    def write_servos(self):
        '''
        Writes the current joint angles to the servos
        '''
        for joint in self._frames:
            # Convert theta from radians to degrees
            theta_deg = degrees(joint.theta)

            if theta_deg > 180.0:
                theta_deg = theta_deg - 360.0
            # Map theta from [-135, 135] to [500, 2500] such that 0 maps to 1500
            pulse_width = ((theta_deg + 135) / 270) * 2000 + 500

            # Write the pulse width to the servo
            self.ser.write(f"#{joint.pin}P{int(pulse_width)}\r".encode())


        
if __name__ == '__main__':

    arm = ArmRobot()
    theta1 = 0
    arm.link1.moveJoint(theta1)
    arm.write_servos()
