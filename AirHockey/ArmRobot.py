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
import json


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
        self.link2.pin = 27
        self.link3.pin = 30

        self.lookup_table = json.load(open("lookup_table.json", "r"))

        self.ser = serial.Serial('COM4', 115200)


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

if __name__ == "__main__":
    arm = ArmRobot()
    print(arm.forward_kinematics())
    arm.link1.moveJoint(0)
    # arm.write_servos()
    # lookup = {}
    # angle = -90
    # bar_width = 18
    # x = -bar_width
    # y = 13*2.54/100
    # for x in range(-15,16):
    #     for theta in range(-90,91,5):
    #         arm.algebraic_inverse_kinematics([x*2.54/100,y],theta*pi/180)
    #         lookup[f"{(x,theta)}"] = [arm.link1.theta, arm.link2.theta, arm.link3.theta]
    # with open("back_lookup_table.json", "w") as json_file:
    #     json.dump(lookup, json_file, indent=4)
    