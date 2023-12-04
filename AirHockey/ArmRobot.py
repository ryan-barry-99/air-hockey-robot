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
import time


LINK1_LENGTH = 11.25*2.54/100
LINK2_LENGTH = 11.5*2.54/100
WRIST_LENGTH = 1.5*2.54/100
class ArmRobot(ArmRobotKinematics):
    def __init__(self, com=None):
        super().__init__()
        '''
        Additional initialization code specific to the arm robot:
        This is where you will define the configuration of your robot with the addFrame method
        '''
        self.link1 = self.addFrame(joint_type=REVOLUTE, a=LINK1_LENGTH, min_lim=-135*pi/180, max_lim=135*pi/180)
        self.link2 = self.addFrame(joint_type=REVOLUTE, a=LINK2_LENGTH, min_lim=-135*pi/180, max_lim=135*pi/180)
        self.link3 = self.addFrame(joint_type=REVOLUTE, a=WRIST_LENGTH, min_lim=-pi/2, max_lim=pi/2)

        self.link1.pin = 30
        self.link2.pin = 29
        self.link3.pin = 27

        self.lookup_table = json.load(open("lookup_table.json", "r"))
        
        if com is not None:
            self.ser = serial.Serial(com, 115200)


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
    arm = ArmRobot("COM4")
    theta = 15
    y = 0.8
    y = int(42*y - 21)
    if y < -13:
        y = -13
    elif y > 13:
        y = 13
    print(y)
    theta = int((theta // 5) * 5)
    joint_values = arm.lookup_table[f"({y}, {theta})"]
    # joint_values = arm.algebraic_inverse_kinematics([13*2.54/100, 13*2.54/100], 0)
    y = y*2.54/100
    for i, joint_value in enumerate(joint_values[0:3]):
        if joint_value > pi:
            joint_values[i] -= 2*pi
    print(joint_values[0]*180/pi, joint_values[1]*180/pi, joint_values[2]*180/pi)
    #arm.link1.moveJoint(-joint_values[0])
    #arm.link2.moveJoint(joint_values[1])
    #arm.link3.moveJoint(joint_values[2])
    # arm.link1.moveJoint(0)
    # arm.link2.moveJoint(0)
    # arm.link3.moveJoint(0)
    for key, value in arm.lookup_table.items():
        if ", 0)" not in key:
            continue
        if value[3] == True:
            joint_values = value[0:3]
            y = y*2.54/100
            for i, joint_value in enumerate(joint_values[0:3]):
                if joint_value > pi:
                    joint_values[i] -= 2*pi
            arm.link1.moveJoint(-joint_values[0])
            arm.link2.moveJoint(joint_values[1])
            arm.link3.moveJoint(joint_values[2])
            arm.write_servos()
            time.sleep(1)
            
    
    print(arm.forward_kinematics())
    arm.write_servos()
    # lookup = {}
    # angle = -30
    # bar_width = 18
    # y = -bar_width
    # x = 13*2.54/100
    # for y in range(-17,18):
    #     for theta in range(-60,61,5):
    #         arm.algebraic_inverse_kinematics([x,y*2.54/100],theta*pi/180)
    #         valid = arm.link1.valid and arm.link2.valid and arm.link3.valid
    #         lookup[f"{y,theta}"] = [arm.link1.theta, arm.link2.theta, arm.link3.theta, valid]
    # with open("lookup_table.json", "w") as json_file:
    #     json.dump(lookup, json_file, indent=4)
    