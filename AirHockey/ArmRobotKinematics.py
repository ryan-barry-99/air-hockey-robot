"""
File: ArmRobotKinematics.py

This module contains thekineamt

Author: Ryan Barry
Date created: June, 26 2023
"""

import numpy as np  # Importing the NumPy library for mathematical operations
from Frame import PRISMATIC, REVOLUTE, FIXED_ROTATION, FIXED_TRANSLATION, Frame
from math import atan2, sqrt, pi, sin, cos, acos

class ArmRobotKinematics:
    def __init__(self):
        """
        This module defines the ArmRobot class, which represents an arm robot with multiple joints.
        It provides functionality to add frames to the arm, move individual joints, and perform 
        forward kinematics to compute the end-effector position and orientation.
        """
        self._frames = []
        
    def addFrame(self, joint_type, theta_fix=0, d=0, a=0, alpha_fix=0, min_lim=None, max_lim=None):
        """
        Adds a frame to the arm robot. 
 
        Parameters: 
            joint_type: The type of joint, either REVOLUTE or PRISMATIC. 
            theta_fix: The fixed angle of rotation about the z axis for a revolute joint. Default is 0. 
            d: The distance of translation along the z axis for a prismatic joint. Default is 0. 
            a: The length of the frame. 
            alpha_fix: The fixed angle of rotation about the x axis. Default is 0. 
        """ 
        new_frame = Frame(joint_type=joint_type, d=d, theta_fix=theta_fix, a=a, alpha_fix=alpha_fix, min_lim=min_lim, max_lim=max_lim)
        self._frames.append(new_frame)
        return new_frame

    def forward_kinematics(self):
        '''
        Computes the forward kinematics of the arm robot. 
 
        Returns the end-effector position and orientation in the form (x, y, z, roll, pitch, yaw). 
        '''
        self.__T = np.identity(4)  # Initialize the transformation matrix as an identity matrix

        for frame in self._frames:
            self.__T = np.dot(self.__T, frame.transform_matrix())  # Multiply the transformation matrix T by A


        x = float(self.__T[0, 3])
        y = float(self.__T[1, 3])
        z = float(self.__T[2, 3])

        roll = atan2(self.__T[2, 1], self.__T[2, 2])
        pitch = atan2(-self.__T[2, 0], sqrt(self.__T[2, 1]**2 + self.__T[2, 2]**2))
        yaw = atan2(self.__T[1, 0], self.__T[0, 0])

        return x, y, z, roll, pitch, yaw
    
    def transform_matrix(self):
        '''
        Computes the transformation matrix of the arm robot. 
 
        Returns the transformation matrix T. 
        '''
        self.forward_kinematics()
        return self.__T
    
    def algebraic_inverse_kinematics(self, target_position, target_orientation):
        # Override the generic inverse kinematics method
        # Implement the specific inverse kinematics calculations for the arm robot
        x = target_position[0] 
        y = target_position[1]
        
        if target_orientation < 0:
            target_orientation = 2*pi + target_orientation

        # Calculate the joint angles
        p2x = x - self.link3.a*cos(target_orientation)
        p2y = y - self.link3.a*sin(target_orientation)
        l1 = self.link1.a # Assign l1 to link 1
        l2 = self.link2.a # Assign l2 to link 2
        print((p2x**2 + p2y**2 - l1**2 - l2**2)/(2*l1*l2))
        theta2 = round(acos((p2x**2 + p2y**2 - l1**2 - l2**2)/(2*l1*l2)),10)
        theta1 = round((atan2(p2y, p2x) - atan2(l2*sin(theta2), l1 + l2*cos(theta2))),10)
        theta3 = round((target_orientation - theta1 - theta2),10)
        print("\nAlgebraic Inverse Kinematics")
        print(f'theta1: {theta1}')
        print(f'theta2: {theta2}')
        print(f'theta3: {theta3}')
        for theta in [theta1, theta2, theta3]:
            if theta > pi:
                theta = theta - 2*pi
                print(theta)
        self.link1.moveJoint(theta1)
        self.link2.moveJoint(theta2)
        self.link3.moveJoint(theta3)
        

    def iterative_inverse_kinematics(self, target_position, target_orientation, tolerance=0.01, max_iterations=100000, momentum=0.2):
        """
        Computes the inverse kinematics using an iterative method.
        target_position: The desired end-effector position [x, y, z]
        target_orientation: The desired end-effector orientation [roll, pitch, yaw]
        tolerance: The acceptable error in the end-effector position. Default is 0.01 meters.
        """
        joint_values = []
        # Convert target_position and target_orientation to numpy arrays
        target_position = np.array(target_position)
        target_orientation = np.array(target_orientation)

        # Initialize variables
        position_error = np.inf
        orientation_error = np.inf
        iterations=0
        # While error is greater than tolerance, iterate
        prev_dq = np.zeros(len(self._frames))
        while np.linalg.norm(position_error) > tolerance or np.linalg.norm(orientation_error) > tolerance:
            # Calculate current end-effector position and orientation
            x, y, z, roll, pitch, yaw= self.forward_kinematics()
            current_position = [x, y, z]
            current_orientation = [roll, pitch, yaw]
            # Calculate position and orientation error
            position_error = target_position - current_position

            # Calculate orientation error
            orientation_error = target_orientation - current_orientation

            # Calculate Jacobian matrix
            J = self.jacobian()

            # Solve for joint increments
            dq = np.linalg.pinv(J) @ np.concatenate((position_error, orientation_error)) + momentum * prev_dq
            prev_dq = dq

            # Update joint angles
            for i in range(len(self._frames)):
                self._frames[i].moveJoint(self._frames[i].theta + dq[i])

            # Recalculate current end-effector position and orientation
            x, y, z, roll, pitch, yaw = self.forward_kinematics()
            current_position = [x, y, z]
            current_orientation = [roll, pitch, yaw]
            # print(f'current_position: {current_position}')
            # print(f'current_orientation: {current_orientation}')

            # Recalculate position and orientation error
            position_error = target_position - current_position
            orientation_error = target_orientation - current_orientation
            
            iterations += 1
            if iterations == max_iterations:
                raise ValueError("Inverse kinematics did not converge.")

            
        for i, frame in enumerate(self._frames):
            if frame.joint_type == REVOLUTE:
                theta = frame.theta
                if theta > pi:
                    theta = theta - 2*pi
                joint_values.append(theta)
            elif frame.joint_type == PRISMATIC:
                joint_values.append(frame.d)
        print("\nIterative Inverse Kinematics")
        non_fixed = [frame for frame in self._frames if frame.joint_type not in [FIXED_TRANSLATION, FIXED_ROTATION]]
        for i, frame in enumerate(non_fixed):
            print(f'theta{i+1}: {joint_values[i]}')
        return tuple(joint_values)
        

    def jacobian(self):
        J = np.zeros((6, len(self._frames)))

        On = self.forward_kinematics()[:3]  # End-effector position
        for i, frame in enumerate(self._frames):
            if frame.joint_type in [FIXED_ROTATION, FIXED_TRANSLATION]:
                continue
            # Get rotation axis
            Zi = frame.transform_matrix()[:3, 2]  # The third column of the rotation matrix
            Oi = frame.transform_matrix()[:3, 3]  # The fourth column of the transformation matrix is the origin of the ith frame

            if frame.joint_type == REVOLUTE:
                # Linear velocity for revolute joint
                J[:3, i] = np.cross(Zi, On - Oi)
                # Angular velocity for revolute joint
                J[3:, i] = Zi
            elif frame.joint_type == PRISMATIC:
                # Linear velocity for prismatic joint
                J[:3, i] = Zi
                # Angular velocity for prismatic joint is zero
                J[3:, i] = 0
            else:
                raise ValueError(f"Unknown joint type: {frame.type}")

        return J
