# Arm Robot 

This repository contains the code for the  `ArmRobot`  module, which represents an arm robot with multiple joints. It provides functionality to add frames to the arm, move individual joints, update the Denavit-Hartenberg (DH) table, and perform forward and inverse kinematics calculations.

## File Structure

-  `ArmRobot.py`  : The main module file that contains the  `ArmRobot`  class. It inherits from  `ArmRobotKinematics`  and provides robot-specific initialization and inverse kinematics methods.
-  `ArmRobotKinematics.py`  : The module that defines the  `ArmRobotKinematics`  class. It contains methods to add frames, update the DH table, and perform forward kinematics. 
-  `frame.py` : The module that defines the  `frame`  class which represents a single frame in an arm robot.  
-  `requirements.txt` : Library dependencies
-  `README.md`  : This readme file to provide information about the repository.

## Usage

To use the  `ArmRobot`  module, follow these steps:

1. In  `ArmRobot.py` , update the  `__init__`  method to define the specific configuration of your arm robot by adding frames with the  `addFrame()`  method inherited from  `ArmRobotKinematics` . 
2. Import the module:
 `from ArmRobot import ArmRobot` 
3. Create an instance of the  `ArmRobot`  class:
 `arm = ArmRobot()` 
4. Update individual joint positions to reflect the state of your robot using the  `moveJoint`  method (inherited from  `ArmRobotKinematics`  class):
 `arm.moveJoint(joint, angle_or_distance)` 
   -  `joint`  : The number of the joint to move.
   -  `angle_or_distance`  : The angle (for revolute joints) or distance (for prismatic joints) to move the joint.
5. Perform forward kinematics to compute the end-effector position and orientation using the  `forward_kinematics`  method (inherited from  `ArmRobotKinematics`  class):
 `transformation_matrix = arm.forward_kinematics()` 
   - Returns the 4x4 transformation matrix representing the end-effector position and orientation.

6. Perform inverse kinematics (implemented in  `ArmRobot.py`  based on your robot's specific configuration) using the  `inverse_kinematics`  method:
 `joint_angles = arm.inverse_kinematics(target_position, target_orientation)` 
   -  `target_position`  : The desired end-effector position (x, y, z)
   -  `target_orientation`  : The desired end-effector orientation (roll, pitch, yaw)

## Author

- Name: Ryan Barry
- Date created: June 26, 2023