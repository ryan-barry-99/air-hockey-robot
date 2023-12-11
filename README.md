# air-hockey-robot
Welcome to the repo for our implementation of a robot that plays air hockey!

This consisted of several aspects, including:
 - The implementation of a custom arm robot kinematics library (see more here https://github.com/ryan-barry-99/arm-robot-kinematics/tree/main)
 - Development of a custom dataset and trainig of a YOLOv8 model to detect the puck and key points on the table (see YOLOv8 directory)
 - Development of a physics model and LSTM for intercept point prediction (view dev-quinn branch)
 - Real time implementation of the above methods to play air hockey against a human (see AirHockey directory)

Click here to read our paper: https://github.com/ryan-barry-99/air-hockey-robot/blob/master/Trajectory_Estimation_Approach_To_Robotic_Air_Hockey_Opponent.pdf
