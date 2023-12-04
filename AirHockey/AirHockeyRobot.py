"""
File: AirHockeyRobot.py

Description This module contains the integration of the air hockey robot's components for
puck and table detection, tracking, and control


Author: Ryan Barry
Date Created: December 3, 2023
"""
import cv2
from ArmRobot import ArmRobot
from Camera import Camera
import torch
from ultralytics import YOLO
import numpy as np
import multiprocessing
from PIL import Image
from Physics_Prediction import Physics_Prediction
from LSTM_Prediction import LSTM_Prediction


class AirHockeyRobot:
    def __init__(self):
        """
        This module contains the integration of the air hockey robot's components for
        puck and table detection, tracking, and control
        """
        self.arm = ArmRobot()
        self.camera = Camera()
        self.yolo = YOLO("YOLOv8_air_hockey.pt")
        self.physics = Physics_Prediction()
        #self.lstm_dt = LSTM_Prediction(path="LSTM_HS80_L2_dt.pt", dt=True)
        self.lstm = LSTM_Prediction(path="LSTM_HS80_L2.pt", dt=False)
        self.puck_bbox = None
        self.table_bbox = None

    def yolo_detect(self, display=False):
        """
        Uses YOLO to detect the puck and table in the image and stores the bounding boxes
        """
        frame = self.camera.get_frame()
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            frame = cv2.resize(frame, (480, 480))
            results = self.yolo(frame)
            if len(results) > 0:
                for r in results:
                    for box in r.boxes:
                        class_id = box.cls.cpu().numpy()[0]
                        
                        if class_id == 0:
                            # Puck
                            self.puck_bbox = box.xywh.cpu().numpy()[0].tolist()
                            self.puck_bbox_xyxy = box.xyxy.cpu().numpy()[0].tolist()  
                            #print(self.puck_bbox.type())
                        elif class_id == 1:
                            # table
                            self.table_bbox = box.xyxy.cpu().numpy()[0].tolist()
                            #print(self.table_bbox.type())
            
            end_time = cv2.getTickCount()
            elapsed_time = (end_time - self.camera.start_time) / cv2.getTickFrequency()
            fps = 1 / elapsed_time
            self.camera.start_time = end_time
            if self.puck_bbox is not None and self.table_bbox is not None:
                self.LSTM_pred = self.lstm.__getitem__(self.table_bbox, self.puck_bbox, elapsed_time)
                print(self.physics.__getitem__(self.table_bbox, self.puck_bbox, elapsed_time))
                self.PHYS_pred = self.physics.__getitem__(self.table_bbox, self.puck_bbox, elapsed_time)
            
            
            print(fps)
            if display:
                try:
                    if self.table_bbox is not None:
                        bounding_box = np.array(self.table_bbox)
                        x1 = int(bounding_box[0])
                        y1 = int(bounding_box[1])
                        x2 = int(bounding_box[2])
                        y2 = int(bounding_box[3])

                        start_point = (x1, y1)
                        end_point = (x2, y2)
                        # Blue color in BGR
                        color = (255, 0, 0)
                        # Line thickness of 2 px
                        thickness = 2
                        # Using cv2.rectangle() method
                        # Draw a rectangle with blue line borders of thickness of 2 px
                        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
                    if self.puck_bbox is not None:
                        bounding_box = np.array(self.puck_bbox_xyxy)
                        x1 = int(bounding_box[0])
                        y1 = int(bounding_box[1])
                        x2 = int(bounding_box[2])
                        y2 = int(bounding_box[3])

                        start_point = (x1, y1)
                        end_point = (x2, y2)
                        # Blue color in BGR
                        color = (0, 255, 0)
                        # Line thickness of 2 px
                        thickness = 2
                        # Using cv2.rectangle() method
                        # Draw a rectangle with blue line borders of thickness of 2 px
                        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
                        print("---------------------------------------------------------------------------",start_point, end_point)
                except:
                    pass

                # Convert the FPS to a string and add it as text on the image
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

                self.camera.start_time = end_time
                cv2.imshow("Camera Feed", frame)
                cv2.waitKey(1)# Calculate and display FPS as text on the image

    
def find_closest_value(dictionary):
    true_values = []
    for key, value in dictionary.items():
        if value[3] == True:
            true_values.append(value)
    
    closest_value = min(true_values, key=lambda x: abs(x[0]))
    return closest_value

if __name__ == "__main__":
    robot = AirHockeyRobot()
    robot.PHYS_pred = None
    while True:
        robot.yolo_detect(display=True)
        if robot.PHYS_pred == None:
            continue
        Phys_ret = robot.PHYS_pred
        theta = robot.physics.theta
        y = robot.physics.y_puck
        y = int((42*y - 21)*2.54/100)
        if y < -13:
            y = -13
        elif y > 13:
            y = 13
        theta = int((theta // 5) * 5)
        joint_angles = ltbl[f"({y}, {theta})"]
        
        if not joint_angles[3]:
            joint_angles = find_closest_value(robot.arm.lookup_table)
        # print("closest joint angles with fourth element True:", closest_joint_angles)
    
        robot.arm.link1.moveJoint(joint_angles[0])
        robot.arm.link2.moveJoint(joint_angles[1])
        robot.arm.link3.moveJoint(joint_angles[2])
        robot.arm.write_servos()
            
    # print(robot.puck_bbox)
    # print(robot.table_bbox)