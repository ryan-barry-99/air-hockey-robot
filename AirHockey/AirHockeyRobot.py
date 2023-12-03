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


class AirHockeyRobot:
    def __init__(self):
        """
        This module contains the integration of the air hockey robot's components for
        puck and table detection, tracking, and control
        """
        self.arm = ArmRobot()
        self.camera = Camera()
        self.yolo = YOLO("YOLOv8_air_hockey.pt")

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
                        bbox = box.xyxy.cpu().numpy()

                        if class_id == 0:
                            # Puck
                            self.puck_bbox = bbox
                        elif class_id == 1:
                            # table
                            self.table_bbox = bbox
                        
            if display:
                # try:
                #     if self.table_bbox is not None:
                #         bounding_box = np.array(self.table_bbox)
                #         x1 = int(bounding_box[0] * (1920 / 480))
                #         y1 = int(bounding_box[1] * (1080 / 480))
                #         x2 = int(bounding_box[2] * (1920 / 480))
                #         y2 = int(bounding_box[3] * (1080 / 480))

                #         start_point = (x1, y1)
                #         end_point = (x2, y2)
                #         # Blue color in BGR
                #         color = (0, 255, 0)
                #         # Line thickness of 2 px
                #         thickness = 5
                #         # Using cv2.rectangle() method
                #         # Draw a rectangle with blue line borders of thickness of 2 px
                #         image = cv2.rectangle(image, start_point, end_point, color, thickness)
                #     if self.puck_bbox is not None:
                #         bounding_box = np.array(self.table_bbox)
                #         x1 = int(bounding_box[0] * (1920 / 480))
                #         y1 = int(bounding_box[1] * (1080 / 480))
                #         x2 = int(bounding_box[2] * (1920 / 480))
                #         y2 = int(bounding_box[3] * (1080 / 480))

                #         start_point = (x1, y1)
                #         end_point = (x2, y2)
                #         # Blue color in BGR
                #         color = (0, 255, 0)
                #         # Line thickness of 2 px
                #         thickness = 5
                #         # Using cv2.rectangle() method
                #         # Draw a rectangle with blue line borders of thickness of 2 px
                #         image = cv2.rectangle(image, start_point, end_point, color, thickness)
                # except:
                #     pass
                # end_time = cv2.getTickCount()
                # elapsed_time = (end_time - self.camera.start_time) / cv2.getTickFrequency()
                # fps = 1 / elapsed_time

                # # Convert the FPS to a string and add it as text on the image
                # fps_text = f"FPS: {fps:.2f}"
                # cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

                # self.camera.start_time = end_time
                cv2.imshow("Camera Feed", frame)
                cv2.waitKey(1)# Calculate and display FPS as text on the image

    
if __name__ == "__main__":
    robot = AirHockeyRobot()
    while True:
        robot.yolo_detect(display=False)
    # print(robot.puck_bbox)
    # print(robot.table_bbox)