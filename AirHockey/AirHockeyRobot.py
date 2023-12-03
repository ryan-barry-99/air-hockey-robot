"""
File: AirHockeyRobot.py

Description This module contains the integration of the air hockey robot's components for
puck and table detection, tracking, and control


Author: Ryan Barry
Date Created: December 3, 2023
"""
from Arm_Robot.ArmRobot import ArmRobot
from Camera import Camera
import torch
from ultralytics import YOLO
import numpy as np


class AirHockeyRobot:
    def __init__(self):
        """
        This module contains the integration of the air hockey robot's components for
        puck and table detection, tracking, and control
        """
        self.arm = ArmRobot()
        self.camera = Camera()
        self.yolo = YOLO("YOLOv8_air_hockey.pt")

    def yolo_predict(self, image):
        """
        Uses YOLO to detect the puck and table in the image and stores the bounding boxes
        """
        image = image.resize(480,480)
        results = self.yolo(image)

        for r in results:
            for box in r.boxes:
                class_id = box.cls.cpu().np()[0]
                bbox = box.xyxy.cpu().np()

                if class_id == 0:
                    # Puck
                    self.puck_bbox = bbox
                elif class_id == 1:
                    # table
                    self.table_bbox = bbox

    