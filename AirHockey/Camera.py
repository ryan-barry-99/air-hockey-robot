"""
File: Camera.py

Description: This module contains the Camera class, which is used to interface 
with an OAK-D camera to capture images.

Author: Ryan Barry 
Date Created: December 3, 2023
"""
import cv2
import depthai as dai
import threading
import numpy as np

class Camera:
    def __init__(self):
        self.pipeline = self.create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.device.startPipeline()
        self.start_time = cv2.getTickCount() 
        self.frame = None
        self.old_frame = None
        self.output_queue = self.device.getOutputQueue("video", maxSize=1, blocking=False)
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def create_pipeline(self):
        pipeline = dai.Pipeline()
        # Create the ColorCamera node
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam.setFps(30)
        # Create an output link for the ColorCamera
        xout_video = pipeline.create(dai.node.XLinkOut)
        xout_video.setStreamName("video")
        cam.video.link(xout_video.input)
        return pipeline

    def run(self):
        while True:
            video_frame = self.output_queue.get()
            if video_frame is not None:
                self.frame = video_frame.getCvFrame()

    def get_frame(self):
        if self.frame is None:
            return None
        if self.old_frame is not None:
            # Compare the current frame with the last frame
            difference = np.abs(self.frame - self.old_frame)
            is_different = np.any(difference > 0)
            if not is_different:
                return None
        
        self.old_frame = self.frame.copy()
        return self.frame