"""
File: Camera.py

Description: This module contains the Camera class, which is used to interface 
with an OAK-D camera to capture images.

Author: Ryan Barry 
Date Created: December 3, 2023
"""
import depthai as dai

class Camera:
    def __init__(self):
        self.pipeline = self.create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.device.startPipeline()

    def create_pipeline(self):
        pipeline = dai.Pipeline()

        # Create the ColorCamera node
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam.setFps(60)

        # Create an output link for the ColorCamera
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("video")
        cam.video.link(xout.input)

        return pipeline
    
    def get_frame(self):
        output_queue = self.device.getOutputQueue("video")
        video_frame = output_queue.get()
        frame = video_frame.getCvFrame()
        return frame