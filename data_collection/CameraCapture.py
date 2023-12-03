"""
File: CameraCapture.py

Description: This module contains the CameraCapture class, which is used to interface 
with an OAK-D camera to capture images and video. It can be used to save images to
a directory, display the video feed, and display the FPS of the video feed.

Author: Ryan Barry 
Date Created: October 24, 2023
"""

import cv2
import depthai as dai

class CameraCapture:
    def __init__(self, display_video=True, save_images=False, display_FPS=True):
        self.display_video = display_video
        self.save_images = save_images
        self.display_fps = display_FPS
        self.pipeline = self.create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.device.startPipeline()
        self.frame_count = 19391
        self.start_time = cv2.getTickCount()  # Initialize the start time for FPS calculation

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

    def run(self):
        with self.device:
            output_queue = self.device.getOutputQueue("video")

            while True:
                video_frame = output_queue.get()
                frame = video_frame.getCvFrame()

                if self.save_images:
                    directory = "C:\\Users\\ryanb\\OneDrive\\Desktop\\RIT\\Robot_Perception\\Project\\air-hockey-robot\\data_collection\\training_data"
                    cv2.imwrite(f"{directory}\\image_{self.frame_count:07d}.jpg", frame)
                    self.frame_count += 1

                if self.display_fps:
                    # Calculate and display FPS as text on the image
                    end_time = cv2.getTickCount()
                    elapsed_time = (end_time - self.start_time) / cv2.getTickFrequency()
                    fps = 1 / elapsed_time

                    # Convert the FPS to a string and add it as text on the image
                    fps_text = f"FPS: {fps:.2f}"
                    cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

                    self.start_time = end_time

                if self.display_video:
                    cv2.imshow("1080P", frame)

                if cv2.waitKey(1) == ord('q'):
                    break
                

import os
import csv
from datetime import datetime
if __name__ == "__main__":
    # Instantiate the CameraCapture class with display_video, save_images , and display_FPS flags
    camera = CameraCapture(display_video=True, save_images=False, display_FPS=True)
    # Start the camera capture
    while True:
        camera.run()
    