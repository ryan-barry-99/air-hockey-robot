import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image

class Annotate_Image:
    def __init__(self, image_num, file_name="training_labels.csv"):
        self.home = os.getcwd() + "/data_collection"
        self.file_df = pd.read_csv(os.path.abspath(os.path.join(self.home, file_name)))
        self.image_num = image_num
        self.image_dir = f"{self.home}/training_data/"

    def open_and_resize_image(self, file_path, size=(480,480)):
        # Open the image
        self.image = Image.open(file_path)
        # Resize the image
        self.image = self.image.resize(size)
    
    def annotate_image(self, size=[480,480]):
        path = self.image_dir + self.file_df.iloc[self.image_num]["File Name"]
        # Reading an image in default mode
        self.open_and_resize_image(path, tuple(size))
        self.image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)

        bounding_box_str = self.file_df.at[self.image_num, "Puck Box"]
        if bounding_box_str is not None:
            # Extract numerical values from the bounding box string
            xyxy = [float(val) for val in bounding_box_str.replace('[', '').replace(']', '').split()]

            # Convert the list of numbers to a NumPy array
            bounding_box = np.array(xyxy)
            x1 = int(bounding_box[0] * (size[0] / 480))
            y1 = int(bounding_box[1] * (size[1] / 480))
            x2 = int(bounding_box[2] * (size[0] / 480))
            y2 = int(bounding_box[3] * (size[1] / 480))

            start_point = (x1, y1)
            end_point = (x2, y2)
            # Blue color in BGR
            color = (255, 0, 0)
            # Line thickness of 2 px
            thickness = 2 * int((size[0] / 480))
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            self.image = cv2.rectangle(self.image, start_point, end_point, color, thickness)


        if bounding_box_str is not None:
            bounding_box_str = self.file_df.at[self.image_num, "Table Box"]

            # Extract numerical values from the bounding box string
            xyxy = [float(val) for val in bounding_box_str.replace('[', '').replace(']', '').split()]


            # Convert the list of numbers to a NumPy array
            bounding_box = np.array(xyxy)
            x1 = int(bounding_box[0] * (size[0] / 480))
            y1 = int(bounding_box[1] * (size[1] / 480))
            x2 = int(bounding_box[2] * (size[0] / 480))
            y2 = int(bounding_box[3] * (size[1] / 480))

            start_point = (x1, y1)
            end_point = (x2, y2)
            # Blue color in BGR
            color = (0, 255, 0)
            # Line thickness of 2 px
            thickness = 2 * int((size[0] / 480))
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            self.image = cv2.rectangle(self.image, start_point, end_point, color, thickness)
        # Convert BGR image to RGB for displaying with matplotlib
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    ai = Annotate_Image(1000)
    ai.annotate_image([1920,1080])