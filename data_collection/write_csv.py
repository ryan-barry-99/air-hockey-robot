import csv
import os
from datetime import datetime

class FileDataCollector:
    def __init__(self, directory):
        self.directory = directory
        self.folder_path = f"{directory}\\training_data"
        self.csv_path = f"{directory}\\training_file_paths.csv"
    
    def collect_file_data(self):
        with open(self.csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["File Name", "Time"])
            for file_name in os.listdir(self.folder_path):
                file_path = os.path.join(self.folder_path, file_name)
                last_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%H:%M:%S.%f")[:-3]
                writer.writerow([file_name, last_modified])
        print("Done")

    def append_data_to_column(self, df, column_name):
        with open(self.csv_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for _, row in df.iterrows():
                file_name = row["File Name"]
                data = row[column_name]
                writer.writerow([file_name, data])

if __name__ == "__main__":
    directory = "C:\\Users\\ryanb\\OneDrive\\Desktop\\RIT\\Robot_Perception\\Project\\air-hockey-robot\\data_collection"
    collector = FileDataCollector(directory)
    collector.collect_file_data()