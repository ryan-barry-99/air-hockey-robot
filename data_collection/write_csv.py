if __name__ == "__main__":
    directory = "C:\\Users\\ryanb\\OneDrive\\Desktop\\RIT\\Robot_Perception\\Project\\air-hockey-robot\\data_collection"
    folder_path = f"{directory}\\training_data"
    csv_path = f"{directory}\\training_file_paths.csv"
    # Open the CSV file for writing
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row of the CSV file
        writer.writerow(["File Name", "Time"])
        # Loop through all the files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Get the last modified time of the file
            last_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%H:%M:%S.%f")[:-3]

            # Write the file name and last modified time to the CSV file
            writer.writerow([file_name, last_modified])
    print("Done")