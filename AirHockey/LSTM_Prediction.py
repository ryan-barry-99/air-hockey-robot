from ArmRobot import ArmRobot
from Camera import Camera
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import queue
from Physics_Prediction import Physics_Prediction

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout =1e-5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

class LSTM_Prediction():
    def __init__(self, path, dt=False):
        self.sequence_length = 10
        self.dt = dt
        #self.path = path

        #print(self.device)
        hidden_size = 80 #50  # Number of LSTM units
        output_size = 1  # Number of output features
        num_layers = 2  # Number of LSTM layers
        if dt == False:
            input_size = 2
            self.sequence = torch.zeros(self.sequence_length, 2)
        else:
            input_size = 3
            self.sequence = torch.zeros(self.sequence_length, 3)
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load("C:/Users/ryanb/OneDrive/Desktop/RIT/Robot_Perception/Final_Project/air-hockey-robot/AirHockey/LSTM_HS80_L2_dt.pt", map_location=torch.device(self.device)))
        self.model.eval()
        self.left_queue = queue.Queue()
        self.right_queue = queue.Queue()
        self.dt_queue = queue.Queue()
        
        for i in range(self.sequence_length):
            self.left_queue.put(0)
            self.right_queue.put(0)
            self.dt_queue.put(0)
            
        
    def handle_queue(self, center_x, center_y, dt=0):
        self.left_queue.get()
        self.right_queue.get()
        self.left_queue.put(center_x)
        self.right_queue.put(center_y)
        
        self.sequence[:,0] = torch.tensor(list(self.left_queue.queue))
        self.sequence[:,1] = torch.tensor(list(self.right_queue.queue))

        if self.dt:
            self.dt_queue.get()
            self.dt_queue.put(dt)
            self.sequence[:,2] = torch.tensor(list(self.dt_queue.queue))

    def __getitem__(self, table_bbox, puck_bbox, dt=0):
        #puck_x = (px-tux)/(tLx - tux)
        # (x relative to table) / size of table
        puck_x = (puck_bbox[0] - table_bbox[0]) / (table_bbox[2]-table_bbox[0])
        puck_y = (puck_bbox[1] - table_bbox[1]) / (table_bbox[3]-table_bbox[1])
        self.handle_queue(puck_x, puck_y, dt)
        sequence = self.sequence.unsqueeze(0).to(self.device)
        testing_outputs = self.model(sequence)

        return(testing_outputs)
            

if __name__=="__main__":
    input_size = 3  # Number of input features
    hidden_size = 80 #50  # Number of LSTM units
    output_size = 1  # Number of output features
    num_layers = 2  # Number of LSTM layers

    ''' 
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.load_state_dict(torch.load("./LSTM_HS80_L2_dt.pt", map_location=torch.device(device)))
    model.eval()
    '''

    LSTM = LSTM_Prediction(path= "./LSTM_HS80_L2_dt.pt", dt = True)
    Phys = Physics_Prediction(dt = True)
    
    table_bbox = [0, 0, 1, 1]
    puck_bbox = [1, 0.5, 0.5, 0.5]
    dt = 0.1
    print("Physics Prediction:", Phys.__getitem__(puck_bbox, table_bbox, dt))
    print("LSTM Prediction:", LSTM.__getitem__(table_bbox, puck_bbox, dt))
    print("")
    puck_bbox = [0.9, 0.39, 0.5, 0.5]
    print("Physics Prediction:", Phys.__getitem__(puck_bbox, table_bbox, dt))
    print("LSTM Prediction:", LSTM.__getitem__(table_bbox, puck_bbox, dt))
    print("")