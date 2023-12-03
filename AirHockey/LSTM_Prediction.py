from ArmRobot import ArmRobot
from Camera import Camera

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import queue

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
    def __init__(self, dt=False):
        self.sequence_length = 10
        self.dt = dt

        if self.dt == False:
            self.input_size = 2  # Number of input features
            self.hidden_size = 80 #50  # Number of LSTM units
            self.output_size = 1  # Number of output features
            self.num_layers = 2  # Number of LSTM layers
            
            PATH = 
            self.model = LSTMModel(self.input_size, self.hidden_size, self.num_layers, self.output_size)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.load_state_dict(torch.load(PATH))
            self.model.to(self.device)
            self.sequence = torch.zeros(self.sequence_length, 2)

        else:
            self.input_size = 3  # Number of input features
            self.hidden_size = 80 #50  # Number of LSTM units
            self.output_size = 1  # Number of output features
            self.num_layers = 2  # Number of LSTM layers
            PATH = 
            self.model = LSTMModel(self.input_size, self.hidden_size, self.num_layers, self.output_size)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.load_state_dict(torch.load(PATH))
            self.model.to(self.device)
            self.sequence = torch.zeros(self.sequence_length, 3)

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
        testing_outputs = self.model(self.sequence)

        return(testing_outputs)
            


