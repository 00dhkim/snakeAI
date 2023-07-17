import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear11 = nn.Linear(hidden_size, hidden_size)
        # self.linear12 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear11(x))
        # x = F.relu(self.linear12(x))
        x = self.linear2(x)
        # x = nn.Softmax(dim=-1)(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = 'Python\\freeCodeCamp\\model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Linear_QNet2(nn.Module):
    '''
    길이 11의 state는 mlp, 길이 50의 window는 cnn으로 처리 후 concat
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(11, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.cnn1 = nn.Conv2d(2, 64, 3, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
        self.cnn3 = nn.Conv2d(64, 2, 3, padding=1)
        
        self.decoder1 = nn.Linear(50+hidden_size, hidden_size)
        self.decoder2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, 61)
        x1 = x[:, :-50] # 앞 11개 state
        # x2 = torch.tensor(x[:, -50:]).view(-1, 2, 5, 5) # 뒷 50개 window
        x2 = torch.tensor(x[:, -50:]) # 뒷 50개 window
        x2 = torch.stack(torch.split(x2, 2, dim=-1), dim=-1).view(-1, 2, 5, 5) # 뒷 50개 window
        
        x1 = F.relu(self.linear1(x1))
        x1 = self.linear2(x1)
        
        x2_ = F.relu(self.cnn1(x2))
        x2_ = F.relu(self.cnn2(x2_))
        x2_ = self.cnn3(x2_)
        x2_ += x2
        x2_ = nn.Flatten()(x2_)
        
        # x1 = (x1 - x1.mean()) / (x1.std() + 1e-8)
        # x2 = (x2 - x2.mean()) / (x2.std() + 1e-8)
        
        x = torch.cat((x1, x2_), dim=-1)
        x = F.relu(self.decoder1(x))
        x = self.decoder2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = 'Python\\freeCodeCamp\\model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[100, 200, 300, 400, 500, 600], gamma=0.5)
        # self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.5)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x), 1차원으로 만들어줘야 함
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # 한개짜리 tuple

        # 1: predicted Q values with current state
        # window = state[-50:] #TODO:
        # window = np.array(window).view((1, 2, 5, 5))#TODO:
        # state = state[:-50]#TODO:
        
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                # Q_new = reward[idx] * (1 - self.gamma) + self.gamma * torch.max(self.model(next_state[idx])) # step에 따라 예전 reward를 덜 강조.
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + gamma * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        # e.g., action: [1, 0, 0] 이면, 0번 자리에 Q_new를 대입

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()  # back propagation and update gradients

        self.optimizer.step()
        # self.lr_scheduler.step()