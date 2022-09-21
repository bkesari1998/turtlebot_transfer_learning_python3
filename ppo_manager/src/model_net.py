'''
Imports
'''
# from email.mime import image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import gym
# import pickle
# import TurtleBot_v0
from torch.distributions import Categorical
import time
from torch.utils.data import DataLoader, TensorDataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.actor = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size = 3, stride = 1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(64,64, kernel_size = 3, stride = 1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Flatten(),   
                #nn.Linear(6400,1024),
                nn.Linear(33856,512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.ReLU(),       
                nn.Linear(64,4)
                # nn.Softmax(dim=-1)
                )            


    def forward(self, state):
        action_probs = self.actor(state)
        # dist = Categorical(action_probs)

        # action = dist.sample()
        # action_logprob = dist.log_prob(action)
        return action_probs
        # return action.detach()
        # return action_logprob.detach()

'''
Render BC Agent and Generate Gifs
'''
class model_net():
    def __init__(self):
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()):
            # torch.cuda.set_device(0)     
            self.device = torch.device('cuda',0) 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")        
        self.net = Net().to(self.device)
        self.model = self.net
        self.model.load_state_dict(torch.load('real_BC_lr=0.003_epochs=2000.pth',map_location=self.device))

    def get_action(self, obs):
        obs = np.array(obs)
        obs = np.reshape(obs, (100, 100, 3))
        obs = np.divide(np.asarray(obs), 255)
        obs = np.swapaxes(obs,0,2)
        obs = np.expand_dims(obs, axis=0)
        obs = torch.from_numpy(obs).to(torch.float32).to(self.device)
        action = self.model.forward(obs).detach()
        if action[0][0].item() > 0.5:
            action_to_take = 0    
        elif action[0][1].item() > 0.5:
            action_to_take = 1
        elif action[0][2].item() > 0.5:
            action_to_take = 2
        elif action[0][3].item() > 0.5:
            action_to_take = 3
        else:
            action_to_take = np.random.randint(0,4)

        return action_to_take        
