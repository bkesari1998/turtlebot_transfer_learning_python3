import rospy
from model_net import model_net
from action_srv.srv import PPO_Action
import torch
import torch.nn as nn
import torch.nn.functional as F

# import gym
# import pickle
# import TurtleBot_v0
from torch.distributions import Categorical
import time
from torch.utils.data import DataLoader, TensorDataset


net = model_net()

def ppo_handler(req):
    global net
    return net.get_action(req.img)

if __name__ == '__main__':

    rospy.init_node("ppo_manager")

    action_srv = rospy.Service("ppo_action_srv", PPO_Action, ppo_handler)
    while not rospy.is_shutdown():
        rospy.spin()
