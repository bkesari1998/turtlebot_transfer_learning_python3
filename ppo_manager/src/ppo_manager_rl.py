import rospy
from ppo_manager.src.junk.real_PPO import PPO
from action_srv.srv import PPO_Action
from action_srv.msg import NetworkInfo
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# import gym
# import pickle
# import TurtleBot_v0
from torch.distributions import Categorical
import time
from torch.utils.data import DataLoader, TensorDataset

net = PPO(lr_actor=0.0001, lr_critic=0.0001, gamma=0.99, K_epochs=7, eps_clip=0.2)
episode_number = 0
episode_rewards = []
episode_reward = 0
episode_dones = []

def network_handler(msg):
    
    global net

    global episode_reward
    global episode_rewards
    global episode_dones
    global episode_number

    net.buffer.rewards.append(msg.reward)
    episode_reward += msg.reward
    net.buffer.is_terminals.append(msg.end_episode)

    if msg.end_episode:

        episode_rewards.append(episode_reward)
        if msg.goal_reached:
            episode_dones.append(1)
        else:
            episode_dones.append(0)
        
        rospy.loginfo("Episode " + str(episode_number) + " with reward " + str(episode_reward) + " finished")

        episode_reward = 0
        
        net.update()
        net.save("/home/mulip/python_3_catkin_ws/src/turtlebot_transfer_learning_python3/ppo_manager/src/ppo_rl.pth")
        np.save("/home/mulip/python_3_catkin_ws/src/turtlebot_transfer_learning_python3/ppo_manager/src/reward1.npy", episode_rewards)
        np.save("/home/mulip/python_3_catkin_ws/src/turtlebot_transfer_learning_python3/ppo_manager/src/dones1.npy", episode_dones)
        
        if len(episode_dones) > 100 and np.mean(episode_dones) > .9:
            rospy.loginfo("RL Learner Converged!")
            rospy.signal_shutdown()

    return
    
def ppo_handler(req):
    global net
    state = req.img
    obs = np.array(state)        
    obs = np.reshape(obs,(100,100,3))
    obs = np.divide(np.asarray(obs), 255)
    obs = np.swapaxes(obs,0,2)
    obs = np.expand_dims(obs, axis=0)

    return net.get_action(obs)

def shutdown():
    global net
    global episode_dones
    global episode_rewards

    net.save("/home/mulip/python_3_catkin_ws/src/turtlebot_transfer_learning_python3/ppo_manager/src/ppo_rl.pth")
    np.save("/home/mulip/python_3_catkin_ws/src/turtlebot_transfer_learning_python3/ppo_manager/src/reward1.npy", episode_rewards)
    np.save("/home/mulip/python_3_catkin_ws/src/turtlebot_transfer_learning_python3/ppo_manager/src/dones1.npy", episode_dones)


if __name__ == '__main__':

    rospy.init_node("ppo_manager")
    rospy.on_shutdown(shutdown)

    net.load("/home/mulip/python_3_catkin_ws/src/turtlebot_transfer_learning_python3/ppo_manager/src/ppo_rl.pth")

    net_pub = rospy.Publisher("net_info", NetworkInfo, queue_size=1)
    action_srv = rospy.Service("ppo_action_srv", PPO_Action, ppo_handler)
    net_sub = rospy.Subscriber("net_info", NetworkInfo, network_handler)
    while not rospy.is_shutdown():
        rospy.spin()
