from neurogine import *
import gym
import numpy as np
import time
from collections import deque
print(gym.envs.registry.all())
class EVBot:
    def __init__(self):
        self.dqn = Neurogine([
            4,24,24,2
        ])
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.alpha = 0.1
        self.gamma = 0.6
        self.memory = deque([],1000000)
        # self.visualizer = NeuroVisualizer()
        # self.dqn.loadVisualizer(self.visualizer)
        # self.visualizer.VisualizeAsync()
        self.env = gym.make("Tennis-ramNoFrameskip-v4")
    def sigmoid(self,x):
            return 1/(1+np.exp(-x))
    def predict_action(self,observation):
        return self.dqn.think(observation) if np.random.random() > self.epsilon else self.env.action_space.sample() 
    def train(self,prediction,state,reward):
        nextmax = self.dqn.thinkLayered(state)
        prediction_s = np.argmax(prediction)
        newValue = (1-self.alpha)*prediction+self.alpha*(reward+self.gamma*nextmax)
        newValue[prediction_s] = 1-newValue[prediction_s]  
        newValue = np.clip(newValue,0,1)
        # print(state,newValue)
        self.dqn.train(np.array([state]),[newValue],iteration=200,learningRate=2)
    def remember(self, done, action, observation, prev_obs):
        self.memory.append([done, action, observation, prev_obs])
        
    def experience_replay(self, update_size=20):
        if (len(self.memory) < update_size):
            return
        else: 
            batch_indices = np.random.choice(len(self.memory), update_size)
            for index in batch_indices:
                done, action_selected, new_obs, prev_obs = self.memory[index]
                action_values = self.dqn.think(prev_obs, True)
                next_action_values = self.dqn.think(new_obs, False)
                
                nextmax = next_action_values
                prediction_s = np.argmax(action_values)
                newValue = (1-self.alpha)*action_values+self.alpha*(1+self.gamma*nextmax)
                newValue[prediction_s] = 1-newValue[prediction_s]  

                self.dqn.trainRelu(np.array(np.asmatrix(prev_obs)),np.array(np.asmatrix(newValue)),1)
        self.epsilon = self.epsilon if self.epsilon < 0.01 else self.epsilon*self.epsilon_decay
        self.dqn.updateStoredSynapses()
    def start(self):
        dtime =time.time()
        eT=0 
        for i_episode in range(1000):
            state = self.env.reset()
            for t in range(200):
                # prediction = self.predict_action(state)
                # action = np.argmax(prediction)
                # prev_state = state
                state, reward, done, info = self.env.step(self.env.action_space.sample())
                # self.remember(done,action,state,prev_state)
                # self.experience_replay(10)
                self.env.render()
                if(done):
                    # print("Episode : %s ET : %s Epsilon : %s"%(i_episode,t,self.epsilon))
                    break
            dtime = time.time()
        self.env.close()
evb = EVBot()
evb.start()
# visualizer = NeuroVisualizer()
# # dqn.loadVisualizer(visualizer)
# # visualizer.VisualizeAsync()

