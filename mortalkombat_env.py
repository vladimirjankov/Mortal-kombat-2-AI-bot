import numpy as np
import gym
import retro
from collections import deque 
import tensorflow as tf

import cv2



class ObservationWraperMK(gym.ObservationWrapper):

    def __init__(self, env):
        super(ObservationWraperMK, self).__init__(env)
        self.num_resets = 0
        self.player_hp = 120 # bot hp
        self.enemy_hp = 120 # in_game bot hp 
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64))
        self.q = deque(maxlen=4)

    @staticmethod
    def process(img):
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        x_t = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        #x_t = np.reshape(x_t, (64, 64))
        x_t = np.nan_to_num(x_t)

        return x_t.astype(np.uint8)

    def observation(self, observation):
        return ObservationWraperMK.process(observation)

    def reset(self, **kwargs):

        # on a reset we set the health back to 120
        self.player_1_hp = 120
        self.player_2_hp = 120

        # reset the environment
        observation = self.env.reset(**kwargs)

        # we restarted inc the number
        self.num_resets += 1

        # the observation
        obs = self.observation(observation)

        # fill up the queue
        for i in range(4):
            self.q.append(obs)
        
        return np.array(list(self.q))

    def step(self,action):
        observation, reward, done, info = self.env.step(action)
        if info["health"] == 0 :
            self.player_hp = 120
            self.enemy_hp =  120
            reward = 0
        else:
            reward = (info["enemy_rounds_won"]+1)*(info["health"] -self.player_hp) + (info["rounds_won"]+2)*(self.enemy_hp - info["enemy_health"])
            self.player_hp = info['health']
            self.enemy_hp = info["enemy_health"]
        if done:
            self.player_hp = 120
            self.enemy_hp =  120
            reward = 0
        obs = self.observation(observation)
        self.q.append(obs) 
        return np.array(list(self.q)), reward, done, info

            


class PlayerOneNetworkControllerWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super(PlayerOneNetworkControllerWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(12)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))


    def action(self, action):
        return self._actions[action].copy()






def make_env():
    env = retro.make(game='MortalKombatII-Genesis',state='Level1.LiuKangVsJax')
    env = ObservationWraperMK(env)
    env = PlayerOneNetworkControllerWrapper(env)
    env.render()


    return env
