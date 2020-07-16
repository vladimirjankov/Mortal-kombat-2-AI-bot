
import numpy as np
import gym
import retro
from collections import deque 
import tensorflow as tf

from gym import wrappers 
import cv2

# ObesrvationWraperClass for mortal kombat 
class ObservationWraperMK(gym.ObservationWrapper):

    def __init__(self, env):
        #base observer
        super(ObservationWraperMK, self).__init__(env)
        
        #number of resets in env
        self.num_resets = 0 
        
        # players hp set to max at beginning (bot in training)
        self.player_hp = 120 
        
        #number of current frame
        self.current_frame_number = 0
        
        #number of frames to skip
        self.frame_skipping = 2
        
        #in game bot hp
        self.enemy_hp = 120
        
        #set gym observation space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(160, 112))
        
        #init deque
        self.q = deque(maxlen=4)

    @staticmethod
    def process(img):
        """
        
        process function of each step, reshapes the image 
        and filters it with laplacian, converts the image 
        to uint8 type
        
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        x_t = cv2.resize(img, (112, 160), interpolation=cv2.INTER_AREA)
        x_t = np.nan_to_num(x_t)
        x_t = cv2.Laplacian(x_t,cv2.CV_8U)

        return x_t.astype(np.uint8)

    def observation(self, observation):
        """
    
        returns a observation 
    
        """
        return ObservationWraperMK.process(observation)

    def reset(self, **kwargs):

        """
    
        resets the environment and sets the player hp to init state
    
        """

        # on a reset we set the health back to 120
        self.player_hp = 120
        self.enemy_hp = 120

        # reset the environment
        
        observation = self.env.reset(**kwargs)

        # we restarted inc the number
        self.num_resets += 1

        # the observation
        obs = self.observation(observation)
        self.current_frame_number = 0
        
        # fill up the queue
        for i in range(4):
            self.q.append(obs)
        
        return np.array(list(self.q))

    def step(self,action):
        """
    
        step function, returns reward and checks if done reward 
        is given based on difference of hp health and enemy_health.
    
        """
        observation, reward, done, info = self.env.step(action)
        if info["health"] <= 0 or info["enemy_health"] <= 0:
            self.player_hp = 120
            self.enemy_hp =  120
            reward = 0
        else:
            self.player_hp = info['health']
            self.enemy_hp = info["enemy_health"]
            reward =   self.player_hp - self.enemy_hp


        if info["enemy_rounds_won"] == 2 or info["rounds_won"] == 2:
            self.player_hp = 120
            self.enemy_hp =  120
            reward = 0
            done = True

        obs = self.observation(observation)
        if self.current_frame_number == self.frame_skipping:
            self.q.append(obs)
            self.current_frame_number = 0 
        self.current_frame_number += 1
        reward = reward / 120 +1
        return np.array(list(self.q)), reward, done, info

            
class PlayerOneNetworkControllerWrapper(gym.ActionWrapper):

    def __init__(self, env):
        """

        sets the buttons that are being used by player one bot
        multyple combination can be set. Inits action space from
        gym module.

        """
        super(PlayerOneNetworkControllerWrapper, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'],['LEFT', 'UP'],['RIGHT', 'UP'],
                   ['DOWN', 'B'],['LEFT', 'UP'],['RIGHT', 'DOWN','B'],['RIGHT', 'DOWN','A'],
                   ['RIGHT', 'UP','B'],['RIGHT', 'UP','A'],['RIGHT', 'UP','C'],
                   ['LEFT', 'UP','B'],['LEFT', 'UP','A'],['LEFT', 'UP','C'],
                   ['C'],['START'], ['B'],['Y'],['X'],['Z'],['A'],['UP'],['MODE']]
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
    """
    creates and mortal kombat 2 sega genesis environment
    with observation wrapper and controller wrapper
    """
    aigym_path = "video/"
    env = retro.make(game='MortalKombatII-Genesis',state='Level1.LiuKangVsJax')
    env = wrappers.Monitor(env, aigym_path,video_callable=False  ,force=True) #, video_callable=False 
    env = ObservationWraperMK(env)
    env = PlayerOneNetworkControllerWrapper(env)
    env._max_episode_steps = 350
    #env.render()

    return env
