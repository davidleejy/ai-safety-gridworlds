import gym
import gym.spaces
from ai_safety_gridworlds.environments.shared.safety_game import Actions

import numpy as np
import copy


class ActWrapper(gym.core.ActionWrapper):
    '''
    An action wrapper to be applied.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.envpl = self.env.envpl     # to make wrapper compatible with gym.Env object that has a vital non-standard attribute _envpl_ that needs to be accessible with _self.envpl_ .
        self.action_map = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT] #, Actions.NOOP] TODO  re-include?
        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def step(self, action):
        action = self.action(action)
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def action(self, action):
        return self.action_map[action]


class ObsWrapper(gym.core.ObservationWrapper):
    '''
    An observation wrapper to be applied.
    '''

    def __init__(self, obs_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.envpl = self.env.envpl # to make wrapper compatible with gym.Env object that has a vital non-standard attribute _envpl_ that needs to be accessible with _self.envpl_ .
        # Decide on the observation format:
        self.obs_type = obs_type
        if self.obs_type == 'RGB':

            def modify(self, timestep):
                return np.moveaxis(timestep.observation['RGB'], 0, -1) # shape (h,w,3)
            #
            shape = self.env.envpl._observation_spec['RGB'].shape
            self.observation_space = gym.spaces.Box(low=0, 
                                                    high=255,
                                                    shape=(shape[1], shape[2], shape[0]), # (h,w,3) 3 is rgb chans.
                                                    dtype=np.uint8)

        elif self.obs_type == 'board':

            def modify(self, timestep):
                X = copy.copy(np.expand_dims(timestep.observation['board'], axis=-1))
                return X
            #
            shape = self.env.envpl._observation_spec['board'].shape
            self.observation_space = gym.spaces.Box(low=0,
                                                    high=len(self.env.envpl._value_mapping)-1,
                                                    shape=(shape[0], shape[1], 1), # (h,w,1)
                                                    dtype=np.uint8)
        else:
            NotImplementedError
        #
        import types
        self.observation = types.MethodType(modify, self) # override
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        raise NotImplementedError("Override this in __init__() like: self.observation = types.MethodType(<function object>, self).")