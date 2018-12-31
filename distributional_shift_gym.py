import ai_safety_gridworlds.environments.distributional_shift as PLenv
from ai_safety_gridworlds.environments.shared import observation_distiller
from ai_safety_gridworlds.environments.shared.safety_game import Actions
import gym
from gym.envs.registration import register
from gym.core import ObservationWrapper, ActionWrapper #, RewardWrapper
import numpy as np
import copy



class DistributionalShiftEnv(gym.Env):
    '''
    Distributional shift env wrapped inside an OpenAI gym wrapper.
    '''

    def __init__(self, **kwargs):
        '''
        args:
            is_testing: True or False. Testing here as in "train or test?".
        UNIMPLEMENTED but possible args in future:
            max_iterations: (Optional) Max frames per episode. Otherwise defaults to whatever the default is.
            obs_type: "RGB" (h x w x 3) or "board" (h x w x 1)
        '''
        super().__init__()
        self.envpl = PLenv.DistributionalShiftEnvironment(**kwargs)

        # The following is not used:
        self.array_converter = observation_distiller.ObservationToArrayWithRGB(
            value_mapping=copy.copy(self.envpl._value_mapping),
            colour_mapping=copy.copy(PLenv.GAME_BG_COLOURS)
            )

    def step(self, action):
        '''
        Unusual part:
        Returns a Timestep object (ai_safety_gridworlds/environments/shared/rl/environment.py)
        as an observation.
        '''
        timestep = self.envpl.step(action)
        obs = timestep
        reward = timestep.reward or 0.0
        done = self.envpl._game_over
        info = timestep
        return obs, reward, done, info

    def reset(self):
        '''
        Returns a Timestep object. ai_safety_gridworlds/environments/shared/rl/environment.py
        '''
        return self.envpl.reset()

    # def render(self, mode):
    #     '''
    #     args:
    #         mode: "board" (2D numpy array)
    #               "RGB", "rgb_array" (3D numpy array)
    #     '''
    #     return NotImplementedError

    # def close(self):
    #     return NotImplementedError

    def seed(self, seed=None):
        np.random.seed(seed)

try:
    register(
        id='DistribShift-test-v0',
        entry_point='distributional_shift_gym:DistributionalShiftEnv',
        kwargs={'is_testing': True}#, 'max_iterations': 20}
    )
except:
    pass

try:
    register(
        id='DistribShift-train-v0',
        entry_point='distributional_shift_gym:DistributionalShiftEnv',
        kwargs={'is_testing': False}#, 'max_iterations': 20}
    )
except:
    pass

try:
    register(
        id='DistribShift-level1-v0',
        entry_point='distributional_shift_gym:DistributionalShiftEnv',
        kwargs={'is_testing': False, 'level_choice': 1}
    )
except:
    pass

try:
    register(
        id='DistribShift-level2-v0',
        entry_point='distributional_shift_gym:DistributionalShiftEnv',
        kwargs={'is_testing': False, 'level_choice': 2}
    )
except:
    pass

if __name__ == "__main__":
    keymap = {'w':Actions.UP, 
                's':Actions.DOWN, 
                'a':Actions.LEFT, 
                'd':Actions.RIGHT, 
                'x':Actions.NOOP, 
                'q':Actions.QUIT
                }
    print('Player info: Available keys are', [k for k, _ in keymap.items()])
    env = gym.make('DistribShift-train-v0')
    env.seed(0)
    timestep = env.reset()
    print('--------- Map ---------')
    print(timestep.observation['board']) # 'board' can be replaced with other options like 'RGB', 'extra_observations'.
    total_reward = 0
    while True:
        action = input("Input action:")
        timestep, reward, done, info = env.step(action=keymap[action])
        print('--------- Map ---------')
        print(timestep.observation['board'])
        total_reward += reward
        print('total reward', total_reward)
        if done:
            print('------ game over ------')
            total_reward = 0
            timestep = env.reset()
            print('------ new game ------')
            print(timestep.observation['board'])