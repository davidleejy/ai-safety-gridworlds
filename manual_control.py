from ai_safety_gridworlds.environments.shared.safety_game import Actions
import gym
import distributional_shift_gym

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