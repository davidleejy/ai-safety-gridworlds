#!/usr/bin/env python3

import argparse
import numpy as np
import gym
import time
import torch
from torch_rl.utils.penv import ParallelEnv
from wrappers_gridworld import ActWrapper, ObsWrapper

from distributional_shift_gym import DistributionalShiftEnv # register env with gym.
# try:
#     import gym_minigrid
# except ImportError:
#     pass

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="The number of worse episodes to show")
parser.add_argument("--obs_type", default='board',
                    help="The kind of observation type to use. Options: RGB, board.")
parser.add_argument("--port", type=int, default=-1,
                    help="Port for using visdom. Remember to start the visdom server beforehand.")         
args = parser.parse_args()

if args.port > 0:
    import visdom
    viz = visdom.Visdom(port=args.port)
    attention_windows = []
    for p in range(args.procs):
        # initialize
        attention_windows.append(viz.heatmap(X=torch.arange(end=9).view(3,3), 
                                    opts=dict(columnnames=['coldidx0', 'colidx1', 'colidx2'],
                                                rownames=['rowidx0', 'rowidx1', 'rowidx2'],
                                                colormap='Jet', 
                                                title='proc {} attention'.format(p))))
    print('This is X', torch.arange(end=9).view(3,3))
    print('Heatmap will plot this flipped across a horizontal line.')
    # exit() # uncomment to see plot.

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(args.seed + 10000*i)
    env = ActWrapper(env)
    env = ObsWrapper(args.obs_type, env)
    envs.append(env)
env = ParallelEnv(envs)

# Define agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(args.env, env.observation_space, model_dir, args.argmax, args.procs)
print("CUDA available: {}\n".format(torch.cuda.is_available()))

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": []}

# Run the agent

start_time = time.time()

obss = env.reset()

log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=agent.device)
log_episode_num_frames = torch.zeros(args.procs, device=agent.device)


# print(agent.acmodel)
# actions = agent.get_actions(obss)
# for name, param in agent.acmodel.named_parameters():
#     print(name)
# mods = agent.acmodel.relational_block.modules()
# print(type(mods))
# for m in mods:
#     print(m)

# exit()

def to_visdom_heatmap_labels(obs, H, W, n_heads):
    # Visdom heatmap labels can't have repeats.
    # Heatmap labels should preferably NOT be numbers.
    # args:
    #   obs is a list of nubmers.
    #   H height, W width, n_heads number of attention heads.
    # returns:
    #   list of symbols with position coords [ '00w', '01-', '02A', ..]
    map = {0:'w', 1:'-', 2:'a', 3:'G', 4:'L'}
    coords=[]
    for w in range(W):
        for h in range(H):
            coords.append('{}{}'.format(w,h))
    rownames = ['{}{}'.format(c, map[x]) for x, c in zip(obs, coords)]
    #
    head_idxs = np.arange(n_heads).reshape(-1,1)
    head_idxs = np.tile(head_idxs, reps=[1, H*W])
    head_idxs = head_idxs.reshape(1,-1).squeeze().tolist()
    prefixes = rownames.copy() * 3
    columnnames = ['{}{}'.format(a, b) for a, b in zip(prefixes, head_idxs)]
    return rownames, columnnames


while log_done_counter < args.episodes:
    actions, values, info = agent.get_actions(obss)
    
    if 'viz' in locals():
        # print('attention', info['attention'].shape)
        attention_processes = info['attention'].clone()
        for p in range(args.procs):
            A = attention_processes[p,:,:,:]
            n_heads = A.shape[0]
            # print(A.shape)
            A = torch.cat([A[head,::] for head in range(n_heads)], dim=-1)
            print(len(obss), obss[p].shape, type(obss[p]))
            obs = obss[p][:,:,0].flatten().astype(dtype=int) #.astype(dtype=str, copy=False)
            # .transpose().numpy()
            print(obs.shape, list(obs))
            rownames, columnnames = to_visdom_heatmap_labels(list(obs), obss[p].shape[0], obss[p].shape[1], n_heads)
            
            print(rownames)
            print(columnnames)
            # obs.numpy()

            viz.heatmap(X=A, opts={'rownames':rownames, 'columnnames':columnnames, 'colormap':'Jet', 
                                                'title':'proc {} attention'.format(p) }, win=attention_windows[p])
            # viz.heatmap(A, opts=dict(rownames=to_visdom_labels(list(obs))))#, win=attention_windows[p])
            # viz.heatmap(A, win=attention_windows[p])
            # update with albels.


        time.sleep(20)
    
    obss, rewards, dones, _ = env.step(actions)
    agent.analyze_feedbacks(rewards, dones)

    log_episode_return += torch.tensor(rewards, device=agent.device, dtype=torch.float)
    log_episode_num_frames += torch.ones(args.procs, device=agent.device)

    for i, done in enumerate(dones):
        if done:
            log_done_counter += 1
            logs["return_per_episode"].append(log_episode_return[i].item())
            logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

    mask = 1 - torch.tensor(dones, device=agent.device, dtype=torch.float)
    log_episode_return *= mask
    log_episode_num_frames *= mask

end_time = time.time()

# Print logs

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
      .format(num_frames, fps, duration,
              *return_per_episode.values(),
              *num_frames_per_episode.values()))

# Print worst episodes

n = args.worst_episodes_to_show
if n > 0:
    print("\n{} worst episodes:".format(n))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))