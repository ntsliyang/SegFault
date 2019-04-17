"""
    Baseline model
        - Proximal Policy Optimization with a value net estimating state value and update policy with GAE.
        - Normal, discrete environment.

    Use Q-function network to fit intrinsic exploration bonus return
    Use value-function network to fit extrinsic reward return
"""

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
import os
from model.ppo_discrete import PolicyNet, ValueNet
from utils.utils2 import plot_durations
from utils.memory import Memory
from utils.onehot import to_one_hot
from Hashing.SimHash import SimHash
from LPLGraph.LPLGraph import LPLGraph
import json
import sys
import copy
from tqdm import tqdm, tqdm_gui


# Utils for saving and loading checkpoints

def save_checkpoint(file_dir, policy_net, value_net_in, value_net_ex,
                    valuenet_in_optim, valuenet_ex_optim, simhash,
                    i_epoch, **kwargs):
    save_dict = {"policy_net": policy_net.state_dict(),
                 "value_net_in": value_net_in.state_dict(),
                 "value_net_ex": value_net_ex.state_dict(),
                 "valuenet_in_optim": valuenet_in_optim.state_dict(),
                 "valuenet_ex_optim": valuenet_ex_optim.state_dict(),
                 "i_epoch": i_epoch,
                 # "lpl_graph": lpl_graph,
                 "simhash": simhash
                 }
    # Save optional contents
    save_dict.update(kwargs)

    # Create the directory if not exist
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    file_name = os.path.join(file_dir, "ckpt_eps%d.pt" % i_epoch)

    # Delete the file if the file already exist
    try:
        os.remove(file_name)
    except OSError:
        pass

    # Save the file
    torch.save(save_dict, file_name)


def load_checkpoint(file_dir, i_epoch, layer_sizes, input_size, device='cuda'):
    checkpoint = torch.load(os.path.join(file_dir, "ckpt_eps%d.pt" % i_epoch), map_location=device)

    policy_net = PolicyNet(layer_sizes).to(device)
    policy_net.load_state_dict(checkpoint["policy_net"])
    policy_net.train()

    value_net_in = ValueNet(input_size).to(device)
    value_net_in.load_state_dict(checkpoint["value_net_in"])
    value_net_in.train()

    value_net_ex = ValueNet(input_size).to(device)
    value_net_ex.load_state_dict(checkpoint["value_net_ex"])
    value_net_ex.train()

    valuenet_in_optim = optim.Adam(value_net_in.parameters())
    valuenet_in_optim.load_state_dict(checkpoint["valuenet_in_optim"])

    valuenet_ex_optim = optim.Adam(value_net_ex.parameters())
    valuenet_ex_optim.load_state_dict(checkpoint["valuenet_ex_optim"])

    # lpl_graph = checkpoint["lpl_graph"]
    simhash = checkpoint["simhash"]

    checkpoint.pop("policy_net")
    checkpoint.pop("value_net_in")
    checkpoint.pop("value_net_ex")
    checkpoint.pop("valuenet_in_optim")
    checkpoint.pop("valuenet_ex_optim")
    checkpoint.pop("i_epoch")

    return policy_net, value_net_in, value_net_ex, valuenet_in_optim, valuenet_ex_optim,\
            simhash, checkpoint


# Load command line arguments

if len(sys.argv) < 2:
    print("Please specify a config file")
    sys.exit(1)


config = json.load(open(sys.argv[1]))



#######################  Parameters  ##############################

# Environment parameter
env_name = config['env_name']
is_unwrapped = config['is_unwrapped']

# Model hyperparameters
input_size = config['input_size']      # Size of state
output_size = config['output_size']     # Number of discrete actions
layer_sizes = config['layer_sizes']         # The MLP network architecture

ckpt_dir = config['ckpt_dir']
save_ckpt_interval = config['save_ckpt_interval']

# Hashing parameter
len_hashcode = config['len_hashcode']
use_preprocessor = config['use_preprocessor']

# Graph parameter
max_reward = config['max_reward']
num_particles = config['num_particles']
curiosity_weight = config['curiosity_weight']

# Memory parameter
capacity = config['capacity']     # How many trajectories to store

# Training parameters
# num_episodes = 1000
i_epoch = config['i_epoch']      # This would determine which checkpoint to load, if the checkpoint exists
batch_size = config['batch_size']

num_vn_iter = config['num_vn_iter']    # Number of iterations to train value net per epoch
num_updates_per_epoch = config['num_updates_per_epoch']
clip_range = config['clip_range']

GAMMA = config['GAMMA']
LAMBDA = config['LAMBDA']
EPS_START = config['EPS_START']
EPS_END = config['EPS_END']
EPS_DECAY = config['EPS_DECAY']

# Rendering and recording options
render = config['render']
plot = config['plot']

render_each_episode = config['render_each_episode']     # Whether to render each episode
                                #   If set to true, then each episode the agent ever endure will be rendered
                                #   Otherwise, only each episode at the start of each epoch will be rendered
                                #   Note: each epoch has exactly 1 model update and batch_size episodes

# record_each_episode_stats = False   # Whether to record the statistics of each episode
                                    #   If set to true, then each episode the agent ever endure will be recorded
                                    #   Otherwise, only each episode at the start of each epoch will be recorded

num_avg_epoch = config['num_avg_epoch']       # The number of epochs to take for calculating average stats

###################################################################


# Turn on pyplot's interactive mode
# VERY IMPORTANT because otherwise training stats plot will hault
plt.ion()

# Create OpenAI gym environment
env = gym.make(env_name)
if is_unwrapped:
    env = env.unwrapped

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current usable device is: ", device)

# Create the model. Two value net
policy_net = PolicyNet(layer_sizes).to(device)      # Policy network
value_net_ex = ValueNet(input_size).to(device)         # Value network for extrinsic reward
value_net_in = ValueNet(input_size + 1 + output_size).to(device)      # One additional input unit to indicate trajectory number

# Set up optimizer
valuenet_in_optimizer = optim.Adam(value_net_in.parameters())
valuenet_ex_optimizer = optim.Adam(value_net_ex.parameters())

# Set up memory
memory = Memory(capacity, GAMMA, LAMBDA, device=device)


# Define observation normalization function. Normalize state vector values to range [-1., 1.]
def state_nomalize(s):
    # Obtain environment observation space limit
    high = env.observation_space.high
    low = env.observation_space.low
    return ((s - low) / (high - low)) * 2 - 1


# Create Hashing function
simhash = SimHash(input_size, len_hashcode, preprocessor=state_nomalize if use_preprocessor else None)

# Create LPL Graph
graph = LPLGraph(len_hashcode, output_size, maximum_reward=max_reward, num_particles=num_particles)

# Set up action counter to infer the dominating action
act_counter = np.zeros((output_size,), dtype=np.int32)


###################################################################
# Start training

# Dictionary for extra training information to save to checkpoints
training_info = {"epoch mean durations" : [],
                 "epoch mean rewards" : [],
                 "max reward achieved": 0,
                 "past %d epochs mean reward" % num_avg_epoch: 0,
                 "extrinsic value net loss": [],
                 "intrinsic value net loss": [],}

# Batch that records trajectories

# To record epoch stats
epoch_durations = []
epoch_rewards = []

while True:

    # print("\n------------- Epoch %d -------------" % (i_epoch + 1))

    finished_rendering_this_epoch = False

    # Every save_ckpt_interval, Check if there is any checkpoint.
    # If there is, load checkpoint and continue training
    # Need to specify the i_episode of the checkpoint intended to load
    # if i_epoch % save_ckpt_interval == 0 and os.path.isfile(os.path.join(ckpt_dir, "ckpt_eps%d.pt" % i_epoch)):
    #     policy_net, value_net_in, value_net_ex, valuenet_in_optimizer, valuenet_ex_optimizer,\
    #     simhash, training_info = \
    #         load_checkpoint(ckpt_dir, i_epoch, layer_sizes, input_size, device=device)
    #     print("\n\tCheckpoint successfully loaded!\n")

    # To record episode stats
    episode_durations = []
    episode_rewards = []

    # Use value net in evaluation mode when collecting trajectories
    value_net_in.eval()
    value_net_ex.eval()

    ###################################################################
    # Collect trajectories

    print("\n\n\tCollecting %d episodes: " % (batch_size))

    for i_episode in tqdm(range(batch_size)):       # Use tqdm to show progress bar

        # Keep track of the running reward
        running_reward = 0

        # Initialize the environment and state
        current_state = env.reset()

        # TODO: Change codes below

        # Estimate the value of the initial state
        ex_val = value_net_ex(torch.tensor([current_state], dtype=torch.float32, device=device)).squeeze()      # squeeze the dimension
        in_val = value_net_in(torch.tensor([np.concatenate((current_state, [i_episode]), axis=0)],
                              dtype=torch.float32, device=device)).squeeze()    # provide i_episode as additional info as input

        # Store the first state and value estimate in memory
        memory.set_initial_state(current_state, initial_ex_val_est=ex_val, initial_in_val_est=in_val)

        # Obtain current state hash code
        current_state_hash = simhash.hash(current_state)

        for t in count():

            # Sample an action given the current state
            action, log_prob = policy_net(torch.tensor([current_state], dtype=torch.float32, device=device))
            log_prob = log_prob.squeeze()

            # Interact with the environment
            next_state, reward, done, _ = env.step(action.item())
            running_reward += reward

            # Estimate the value of the next state
            ex_val = value_net_ex(torch.tensor([next_state], dtype=torch.float32, device=device)).squeeze()     # squeeze the dimension
            in_val = value_net_in(torch.tensor([np.concatenate((next_state, [i_episode]), axis=0)],
                                  dtype=torch.float32, device=device)).squeeze()    # provide i_episode as additional info as input

            # Obtain next state hash code
            next_state_hash = simhash.hash(next_state)

            # Update action counter
            act_counter[action.item()] += 1

            # If next state hashed to a different code than the current state, then infer the dominating action,
            #   update causal link, and clear action counter
            if next_state_hash != current_state_hash:
                main_action = np.argmax(act_counter)
                graph.update_transition(current_state_hash, main_action, next_state_hash)
                act_counter = np.zeros((output_size,), dtype=np.int32)

            # Take the action confidence with current state hash code as the intrinsic reward
            in_reward = curiosity_weight * graph.action_confidence(current_state_hash, action.item())
            # in_reward = curiosity_weight * np.sqrt(in_reward)       # Take the square root of confidence value

            # Record transition in memory
            memory.add_transition(action, log_prob, next_state,
                                  extrinsic_reward=reward, extrinsic_value_estimate=ex_val,
                                  intrinsic_reward=in_reward, intrinsic_value_estimate=in_val)
            # memory.add_transition(action, log_prob, next_state,
            #                       extrinsic_reward=running_reward if done else 0., extrinsic_value_estimate=ex_val,
            #                       intrinsic_reward=in_reward, intrinsic_value_estimate=in_val)


            # Update current state
            current_state = next_state
            current_state_hash = next_state_hash

            # Render this episode
            if render and (render_each_episode or (not finished_rendering_this_epoch)):
                env.render()

            if done:
                # Load and print episode stats after each episode ends
                episode_durations.append(t + 1)
                episode_rewards.append(running_reward)
                if running_reward > training_info["max reward achieved"]:
                    training_info["max reward achieved"] = running_reward

                # Decide whether to render next episode
                if not render_each_episode:
                    finished_rendering_this_epoch = True

                break

    ###################################################################
    # optimize the model


    # Optimize the PolicyNet for a given number of steps
    # policy_candiate = PolicyNet(layer_sizes).to(device)
    policy_candidate = copy.deepcopy(policy_net).to(device)

    # initialize the optimizer for policy_net
    policy_candidate_optimizer = optim.Adam(policy_candidate.parameters())

    ex_gae = memory.extrinsic_gae(batch_size)
    in_gae = memory.intrinsic_gae(batch_size)
    old_act_log_prob = memory.act_log_prob(batch_size)
    states = memory.states(batch_size)
    actions = memory.actions(batch_size)

    # Proximal Policy Optimization
    loss = 0

    print("\n\n\tUpdate Policy Net for %d steps:" % (num_updates_per_epoch))

    for i in tqdm(range(num_updates_per_epoch)):        # Use tqdm to show progress bar

        num = 0
        for j in range(batch_size):
            # Calculate the log probabilities of the actions stored in memory from the distribution parameterized by the
            #   new candidate network
            new_act_log_prob = policy_candidate(states[j][:-1], action_query=actions[j].squeeze())   # Igno re last state
            ratio = torch.exp(new_act_log_prob - old_act_log_prob[j].detach())      # Detach old action log prob

            if torch.sum(torch.abs(ratio - 1) > clip_range) == ratio.shape[0]:
                print("\t\tFully Clipped!")

            gae = ex_gae[j] + in_gae[j]          # intrinsic gae + extrinsic gae

            surr1 = ratio * gae
            surr2 = (((gae < 0.).type(torch.float32) * (1 - clip_range) +
                      (gae > 0.).type(torch.float32) * (1 + clip_range))) * gae
            loss += - torch.sum(torch.min(surr1, surr2))
            num += ratio.shape[0]

        loss /= torch.tensor(num, device=device, dtype=torch.float32)

        policy_candidate_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm(policy_candidate.parameters(), 1.)      # Clip gradients
        policy_candidate_optimizer.step()

    policy_net = copy.deepcopy(policy_candidate).to(device)

    # Optimize value net for a given number of steps
    # Set value net in training mode
    value_net_in.train()
    value_net_ex.train()
    ex_rtg = memory.extrinsic_discounted_rtg(batch_size)      # Use undiscounted reward-to-go to fit the value net
    in_rtg = memory.intrinsic_rtg(batch_size)
    ex_val_est = []
    in_val_est = []

    print("\n\n\tUpdate Value Net for %d steps" % (num_vn_iter))

    for i in tqdm(range(num_vn_iter)):          # Use tqdm to show progress bar
        for j in range(batch_size):
            in_val_traj = value_net_in(torch.cat([states[j], torch.ones((states[j].shape[0], 1), dtype=torch.float32, device=device) * j], dim=1)).squeeze()
            ex_val_traj = value_net_ex(states[j]).squeeze()
            in_val_est.append(in_val_traj)
            ex_val_est.append(ex_val_traj)

        in_value_net_mse = value_net_in.optimize_model(in_val_est, in_rtg, valuenet_in_optimizer)
        ex_value_net_mse = value_net_ex.optimize_model(ex_val_est, ex_rtg, valuenet_ex_optimizer)

    # Reset Flags
    if not(render_each_episode):
        finished_rendering_this_epoch = False

    ###################################################################


    # Record epoch stats
    epoch_durations.append(sum(episode_durations) / batch_size)
    epoch_rewards.append(sum(episode_rewards) / batch_size)

    training_info["epoch mean durations"].append(epoch_durations[-1])
    training_info["epoch mean rewards"].append(epoch_rewards[-1])
    training_info["extrinsic value net loss"].append(ex_value_net_mse)
    training_info["intrinsic value net loss"].append(in_value_net_mse)
    if (i_epoch + 1) % num_avg_epoch:
        training_info["past %d epochs mean reward" %  (num_avg_epoch)] = \
            (sum(training_info["epoch mean rewards"][-num_avg_epoch:]) / num_avg_epoch) \
                if len(training_info["epoch mean rewards"]) >= num_avg_epoch else 0

    # Print stats
    print("\n\n=============  Epoch: %d  =============" % (i_epoch + 1))
    print("epoch mean durations: %f" % (epoch_durations[-1]))
    print("epoch mean rewards: %f" % (epoch_rewards[-1]))
    print("Max reward achieved: %f" % training_info["max reward achieved"])
    print("extrinsic value net loss: %f" % ex_value_net_mse)
    print("intrinsic value net loss: %f" % in_value_net_mse)

    # Plot stats
    if plot:
        plot_durations(training_info["epoch mean rewards"],
                       training_info["extrinsic value net loss"], training_info["intrinsic value net loss"])

    # Update counter
    i_epoch += 1

    # Every save_ckpt_interval, save a checkpoint according to current i_episode.
    # if i_epoch % save_ckpt_interval == 0:
    #     save_checkpoint(ckpt_dir, policy_net, value_net_in, value_net_ex,
    #                     valuenet_in_optimizer, valuenet_ex_optimizer, simhash, i_epoch, **training_info)

