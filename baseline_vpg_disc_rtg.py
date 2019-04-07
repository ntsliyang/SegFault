"""
    Baseline model
        - Vanilla Policy Gradient update with reward-to-go.
        - Normal, discrete environment.
"""

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gym
import torch
import torch.optim as optim
from itertools import count
import os
from model.vpg_discrete_rtg import PolicyNet, ValueNet
from utils.utils import plot_durations
from utils.memory import Memory
import json
import sys


# Utils for saving and loading checkpoints

def save_checkpoint(file_dir, policy_net, policynet_optim,
                    i_epoch, policy_lr, **kwargs):
    save_dict = {"policy_net": policy_net.state_dict(),
                 "policynet_optim": policynet_optim.state_dict(),
                 "i_epoch": i_epoch,
                 "policy_lr": policy_lr,
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

    policy_lr = checkpoint["policy_lr"]

    policynet_optim = optim.Adam(policy_net.parameters(), lr=policy_lr)
    policynet_optim.load_state_dict(checkpoint["policynet_optim"])

    checkpoint.pop("policy_net")
    checkpoint.pop("policynet_optim")
    checkpoint.pop("i_epoch")
    checkpoint.pop("policy_lr")

    return policy_net, policynet_optim, checkpoint




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

# Memory parameter
capacity = config['capacity']     # How many trajectories to store

# Training parameters
# num_episodes = 1000
i_epoch = config['i_epoch']      # This would determine which checkpoint to load, if the checkpoint exists
batch_size = config['batch_size']
policy_lr = config['policy_lr']

num_vn_iter = config['num_vn_iter']    # Number of iterations to train value net per epoch

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

# Create the model
policy_net = PolicyNet(layer_sizes).to(device)      # Policy network

# Set up memory
memory = Memory(capacity, GAMMA, LAMBDA, device)

# Set up optimizer
policynet_optimizer = optim.Adam(policy_net.parameters())

###################################################################
# Start training

# Dictionary for extra training information to save to checkpoints
training_info = {"epoch mean durations" : [],
                 "epoch mean rewards" : [],
                 "max reward achieved": 0,
                 "past %d epochs mean reward" % num_avg_epoch: 0}

# Batch that records trajectories

# To record epoch stats
epoch_durations = []
epoch_rewards = []

while True:

    finished_rendering_this_epoch = False

    # Every save_ckpt_interval, Check if there is any checkpoint.
    # If there is, load checkpoint and continue training
    # Need to specify the i_episode of the checkpoint intended to load
    if i_epoch % save_ckpt_interval == 0 and os.path.isfile(os.path.join(ckpt_dir, "ckpt_eps%d.pt" % i_epoch)):
        policy_net, policynet_optimizer, training_info = \
            load_checkpoint(ckpt_dir, i_epoch, layer_sizes, input_size, device=device)

    # To record episode stats
    episode_durations = []
    episode_rewards = []

    for i_episode in range(batch_size):

        # Keep track of the running reward
        running_reward = 0

        # Initialize the environment and state
        current_state = env.reset()

        # Store the first state and value estimate in memory
        memory.set_initial_state(current_state)

        for t in count():
            # Make sure that policy net and value net is in training mode
            policy_net.train()

            # Sample an action given the current state
            action, log_prob = policy_net(torch.tensor([current_state], device=device))
            log_prob = log_prob.squeeze()

            # Interact with the environment
            next_state, reward, done, _ = env.step(action.item())
            running_reward += reward

            # Render this episode
            if render and (render_each_episode or (not finished_rendering_this_epoch)):
                env.render()

            # Record transition in memory
            memory.add_transition(action, log_prob, next_state, extrinsic_reward=reward)

            # Update current state
            current_state = next_state

            if done:
                # Load and print episode stats after each episode ends
                episode_durations.append(t + 1)
                episode_rewards.append(running_reward)
                if running_reward > training_info["max reward achieved"]:
                    training_info["max reward achieved"] = running_reward

                # Decide whether to render next episode
                if not(render_each_episode):
                    finished_rendering_this_epoch = True

                break

    # At the end of each epoch

    # Record epoch stats
    epoch_durations.append(sum(episode_durations) / batch_size)
    epoch_rewards.append(sum(episode_rewards) / batch_size)

    # Optimize the PolicyNet for one step after collecting enough trajectories
    policy_net.optimize_model(memory, batch_size, policynet_optimizer, device=device)

    # Reset Flags
    if not(render_each_episode):
        finished_rendering_this_epoch = False

    # Record stats
    training_info["epoch mean durations"].append(epoch_durations[-1])
    training_info["epoch mean rewards"].append(epoch_rewards[-1])
    if (i_epoch + 1) % num_avg_epoch:
        training_info["past %d epochs mean reward" %  (num_avg_epoch)] = \
            (sum(training_info["epoch mean rewards"][-num_avg_epoch:]) / num_avg_epoch) \
                if len(training_info["epoch mean rewards"]) >= num_avg_epoch else 0

    # Print stats
    print("\n\n=============  Epoch: %d  =============" % (i_epoch + 1))
    print("epoch mean durations: %f" % (epoch_durations[-1]))
    print("epoch mean rewards: %f" % (epoch_rewards[-1]))
    print("Max reward achieved: %f" % training_info["max reward achieved"])

    # Plot stats
    if plot:
        plot_durations(training_info["epoch mean rewards"])

    # Update counter
    i_epoch += 1

    # Every save_ckpt_interval, save a checkpoint according to current i_episode.
    if i_epoch % save_ckpt_interval == 0:
        save_checkpoint(ckpt_dir, policy_net, policynet_optimizer, i_epoch,
                        policy_lr=policy_lr, **training_info)

