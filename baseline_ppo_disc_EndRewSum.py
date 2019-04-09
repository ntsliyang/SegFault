"""
    Baseline model
        - Proximal Policy Optimization with a value net estimating state value and update policy using GAE.
        - discrete environment.
        - Rewards along an episode is summed up to a trajectory return. Only this trajectory return is fed to the
            agent at the end of each episode.
"""

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
import os
from model.ppo_discrete import PolicyNet, ValueNet
from utils.utils import plot_durations
from utils.memory import Memory
import json
import sys
import copy
from tqdm import tqdm, tqdm_gui


# Utils for saving and loading checkpoints

def save_checkpoint(file_dir, policy_net, value_net, valuenet_optim,
                    i_epoch, policy_lr, valuenet_lr, **kwargs):
    save_dict = {"policy_net": policy_net.state_dict(),
                 "value_net": value_net.state_dict(),
                 "valuenet_optim": valuenet_optim.state_dict(),
                 "i_epoch": i_epoch,
                 "policy_lr": policy_lr,
                 "valuenet_lr": valuenet_lr
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
    value_net = ValueNet(input_size).to(device)
    policy_net.load_state_dict(checkpoint["policy_net"])
    policy_net.train()
    value_net.load_state_dict(checkpoint["value_net"])
    value_net.train()

    policy_lr = checkpoint["policy_lr"]
    valuenet_lr = checkpoint["valuenet_lr"]

    valuenet_optim = optim.Adam(value_net.parameters(), lr=valuenet_lr)
    valuenet_optim.load_state_dict(checkpoint["valuenet_optim"])

    checkpoint.pop("policy_net")
    checkpoint.pop("value_net")
    checkpoint.pop("valuenet_optim")
    checkpoint.pop("i_epoch")
    checkpoint.pop("policy_lr")
    checkpoint.pop("valuenet_lr")

    return policy_net, value_net, valuenet_optim, checkpoint




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
valuenet_lr = config['valuenet_lr']

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

# Create the model
policy_net = PolicyNet(layer_sizes).to(device)      # Policy network
value_net = ValueNet(input_size).to(device)         # Value network

# Set up memory
memory = Memory(capacity, GAMMA, LAMBDA, device)

# Set up optimizer
policynet_optimizer = optim.Adam(policy_net.parameters())
valuenet_optimizer = optim.Adam(value_net.parameters())

###################################################################
# Start training

# Dictionary for extra training information to save to checkpoints
training_info = {"epoch mean durations" : [],
                 "epoch mean rewards" : [],
                 "max reward achieved": 0,
                 "past %d epochs mean reward" % num_avg_epoch: 0,
                 "value net loss": []}

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
    if i_epoch % save_ckpt_interval == 0 and os.path.isfile(os.path.join(ckpt_dir, "ckpt_eps%d.pt" % i_epoch)):
        policy_net, value_net, valuenet_optimizer, training_info = \
            load_checkpoint(ckpt_dir, i_epoch, layer_sizes, input_size, device=device)

    # To record episode stats
    episode_durations = []
    episode_rewards = []

    # Use value net in evaluation mode when collecting trajectories
    value_net.eval()

    ###################################################################
    # Collect trajectories

    print("\n\n\tCollecting %d episodes: " % (batch_size))

    for i_episode in tqdm(range(batch_size)):       # Use tqdm to show progress bar

        # Keep track of the running reward
        running_reward = 0

        # Initialize the environment and state
        current_state = env.reset()

        # Estimate the value of the initial state
        ex_val = value_net(torch.tensor([current_state], device=device)).squeeze()      # squeeze the dimension

        # Store the first state and value estimate in memory
        memory.set_initial_state(current_state, initial_ex_val_est=ex_val)

        for t in count():

            # Sample an action given the current state
            action, log_prob = policy_net(torch.tensor([current_state], device=device))
            log_prob = log_prob.squeeze()

            # Interact with the environment
            next_state, reward, done, _ = env.step(action.item())
            running_reward += reward

            # Estimate the value of the next state
            value = value_net(torch.tensor([next_state], device=device)).squeeze()     # squeeze the dimension

            # Render this episode
            if render and (render_each_episode or (not finished_rendering_this_epoch)):
                env.render()

            # Record transition in memory
            # Only reward given at the end of the episode is fed to the agent
            if done:
                memory.add_transition(action, log_prob, next_state, extrinsic_reward=running_reward,
                                      extrinsic_value_estimate=value)
            else:
                memory.add_transition(action, log_prob, next_state, extrinsic_reward=0.,
                                      extrinsic_value_estimate=value)

            # Update current state
            current_state = next_state

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

    # copy weights from policy_net to the policy_candidate
    # policy_candidate.load_state_dict(policy_net.state_dict())

    # initialize the optimizer for policy_net
    policy_candidate_optimizer = optim.Adam(policy_candidate.parameters())

    ex_gae = memory.extrinsic_gae(batch_size)
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
            # _, new_act_log_prob = policy_candidate(states[j][:-1])       # Ignore last state
            new_act_log_prob = policy_candidate(states[j][:-1], action_query=actions[j].squeeze())   # Ignore last state
            ratio = torch.exp(new_act_log_prob - old_act_log_prob[j].detach())      # Detach old action log prob

            if torch.sum(torch.abs(ratio - 1) > clip_range) == ratio.shape[0]:
                print("\t\tFully Clipped!")

            surr1 = ratio * ex_gae[j]
            surr2 = (((ex_gae[j] < 0.).type(torch.float32) * (1 - clip_range) +
                      (ex_gae[j] > 0.).type(torch.float32) * (1 + clip_range))) * ex_gae[j]
            # surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * ex_gae[j]
            # loss += - torch.mean(torch.min(surr1, surr2))
            loss += - torch.sum(torch.min(surr1, surr2))
            num += ratio.shape[0]

            # traj_loss = 0
            # for k in range(len(ex_gae[j])):
            #     _, new_act_log_prob = policy_candiate(states[j][k])
            #     ratio = torch.exp(new_act_log_prob.squeeze() - old_act_log_prob[j][k])
            #     surr1 = ratio * ex_gae[j][k]
            #     surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * ex_gae[j][k]
            #     traj_loss += - torch.min(surr1, surr2)
            # loss += traj_loss / ex_gae[j].shape[0]

        # loss /= torch.tensor(batch_size, device=device, dtype=torch.float32)
        loss /= torch.tensor(num, device=device, dtype=torch.float32)

        policy_candidate_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm(policy_candidate.parameters(), 1.)      # Clip gradients
        policy_candidate_optimizer.step()

    # policy_net.load_state_dict(policy_candiate.state_dict())
    policy_net = copy.deepcopy(policy_candidate).to(device)

    # # Vanilla Policy Gradient
    # for gae, act_log_prob in zip(ex_gae, old_act_log_prob):
    #     loss += - torch.sum(gae * act_log_prob)
    # loss /= torch.tensor(batch_size, device=device, dtype=torch.float32)
    #
    # policynet_optimizer.zero_grad()
    # loss.backward()
    # policynet_optimizer.step()


    # Optimize value net for a given number of steps
    # Set value net in training mode
    value_net.train()
    ex_rtg = memory.extrinsic_discounted_rtg(batch_size)      # Use undiscounted reward-to-go to fit the value net
    val_est = []

    print("\n\n\tUpdate Value Net for %d steps" % (num_vn_iter))

    for i in tqdm(range(num_vn_iter)):          # Use tqdm to show progress bar
        for j in range(batch_size):
            val_est_traj = value_net(states[j]).squeeze()
            val_est.append(val_est_traj)
        value_net_mse = value_net.optimize_model(val_est, ex_rtg, valuenet_optimizer)

    # Reset Flags
    if not(render_each_episode):
        finished_rendering_this_epoch = False

    ###################################################################


    # Record epoch stats
    epoch_durations.append(sum(episode_durations) / batch_size)
    epoch_rewards.append(sum(episode_rewards) / batch_size)

    training_info["epoch mean durations"].append(epoch_durations[-1])
    training_info["epoch mean rewards"].append(epoch_rewards[-1])
    training_info["value net loss"].append(value_net_mse)
    if (i_epoch + 1) % num_avg_epoch:
        training_info["past %d epochs mean reward" %  (num_avg_epoch)] = \
            (sum(training_info["epoch mean rewards"][-num_avg_epoch:]) / num_avg_epoch) \
                if len(training_info["epoch mean rewards"]) >= num_avg_epoch else 0

    # Print stats
    print("\n\n=============  Epoch: %d  =============" % (i_epoch + 1))
    print("epoch mean durations: %f" % (epoch_durations[-1]))
    print("epoch mean rewards: %f" % (epoch_rewards[-1]))
    print("Max reward achieved: %f" % training_info["max reward achieved"])
    print("value net loss: %f" % value_net_mse)

    # Plot stats
    if plot:
        plot_durations(training_info["epoch mean rewards"], training_info["value net loss"])

    # Update counter
    i_epoch += 1

    # Every save_ckpt_interval, save a checkpoint according to current i_episode.
    if i_epoch % save_ckpt_interval == 0:
        save_checkpoint(ckpt_dir, policy_net, value_net, valuenet_optimizer, i_epoch,
                        policy_lr=policy_lr, valuenet_lr=valuenet_lr, **training_info)

