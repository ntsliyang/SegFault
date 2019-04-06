import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gym
import torch
import torch.optim as optim
from itertools import count
import os
from model import PolicyNet, ValueNet
from save_and_load import save_checkpoint, load_checkpoint
from utils import plot_durations
from memory import Memory


#######################  Parameters  ##############################

# Environment parameter
env_name = "LunarLanderContinuous-v2"
is_unwrapped = False

# Model hyperparameters
input_size = 8      # Size of state
output_size = 2     # Number of discrete actions
layer_sizes = {
                "encoding": [input_size, 32, 64],
                "mean": [64, 32, output_size],
                "std": [64, 32, output_size]
              }        # The MLP network architecture
action_lim = 1.

ckpt_dir = "simplePG_Adam_%s_obs_checkpoints/" % (env_name)
save_ckpt_interval = 10

# Memory parameter
capacity = 33      # How many trajectories to store

# Training parameters
# num_episodes = 1000
i_epoch = 0      # This would determine which checkpoint to load, if the checkpoint exists
batch_size = 32
policy_lr = 0.0003
valuenet_lr = 0.001

num_vn_iter = 10    # Number of iterations to train value net per epoch

GAMMA = 0.98
LAMBDA = 0.96
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# Rendering and recording options
render = True
plot = True

render_each_episode = False     # Whether to render each episode
                                #   If set to true, then each episode the agent ever endure will be rendered
                                #   Otherwise, only each episode at the start of each epoch will be rendered
                                #   Note: each epoch has exactly 1 model update and batch_size episodes

# record_each_episode_stats = False   # Whether to record the statistics of each episode
                                    #   If set to true, then each episode the agent ever endure will be recorded
                                    #   Otherwise, only each episode at the start of each epoch will be recorded

num_avg_epoch = 5       # The number of epochs to take for calculating average stats

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
policy_net = PolicyNet(layer_sizes, action_lim).to(device)      # Policy network
value_net = ValueNet(input_size).to(device)                     # Value network

# Set up memory
memory = Memory(capacity, device)

# Set up optimizer
policynet_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
valuenet_optimizer = optim.Adam(value_net.parameters(), lr=valuenet_lr)

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

    finished_rendering_this_epoch = False

    # Every save_ckpt_interval, Check if there is any checkpoint.
    # If there is, load checkpoint and continue training
    # Need to specify the i_episode of the checkpoint intended to load
    if i_epoch % save_ckpt_interval == 0 and os.path.isfile(os.path.join(ckpt_dir, "ckpt_eps%d.pt" % i_epoch)):
        policy_net, value_net, policynet_optimizer, valuenet_optimizer, training_info = \
            load_checkpoint(ckpt_dir, i_epoch, layer_sizes, input_size, action_lim, device=device)

    # To record episode stats
    episode_durations = []
    episode_rewards = []

    for i_episode in range(batch_size):

        # Keep track of the running reward
        running_reward = 0

        # Initialize the environment and state
        current_state = env.reset()

        # Estimate the value of the initial state
        ex_val = value_net(torch.tensor([current_state], device=device)).squeeze()      # squeeze the dimension

        # Store the first state and value estimate in memory
        memory.set_initial_state(current_state, initial_ex_val_est=ex_val)

        for t in count():
            # Make sure that policy net is in training mode
            policy_net.train()

            # Sample an action given the current state
            action, log_prob = policy_net(torch.tensor([current_state], device=device))
            log_prob = log_prob.squeeze()

            # Interact with the environment
            next_state, reward, done, _ = env.step(action.to('cpu').numpy())
            running_reward += reward

            # Estimate the value of the next state
            value = value_net(torch.tensor([next_state], device=device)).squeeze()     # squeeze the dimension

            # Render this episode
            if render and (render_each_episode or (not finished_rendering_this_epoch)):
                env.render()

            # Record transition in memory
            memory.add_transition(action, log_prob, next_state,
                                  extrinsic_reward=reward, extrinsic_value_estimate=value)

            if done:
                # Load and print episode stats after each episode ends
                episode_durations.append(t + 1)
                episode_rewards.append(sum(memory.trajectory_extrinsic_return(1)))
                if running_reward > training_info["max reward achieved"]:
                    training_info["max reward achieved"] = running_reward

                # Check if the problem is solved
                #  CartPole standard: average reward for the past 100 episode above 195
                # if training_info["past 100 episodes mean reward"] > 195:
                #     print("\n\n\t Problem Solved !!!\n\n\n")

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

    # Obtain batch extrinsic value estimates and rewards-to-go and fit value estimate network by regression on MSE
    #   for multiple steps
    val_est = memory.extrinsic_val_est(batch_size)
    ex_rtg = memory.extrinsic_rtg(batch_size)
    for i in range(num_vn_iter):
        value_net_mse = value_net.optimize_model(val_est, ex_rtg, valuenet_optimizer)

    # Reset Flags
    if not(render_each_episode):
        finished_rendering_this_epoch = False

    # Record stats
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
        save_checkpoint(ckpt_dir, policy_net, value_net, policynet_optimizer, valuenet_optimizer, i_epoch,
                        policy_lr=policy_lr, valuenet_lr=valuenet_lr, **training_info)

