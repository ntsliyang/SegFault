"""
    Baseline model
        - Proximal Policy Optimization with a value net estimating state value and update policy with GAE.
        - Normal, discrete environment.
"""

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor, ToPILImage
from itertools import count
import os
from model.ppo_discrete_pixel import ActorCritic
from utils.utils import plot_durations
from utils.memory import Memory
import json
import sys
import copy
from tqdm import tqdm


# Utils for saving and loading checkpoints

def save_checkpoint(file_dir, actor_critic, i_epoch, **kwargs):
    save_dict = {"actor_critic": actor_critic.state_dict(),
                 "i_epoch": i_epoch
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


def load_checkpoint(file_dir, i_epoch, actor_layer_sizes, critic_layer_sizes, grayscale=True, device='cuda'):
    checkpoint = torch.load(os.path.join(file_dir, "ckpt_eps%d.pt" % i_epoch), map_location=device)

    actor_critic = ActorCritic(actor_layer_sizes, critic_layer_sizes, grayscale).to(device)
    actor_critic.load_state_dict(checkpoint["actor_critic"])

    checkpoint.pop("actor_critic")
    checkpoint.pop("i_epoch")

    return actor_critic, checkpoint


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
actor_layer_sizes = config['actor_layer_sizes']         # The MLP network architecture
critic_layer_sizes = config['critic_layer_sizes']
grayscale = config['grayscale']

ckpt_dir = config['ckpt_dir']
save_ckpt_interval = config['save_ckpt_interval']

# Memory parameter
capacity = config['capacity']     # How many trajectories to store

# Training parameters
# num_episodes = 1000
i_epoch = config['i_epoch']      # This would determine which checkpoint to load, if the checkpoint exists
batch_size = config['batch_size']

num_updates_per_epoch = config['num_updates_per_epoch']
clip_range = config['clip_range']
vf_coef = config['vf_coef']

GAMMA = config['GAMMA']
LAMBDA = config['LAMBDA']

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
actor_critic = ActorCritic(actor_layer_sizes, critic_layer_sizes, grayscale=grayscale).to(device)

# Set up memory
memory = Memory(capacity, GAMMA, LAMBDA, 'cpu')     # Put memory on cpu to save space

# Set up pixel observation preprocessing
transform = Compose([
    ToPILImage(),
    Grayscale(num_output_channels=1),   # Turn frame into grayscale
    Resize((52, 52)),
    ToTensor()
])

###################################################################
# Start training

# Dictionary for extra training information to save to checkpoints
training_info = {"epoch mean durations" : [],
                 "epoch mean rewards" : [],
                 "max reward achieved": 0,
                 "past %d epochs mean reward" % num_avg_epoch: 0,
                 "value net loss": []}

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
        actor_critic, training_info = \
            load_checkpoint(ckpt_dir, i_epoch, actor_layer_sizes, critic_layer_sizes, device=device)

    # To record episode stats
    episode_durations = []
    episode_rewards = []

    ###################################################################
    # Collect trajectories

    # Set model to evaluation mode to collect trajectories
    actor_critic.eval()

    print("\n\n\tCollecting %d episodes: " % (batch_size))

    for i_episode in tqdm(range(batch_size)):       # Use tqdm to show progress bar

        # Keep track of the running reward
        running_reward = 0

        # Initialize the environment and state
        frame_list = []
        initial_frame = env.reset()
        frame_list.append(transform(initial_frame))

        # Obtain three more frames to form the first observation
        for i in range(3):
            next_frame, _, _, _ = env.step(0)    # Take an arbitrary action
            frame_list.append(transform(next_frame))

        current_state = torch.cat(frame_list, dim=0).to(device)     # Stack the images. Note that image shape is (N, C, H, W)

        # Obtain action, log probability, and value estimate for the initial state
        # Move the outputs to cpu to save memory
        action, log_prob, ex_val = actor_critic(current_state.unsqueeze(dim=0))
        action = action.squeeze().cpu()
        log_prob = log_prob.squeeze().cpu()
        ex_val = ex_val.squeeze().cpu()

        # Store the first state and value estimate in memory
        memory.set_initial_state(current_state, initial_ex_val_est=ex_val)

        for t in count():

            # Interact with the environment
            next_frame, reward, done, _ = env.step(action.item())
            running_reward += reward

            # Pop the frame from the top of the list and append the new frame, and stack to form the current state
            frame_list.pop(0)
            frame_list.append(transform(next_frame))
            next_state = torch.cat(frame_list, dim=0).to(device)     # Stack the images

            # Obtain action, log probability and value estimate for the next state in a single propagation
            # Move the outputs to cpu to save memory
            next_action, next_log_prob, value = actor_critic(next_state.unsqueeze(dim=0))
            next_action = next_action.squeeze().cpu()
            next_log_prob = next_log_prob.squeeze().cpu()
            value = value.squeeze().cpu()

            # Record transition in memory
            memory.add_transition(action, log_prob, next_state.cpu(), extrinsic_reward=reward, extrinsic_value_estimate=value)

            # Update current state and action
            action = next_action
            log_prob = next_log_prob

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
    # Optimize the model for a given number of steps

    # Make a candidate to update parameters
    actor_critic_candidate = copy.deepcopy(actor_critic).to(device)
    actor_critic_candidate.train()

    # initialize the optimizer
    candidate_optimizer = optim.Adam(actor_critic_candidate.parameters())

    # Get batch data
    ex_rtg = memory.extrinsic_discounted_rtg(batch_size)
    ex_gae = memory.extrinsic_gae(batch_size)
    old_act_log_prob = memory.act_log_prob(batch_size)
    states = memory.states(batch_size)
    actions = memory.actions(batch_size)

    # Proximal Policy Optimization - Calculate joint loss for both actor and critic network
    loss = 0
    critic_loss_total = 0

    print("\n\n\tUpdate Actor Critic for %d steps:" % num_updates_per_epoch)

    for i in tqdm(range(num_updates_per_epoch)):        # Use tqdm to show progress bar

        num = 0
        for j in range(batch_size):
            # Move the slices to specified device
            ex_rtg[j] = ex_rtg[j].to(device)
            ex_gae[j] = ex_gae[j].to(device)
            old_act_log_prob[j] = old_act_log_prob[j].to(device)
            states[j] = states[j].to(device)
            actions[j] = actions[j].to(device)

            # Calculate the log probabilities of the actions stored in memory from the distribution parameterized by the
            #   new candidate network
            _, new_act_log_prob, val_est = actor_critic_candidate(states[j][:-1], action_query=actions[j])    # Ignore last state

            # Actor part
            ratio = torch.exp(new_act_log_prob - old_act_log_prob[j].detach())      # Detach old action log prob

            surr1 = ratio * ex_gae[j]
            surr2 = (((ex_gae[j] < 0.).type(torch.float32) * (1 - clip_range) +
                      (ex_gae[j] > 0.).type(torch.float32) * (1 + clip_range))) * ex_gae[j]
            actor_loss = - torch.sum(torch.min(surr1, surr2))

            # Critic part
            critic_loss = F.mse_loss(val_est.squeeze(), ex_rtg[j], reduction='sum')
            critic_loss_total += critic_loss        # Record critic loss to print

            # Joint loss
            loss += actor_loss + vf_coef * critic_loss

            num += ratio.shape[0]

        loss /= torch.tensor(num, device=device, dtype=torch.float32)
        critic_loss_total /= torch.tensor(num, device=device, dtype=torch.float32)

        candidate_optimizer.zero_grad()
        loss.backward(retain_graph=True if i < num_updates_per_epoch - 1 else False)    # Free buffer the last time doing backprop

        # Clip the gradients in the actor network
        for layer in actor_critic_candidate.actor_layers:
            for param in layer.parameters():
                param.grad.data.clamp_(-1., 1.)

        candidate_optimizer.step()

    # Restore the updated parameters
    actor_critic_candidate.to('cpu')
    actor_critic = copy.deepcopy(actor_critic_candidate).to(device)

    # Reset Flags
    if not(render_each_episode):
        finished_rendering_this_epoch = False

    ###################################################################

    # Record epoch stats
    epoch_durations.append(sum(episode_durations) / batch_size)
    epoch_rewards.append(sum(episode_rewards) / batch_size)

    training_info["epoch mean durations"].append(epoch_durations[-1])
    training_info["epoch mean rewards"].append(epoch_rewards[-1])
    training_info["value net loss"].append(critic_loss_total)
    if (i_epoch + 1) % num_avg_epoch:
        training_info["past %d epochs mean reward" %  (num_avg_epoch)] = \
            (sum(training_info["epoch mean rewards"][-num_avg_epoch:]) / num_avg_epoch) \
                if len(training_info["epoch mean rewards"]) >= num_avg_epoch else 0

    # Print stats
    print("\n\n=============  Epoch: %d  =============" % (i_epoch + 1))
    print("epoch mean durations: %f" % (epoch_durations[-1]))
    print("epoch mean rewards: %f" % (epoch_rewards[-1]))
    print("Max reward achieved: %f" % training_info["max reward achieved"])
    print("value net loss: %f" % critic_loss_total)

    # Plot stats
    if plot:
        # plot_durations(training_info["epoch mean rewards"], training_info["value net loss"])
        plot_durations(training_info["epoch mean rewards"], training_info["value net loss"])

    # Update counter
    i_epoch += 1

    # Every save_ckpt_interval, save a checkpoint according to current i_episode.
    if i_epoch % save_ckpt_interval == 0:
        save_checkpoint(ckpt_dir, actor_critic, i_epoch, **training_info)
