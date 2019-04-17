"""
    Baseline model
        - Proximal Policy Optimization with a value net estimating state value and update policy with GAE.
        - Normal, discrete environment.
"""

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor, ToPILImage
from itertools import count
import os
from model.ppo_discrete_pixel_LSTM import ActorCriticLSTM
from Hashing.AEHash import AEHash
from Hashing.SimHash import SimHash
from LPLGraph.LPLGraph import LPLGraph
from utils.utils3 import plot_durations
from utils.memory import Memory
from utils.visualize import visualize_aehash
import json
import sys
import copy
from tqdm import tqdm


# Utils for saving and loading checkpoints
#
# def save_checkpoint(file_dir, actor_critic, ae_hash, ae_hash_optim, i_epoch, **kwargs):
#     save_dict = {"actor_critic": actor_critic.state_dict(),
#                  "ae_hash": ae_hash.state_dict(),
#                  "ae_hash_optim": ae_hash_optim.state_dict(),
#                  "i_epoch": i_epoch
#                  }
#     # Save optional contents
#     save_dict.update(kwargs)
#
#     # Create the directory if not exist
#     if not os.path.isdir(file_dir):
#         os.makedirs(file_dir)
#
#     file_name = os.path.join(file_dir, "ckpt_eps%d.pt" % i_epoch)
#
#     # Delete the file if the file already exist
#     try:
#         os.remove(file_name)
#     except OSError:
#         pass
#
#     # Save the file
#     torch.save(save_dict, file_name)
#
#
# def load_checkpoint(file_dir, i_epoch, actor_layer_sizes, critic_layer_sizes,
#                     len_hashcode, stacked, grayscale=True, device='cuda'):
#     checkpoint = torch.load(os.path.join(file_dir, "ckpt_eps%d.pt" % i_epoch), map_location=device)
#
#     actor_critic = ActorCritic(actor_layer_sizes, critic_layer_sizes, grayscale).to(device)
#     actor_critic.load_state_dict(checkpoint["actor_critic"])
#
#     ae_hash = AEHash(len_hashcode, 4 if stacked else 1, device=device).to(device)
#     ae_hash.load_state_dict(checkpoint["ae_hash"])
#
#     ae_hash_optim = optim.Adam(ae_hash.parameters())
#     ae_hash_optim.load_state_dict(checkpoint["ae_hash_optim"])
#
#     checkpoint.pop("actor_critic")
#     checkpoint.pop("i_epoch")
#     checkpoint.pop("ae_hash")
#     checkpoint.pop("ae_hash_optim")
#
#     return actor_critic, ae_hash, ae_hash_optim, checkpoint


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
critic_1_layer_sizes = config['critic_1_layer_sizes']
grayscale = config['grayscale']

ckpt_dir = config['ckpt_dir']
save_ckpt_interval = config['save_ckpt_interval']

# Hashing parameter
stacked = config['stacked']
len_AE_hashcode = config['len_AE_hashcode']
len_SimHash_hashcode = config['len_SimHash_hashcode']
noise_scale = config['noise_scale']
saturating_weight = config['saturating_weight']
hash_batchsize = config['hash_batchsize']
hash_num_updates_per_epoch = config['hash_num_updates_per_epoch']

# LPLGraph parameters
max_reward = config['max_reward']
num_particles = config['num_particles']
curiosity_weight = config['curiosity_weight']
curiosity_delay = config['curiosity_delay']     # Number of epochs to train autoencoder before applying LPLGraph to induce exploration bonus

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
actor_critic = ActorCriticLSTM(actor_layer_sizes, critic_1_layer_sizes, critic_2_extra_input=1, use_lstm=False, grayscale=grayscale, device=device).to(device)

# Create AE Hashing model and optimizers
ae_hash = AEHash(len_AE_hashcode, 4 if stacked else 1, noise_scale, saturating_weight, device=device).to(device)
ae_hash_optim = optim.Adam(ae_hash.parameters())

# Create SimHash
sim_hash = SimHash(len_AE_hashcode, len_SimHash_hashcode)

# Create LPLGraph
graph = LPLGraph(len_SimHash_hashcode, actor_layer_sizes[-1], max_reward, num_particles=num_particles)

# Set up action counter to infer the dominating action
act_counter = np.zeros((actor_layer_sizes[-1],), dtype=np.int32)

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
                 "extrinsic value net loss": [],
                 "intrinsic value net loss": [],
                 "AE Hash loss": []}

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
    #     actor_critic, ae_hash, ae_hash_optim, training_info = \
    #         load_checkpoint(ckpt_dir, i_epoch, actor_layer_sizes, critic_layer_sizes, len_AE_hashcode, stacked, device=device)
    #     print("\n\t Checkpoint successfully loaded! \n")

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

        # Reset LSTM critic
        actor_critic.reset_critic_2()
        actor_critic.eval()

        # Obtain action, log probability, and value estimate for the initial state
        # Move the outputs to cpu to save memory
        action, log_prob, ex_val, in_val = actor_critic(current_state.unsqueeze(dim=0), i_episode=i_episode)
        action = action.squeeze().cpu()
        log_prob = log_prob.squeeze().cpu()
        ex_val = ex_val.squeeze().cpu()
        in_val = in_val.squeeze().cpu()

        # Store the first state and value estimate in memory
        memory.set_initial_state(current_state.clone().detach().cpu(), initial_ex_val_est=ex_val, initial_in_val_est=in_val)

        # Obtain current state hash code
        if i_epoch > curiosity_delay:
            current_state_hash, _ = ae_hash.hash((current_state if stacked else current_state[-1:]).unsqueeze(dim=0),
                                              base_ten=False)
            current_state_hash = sim_hash.hash(current_state_hash.squeeze(), base_ten=True)   # Downsample

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
            next_action, next_log_prob, ex_val, in_val = actor_critic(next_state.unsqueeze(dim=0), i_episode=i_episode)
            next_action = next_action.squeeze().cpu()
            next_log_prob = next_log_prob.squeeze().cpu()
            ex_val = ex_val.squeeze().cpu()
            in_val = in_val.squeeze().cpu()

            if i_epoch > curiosity_delay:

                # Update action counter
                act_counter[action.item()] += 1

                # Obtain next state hash code
                ae_hash.eval()
                next_state_hash, _ = ae_hash.hash((next_state if stacked else next_state[-1:]).unsqueeze(dim=0),
                                               base_ten=False)
                next_state_hash = sim_hash.hash(next_state_hash.squeeze(), base_ten=True)     # Downsample

                # If next state hashed to a different code than the current state, then infer the dominating action,
                #   update causal link, and clear action counter
                if next_state_hash != current_state_hash:
                    main_action = np.argmax(act_counter)
                    graph.update_transition(current_state_hash, main_action, next_state_hash)
                    act_counter = np.zeros((actor_layer_sizes[-1],), dtype=np.int32)

                in_reward = graph.action_confidence(current_state_hash, action.item())
                in_reward = curiosity_weight * np.sqrt(in_reward)  # Take the square root of confidence value

                # Store transition in memory
                memory.add_transition(action, log_prob.cpu(), next_state.clone().detach().cpu(),
                                      extrinsic_reward=reward, extrinsic_value_estimate=ex_val,
                                      intrinsic_reward=in_reward, intrinsic_value_estimate=in_val)

                current_state_hash = next_state_hash

            else:

                memory.add_transition(action, log_prob.cpu(), next_state.clone().detach().cpu(),
                                      extrinsic_reward=reward, extrinsic_value_estimate=ex_val,
                                      intrinsic_reward=0.0, intrinsic_value_estimate=in_val)

            current_state = next_state
            action = next_action
            log_prob = next_log_prob


            # Visualizing AE Hash
            # ae_hash.eval()      # Set in evaluation mode
            # if stacked:
            #     code, latent = ae_hash.hash(next_state.unsqueeze(dim=0), base_ten=False)
            #     recon_state, _ = ae_hash(next_state.unsqueeze(dim=0))
            # else:
            #     code, latent = ae_hash.hash(next_state[-1:].unsqueeze(dim=0), base_ten=False)
            #     recon_state, _ = ae_hash(next_state[-1:].unsqueeze(dim=0))
            #
            # sim_code = sim_hash.hash(code.squeeze())

            # visualize_aehash(next_state.cpu().numpy(), recon_state.squeeze(dim=0).cpu().detach().numpy(), code.squeeze(), latent.squeeze().cpu().detach().numpy())


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
    actor_critic.reset_critic_2()
    actor_critic_candidate = copy.deepcopy(actor_critic).to(device)
    actor_critic_candidate.train()

    # initialize the optimizer
    candidate_optimizer = optim.Adam(actor_critic_candidate.parameters())

    # Get batch data
    ex_rtg = memory.extrinsic_discounted_rtg(batch_size)
    ex_gae = memory.extrinsic_gae(batch_size)
    in_rtg = memory.intrinsic_rtg(batch_size)
    in_gae = memory.intrinsic_gae(batch_size)
    old_act_log_prob = memory.act_log_prob(batch_size)
    states = memory.states(batch_size)
    actions = memory.actions(batch_size)

    # Proximal Policy Optimization - Calculate joint loss for both actor and critic network
    loss = 0
    ex_critic_loss_total = 0
    in_critic_loss_total = 0

    print("\n\n\tUpdate Actor Critic for %d steps:" % num_updates_per_epoch)

    for i in tqdm(range(num_updates_per_epoch)):        # Use tqdm to show progress bar

        actor_critic_candidate.reset_critic_2()

        num = 0
        for j in range(batch_size):
            # Move the slices to specified device
            ex_rtg[j] = ex_rtg[j].detach().to(device)
            ex_gae[j] = ex_gae[j].detach().to(device)
            in_rtg[j] = in_rtg[j].detach().to(device)
            in_gae[j] = in_gae[j].detach().to(device)
            old_act_log_prob[j] = old_act_log_prob[j].detach().to(device)
            states[j] = states[j].detach().to(device)
            actions[j] = actions[j].detach().to(device)

            # Calculate the log probabilities of the actions stored in memory from the distribution parameterized by the
            #   new candidate network
            _, new_act_log_prob, ex_val_est, in_val_est = actor_critic_candidate(states[j][:-1], i_episode=j, action_query=actions[j])    # Ignore last state

            # Actor part
            ratio = torch.exp(new_act_log_prob - old_act_log_prob[j])      # Detach old action log prob

            gae = ex_gae[j] + in_gae[j]

            surr1 = ratio * gae
            surr2 = (((gae < 0.).type(torch.float32) * (1 - clip_range) +
                      (gae > 0.).type(torch.float32) * (1 + clip_range))) * gae
            actor_loss = - torch.sum(torch.min(surr1, surr2))

            # Extrinsic (FC) Critic part
            ex_critic_loss = F.mse_loss(ex_val_est.squeeze(), ex_rtg[j], reduction='sum')
            ex_critic_loss_total += ex_critic_loss

            if i_epoch > curiosity_delay:
                # Intrinsic (LSTM) Critic part
                in_critic_loss = F.mse_loss(in_val_est.squeeze(), in_rtg[j], reduction='sum')
                in_critic_loss_total += in_critic_loss

                loss += actor_loss + vf_coef * (ex_critic_loss + in_critic_loss)    # joint loss
            else:
                loss += actor_loss + vf_coef * ex_critic_loss       # joint loss

            num += ratio.shape[0]

        loss /= torch.tensor(num, device=device, dtype=torch.float32)
        ex_critic_loss_total /= torch.tensor(num, device=device, dtype=torch.float32)
        in_critic_loss_total /= torch.tensor(num, device=device, dtype=torch.float32)

        candidate_optimizer.zero_grad()
        loss.backward(retain_graph=True if i < num_updates_per_epoch - 1 else False)    # Free buffer the last time doing backprop

        # Clip the gradients in the actor network
        for layer in actor_critic_candidate.actor_layers:
            for param in layer.parameters():
                param.grad.data.clamp_(-1., 1.)

        candidate_optimizer.step()

    # Restore the updated parameters
    actor_critic_candidate.reset_critic_2()
    actor_critic = copy.deepcopy(actor_critic_candidate).to(device)


    # Update Autoencoder hashing model parameters for a specified number of steps

    print("\n\n\tUpdate Autoencoder Hashing model for %d steps:" % hash_num_updates_per_epoch)

    ae_hash.train()     # Set in training mode

    ae_hash_loss = 0
    for i in tqdm(range(hash_num_updates_per_epoch)):
        # Sample a batch of states
        states_sampled = memory.sample_states(hash_batchsize).clone().to(device)

        if stacked:
            ae_hash_loss = ae_hash.optimize_model(states_sampled, ae_hash_optim)
        else:
            ae_hash_loss = ae_hash.optimize_model(states_sampled[:, -1:, :, :], ae_hash_optim)  # If not stacked, take the last channel

    # Reset Flags
    if not(render_each_episode):
        finished_rendering_this_epoch = False

    ###################################################################

    # Record epoch stats
    epoch_durations.append(sum(episode_durations) / batch_size)
    epoch_rewards.append(sum(episode_rewards) / batch_size)

    training_info["epoch mean durations"].append(epoch_durations[-1])
    training_info["epoch mean rewards"].append(epoch_rewards[-1])
    training_info["extrinsic value net loss"].append(ex_critic_loss_total)
    training_info["intrinsic value net loss"].append(in_critic_loss_total)
    training_info["AE Hash loss"].append(ae_hash_loss)
    if (i_epoch + 1) % num_avg_epoch:
        training_info["past %d epochs mean reward" %  (num_avg_epoch)] = \
            (sum(training_info["epoch mean rewards"][-num_avg_epoch:]) / num_avg_epoch) \
                if len(training_info["epoch mean rewards"]) >= num_avg_epoch else 0

    # Print stats
    print("\n\n=============  Epoch: %d  =============" % (i_epoch + 1))
    print("epoch mean durations: %f" % (epoch_durations[-1]))
    print("epoch mean rewards: %f" % (epoch_rewards[-1]))
    print("Max reward achieved: %f" % training_info["max reward achieved"])
    print("extrinsic value net loss: %f" % ex_critic_loss_total)
    print("intrinsic value net loss: %f" % in_critic_loss_total)
    print("Autoencoder Hashing model loss: %f" % ae_hash_loss)

    # Plot stats
    if plot:
        # plot_durations(training_info["epoch mean rewards"], training_info["value net loss"])
        plot_durations(training_info["epoch mean rewards"], training_info["extrinsic value net loss"],
                       training_info["intrinsic value net loss"], training_info["AE Hash loss"])

    # Update counter
    i_epoch += 1

    # Every save_ckpt_interval, save a checkpoint according to current i_episode.
    # if i_epoch % save_ckpt_interval == 0:
    #     save_checkpoint(ckpt_dir, actor_critic, ae_hash, ae_hash_optim, i_epoch, **training_info)
