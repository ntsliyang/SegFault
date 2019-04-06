import random
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
from memory import Memory


class PolicyNet(nn.Module):
    """
        Simple Multilayer Perceptron for Policy Gradient with Gaussian policies

        There is a neural network for calculating mean actions  miu_theta(s),
            and a separate neural network for calculating the log standard deviations sigma_theta(s).
        These two neural networks should share some common layers as the encoding network.

        layer_sizes should be a dictionary containing the following key-value entries:
        layer_sizes = {
                        "encoding" : (a list representing encoding network layer sizes)
                        "mean" : (a list representing mean network layer sizes)
                        "std" : (a list representing log standard deviation network layer sizes)
                      }
        Note:
            1) the last entry of "encoding" and the first entry of "mean" and that of "std" should
               be the same.
            2) the last entry of "mean" and that of "std" should be the same, representing the number
               of dimension of the action space.

    """

    def __init__(self, layer_sizes, action_lim):
        super(PolicyNet, self).__init__()

        encoding_layer_sizes = layer_sizes['encoding']
        mean_layer_sizes = layer_sizes['mean']
        std_layer_sizes = layer_sizes['std']

        # Record the action space dimension
        self.k = mean_layer_sizes[-1]

        # Action value upper and lower bound
        self.action_lim = action_lim

        # Store the network layers in a ModuleList
        self.encoding_layers = nn.ModuleList()
        self.mean_layers = nn.ModuleList()
        self.std_layers = nn.ModuleList()

        # Construct and register layers
        input_size = encoding_layer_sizes[0]
        for output_size in encoding_layer_sizes[1:]:
            self.encoding_layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

        input_size = mean_layer_sizes[0]
        for output_size in mean_layer_sizes[1:]:
            self.mean_layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

        input_size = mean_layer_sizes[0]
        for output_size in std_layer_sizes[1:]:
            self.std_layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

        self.Elu = nn.ELU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward propagation
        x_encoding = x
        for encoding_layer in self.encoding_layers:
            x_encoding = self.Elu(encoding_layer(x_encoding))

        x_mean = x_encoding
        for mean_layer in self.mean_layers[:-1]:
            x_mean = self.Elu(mean_layer(x_mean))
        x_mean = self.tanh(self.mean_layers[-1](x_mean)) * self.action_lim

        x_log_std = x_encoding
        for std_layer in self.std_layers[:-1]:
            x_log_std = self.Elu(std_layer(x_log_std))
        x_log_std = self.std_layers[-1](x_log_std)

        # Take the exponential of x_log_std to calculate the standard deviations
        x_std = torch.exp(x_log_std)

        # If the model is in evaluation mode, use values of mean action as the optimal action
        # Else, return an action sampled from N(x_mean, x_std) and the log of its probability
        if not self.training:
            return x_mean.squeeze(0)
        else:
            # Instantiate a Multivariate Normal distribution that can be used to sample action
            #   and compute the action log-probabilities

            # Covariance matrix is a diagonal matrix with x_std as the diagonal entries
            # m = MultivariateNormal(x_mean, covariance_matrix=torch.diag(x_std).unsqueeze(0))

            m = []
            action = None
            log_prob = None
            for i in range(self.k):
                m.append(Normal(x_mean[0][i], x_std[0][i]))
                if action is None:
                    action = m[-1].sample().unsqueeze(0)
                    log_prob = m[-1].log_prob(action)
                else:
                    action = torch.cat([action, m[-1].sample().unsqueeze(0)])
                    log_prob += m[-1].log_prob(action[-1])

            return action, log_prob

    def optimize_model(self, memory, batch_size, optimizer, device='cpu'):
        """
             Optimize the model for one step
        :param memory:
        :param batch_size:
        :param optimizer:
        :return:
        """

        act_log_prob = memory.act_log_prob(batch_size)

        in_gae = memory.intrinsic_gae(batch_size)
        ex_gae = memory.extrinsic_gae(batch_size)

        # Concatenate act_log_prob and ex_gae. Note that we are missing one value at each episode, so we compensate by
        #   inserting a value 0.
        act_log_prob_list = []
        ex_gae_list = []
        for i in range(batch_size):
            act_log_prob_com = torch.cat([torch.tensor([0.], device=device), act_log_prob[i]], dim=0)
            ex_gae_com = torch.cat([torch.tensor([0.], device=device), ex_gae[i]], dim=0)
            act_log_prob_list.append(act_log_prob_com)
            ex_gae_list.append(ex_gae_com)
        alp_cat = torch.cat(act_log_prob_list, dim=0)
        ex_gae_cat = torch.cat(ex_gae_list, dim=0)
        # Remove the leading 0
        alp_cat = alp_cat[1:]
        ex_gae_cat = ex_gae_cat[1:]

        # Calculate policy gradient. Intrinsic GAE and extrinsic GAE are added together
        loss = - torch.sum(alp_cat * (ex_gae_cat + in_gae)) / torch.tensor(batch_size, device=device)


        # # Using reward-to-go to compute policy gradients
        # act_log_prob = memory.act_log_prob(batch_size)
        # ex_rtg = memory.extrinsic_rtg(batch_size)
        #
        # loss = 0
        # for i in range(batch_size):
        #     loss += - torch.sum(act_log_prob[i] * ex_rtg[i])
        # loss /= torch.tensor(batch_size, device=device)

        optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


class ValueNet(nn.Module):
    """
        Value network to estimate state value.
    """
    def __init__(self, input_size):
        super(ValueNet, self).__init__()

        self.FC1 = nn.Linear(input_size, 64)
        #self.FC2 = nn.Linear(32, 64)
        self.FC3 = nn.Linear(64, 64)
        self.FC4 = nn.Linear(64, 1)

        self.Elu = nn.ELU()

    def forward(self, x):
        x = self.Elu(self.FC1(x))
        #x = self.Elu(self.FC2(x))
        x = self.Elu(self.FC3(x))
        x = self.FC4(x)

        return x

    def optimize_model(self, value_estimate, reward_to_go, optimizer):
        """
            Optimize the model for one step
        :param value_estimate:  a list of value estimates for a batch of trajectories
        :param reward_to_go:    a list of reward to go for a batch of trajectories
        :param optimizer:
        :return: loss value
        """
        # Assume that the value estimate of the last state is included. This value estimate will be discarded.
        num = 0
        mse = None
        for val_est, rtg in zip(value_estimate, reward_to_go):
            if mse is None:
                mse = torch.sum((val_est[:-1] - rtg) ** 2)
            else:
                mse += torch.sum((val_est[:-1] - rtg) ** 2)
            num += val_est[:-1].shape[0]
        mse /= num

        optimizer.zero_grad()
        mse.backward(retain_graph=True)
        optimizer.step()

        return mse.item()
