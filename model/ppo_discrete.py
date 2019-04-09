import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, layer_sizes):
        super(PolicyNet, self).__init__()

        # Store the network layers in a ModuleList
        self.layers = nn.ModuleList()
        input_size = layer_sizes[0]
        for output_size in layer_sizes[1:]:
            self.layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

        self.Elu = nn.ELU()

    def forward(self, x, action_query=None):
        """
            Forward propogation
        :param x: Input
        :param action_query: (Optional) Specify an action to query if want the log probability of this specific action
        :return:
        """
        # Forward propagation
        for layer in self.layers[:-1]:
            x = self.Elu(layer(x))

        # Compute the logits that will be later used to compute action probability
        logits = self.layers[-1](x)

        # If the model is in evaluation mode, deterministically select the best action
        # Else, return an action sampled and the log of its probability
        if not self.training:
            return logits.argmax()
        else:
            # Instantiate a Categorical (multinomial) distribution that can be used to sample action
            #   or compute the action log-probabilities
            m = Categorical(logits=logits)

            # If action_query is None, then we are free to sample a new action from the distribution and calculate the
            #   the corresponding log probability
            if action_query is None:
                action = m.sample()
                log_prob = m.log_prob(action)
                return action, log_prob

            # Otherwise, we calculate the log probabilities of the specified actions as if they were sampled from this
            #   distribution, and return these log probabilities only.
            else:
                # Sample a sample action to check the correctness of size
                sample_action = m.sample()
                assert action_query.shape == sample_action.shape, \
                    "action_query has the wrong shape. It now has shape {} but should have shape {}"\
                        .format(tuple(action_query.shape), tuple(sample_action.shape))

                # Calculate the log probabilities of the specified action and return
                log_prob = m.log_prob(action_query)
                return log_prob

class ValueNet(nn.Module):
    """
        Value network to estimate state value.
    """
    def __init__(self, input_size):
        super(ValueNet, self).__init__()

        self.FC1 = nn.Linear(input_size, 128)
        self.FC2 = nn.Linear(128, 128)
        self.FC3 = nn.Linear(128, 128)
        self.FC4 = nn.Linear(128, 1)

        # self.BN1 = nn.BatchNorm1d(128)
        # self.BN2 = nn.BatchNorm1d(128)
        # self.BN3 = nn.BatchNorm1d(128)

        self.Elu = nn.ELU()

    def forward(self, x):
        # x = self.Elu(self.BN1(self.FC1(x)))
        # x = self.Elu(self.BN2(self.FC2(x)))
        # x = self.Elu(self.BN3(self.FC3(x)))

        x = self.Elu(self.FC1(x))
        x = self.Elu(self.FC2(x))
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
                mse = F.mse_loss(val_est[:-1], rtg, reduction='sum')
            else:
                mse += F.mse_loss(val_est[:-1], rtg, reduction='sum')
            num += val_est[:-1].shape[0]
        mse /= num

        # val_est = torch.stack(value_estimate, dim=0)
        # rtg = torch.stack(reward_to_go, dim=0)
        # mse = F.mse_loss(val_est[:, :-1], rtg, reduction='mean')

        optimizer.zero_grad()
        mse.backward(retain_graph=True)
        optimizer.step()

        return mse.item()
