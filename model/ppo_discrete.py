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

    def forward(self, x):
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

            action = m.sample()
            log_prob = m.log_prob(action)
            return action, log_prob

class ValueNet(nn.Module):
    """
        Value network to estimate state value.
    """
    def __init__(self, input_size):
        super(ValueNet, self).__init__()

        self.FC1 = nn.Linear(input_size, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, 1)

        self.Elu = nn.ELU()

    def forward(self, x):
        x = self.Elu(self.FC1(x))
        x = self.Elu(self.FC2(x))
        x = self.FC3(x)

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
