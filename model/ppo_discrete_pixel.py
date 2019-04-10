import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
        Input is 4 consecutive frames stacked together.

        Encoder Architecture:
            - Input: size = (N, 4, 52, 52) if grayscale is True,
                     size = (N, 12, 52, 52) if grayscale is False (each frame has 3 RGB channels)
            - Conv1: kernel size = 5, stride = 2, dilation = 1, num_kernel = 64 -> Output size = (N, 64, 24, 24)
            - Conv2: kernel size = 5, stride = 2, dilation = 1, num_kernel = 64 -> Output size = (N, 64, 10, 10)
            - Conv3: kernel size = 3, stride = 2, dilation = 1, num_kernel = 64 -> Output size = (N, 64, 4, 4)
            - Reshape: Input size = (N, 64, 4, 4) -> Output size = (N, 1024)

        Actor Architecture:
            - Input: (N, 1024)
            - ActorLinear layers specified by actor_layer_sizes
            - Output: (N, num_actions)

        Critic Architecture:
            - Input: (N, 1024)
            - CriticLinear layers specified by critic_layer_sizes
            - Output: (N, 1)
    """

    def __init__(self, actor_layer_sizes, critic_layer_sizes, grayscale=True):
        super(ActorCritic, self).__init__()

        in_channels = 4 if grayscale else 12
        self.grayscale = grayscale

        self.conv1 = Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=2)
        self.conv2 = Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.conv3 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)

        # Store the network layers in a ModuleList
        self.actor_layers = nn.ModuleList()
        input_size = 1024
        for output_size in actor_layer_sizes:
            self.actor_layers.append(Linear(input_size, output_size))
            input_size = output_size

        self.critic_layers = nn.ModuleList()
        input_size = 1024
        for output_size in critic_layer_sizes:
            self.critic_layers.append(Linear(input_size, output_size))
            input_size = output_size

    def _encoding(self, x):
        # Check input size
        assert (self.grayscale and x.shape[1:] == (4, 52, 52)) or (not self.grayscale and x.shape[1:] == (12, 52, 52)), \
            "Input x must have shape (N, 4, 52, 52)" if self.grayscale else "Input x must have shape (N, 12, 52, 52)"

        # Forward propogate
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))

        # Reshape result
        x = x.view(-1, 1024)

        return x

    def _actor(self, x, action_query=None):

        # Forward propagation
        for layer in self.actor_layers[:-1]:
            x = F.elu(layer(x))

        # Compute the logits that will be later used to compute action probability
        logits = self.actor_layers[-1](x)

        # Instantiate a Categorical distribution
        m = Categorical(logits=logits)

        # If action_query is None, then we are free to sample a new action from the distribution and calculate the
        #   the corresponding log probability
        if action_query is None:
            action = m.sample()
            log_prob = m.log_prob(action)

        # Otherwise, we calculate the log probabilities of the specified actions as if they were sampled from this
        #   distribution, and return these log probabilities only.
        else:
            # Sample a sample action to check the correctness of size
            action = action_query
            sample_action = m.sample()
            assert action.shape == sample_action.shape, \
                "action_query has the wrong shape. It now has shape {} but should have shape {}" \
                    .format(tuple(action.shape), tuple(sample_action.shape))

            # Calculate the log probabilities of the specified action and return
            log_prob = m.log_prob(action)

        return action, log_prob

    def _critic(self, x):

        # Forward propogation
        for layer in self.critic_layers[:-1]:
            x = F.elu(layer(x))

        # Compute the value
        v = self.critic_layers[-1](x)

        return v

    def forward(self, x, action_query=None):
        # A single pass through both actor and critic

        # Obtain encoding vector
        x = self._encoding(x)

        # Pass through actor network
        action, log_prob = self._actor(x, action_query)

        # Pass through critic network
        value = self._critic(x)

        return action, log_prob, value


