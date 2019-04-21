import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, LSTM
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCriticLSTM(nn.Module):
    """
        Actor-Critic model with 2 value heads. one feed-forward FC, one LSTM

        Input is 4 consecutive frames stacked together.

        Encoder Architecture:
            - Input: size = (N, 4, 52, 52) if grayscale is True,
                     size = (N, 12, 52, 52) if grayscale is False (each frame has 3 RGB channels)
            - Conv1: kernel size = 5, stride = 2, dilation = 1, num_kernel = 64 -> Output size = (N, 64, 24, 24)
            - Conv2: kernel size = 5, stride = 2, dilation = 1, num_kernel = 64 -> Output size = (N, 64, 10, 10)
            - Conv3: kernel size = 3, stride = 2, dilation = 1, num_kernel = 64 -> Output size = (N, 64, 4, 4)
            - Reshape: Input size = (N, 64, 4, 4) -> Output size = (N, 1024)
            - FC1: Input size = (N, 1024) -> Output size = (N, 128)

        Actor Architecture:
            - Input: (N, 128)
            - ActorLinear layers specified by actor_layer_sizes
            - Output: (N, num_actions)

        FC Critic Architecture:
            - Input: (N, 128)
            - CriticLinear layers specified by critic_layer_sizes
            - Output: (N, 1)

        LSTM Critic Architecture:
            - cell size = hidden size = (N, 128)
            - Input size: (seq_len, N, 128)
            - Output FC: (seq_len, N, 128) -> (seq_len, N, 1)
    """

    def __init__(self, actor_layer_sizes, critic_1_layer_sizes,
                 critic_2_extra_input=None, use_lstm=True, grayscale=True, device='cpu'):
        super(ActorCriticLSTM, self).__init__()

        self.device = device
        self.use_lstm=use_lstm

        in_channels = 4 if grayscale else 12
        self.grayscale = grayscale

        self.conv1 = Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=2)
        self.conv2 = Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.conv3 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.encoder_FC1 = Linear(1024, 128)

        # Store the network layers in a ModuleList
        # Actor module
        self.actor_layers = nn.ModuleList()
        input_size = 128
        for output_size in actor_layer_sizes:
            self.actor_layers.append(Linear(input_size, output_size))
            input_size = output_size

        # FC Critic module
        self.critic_1_layers = nn.ModuleList()
        input_size = 128
        for output_size in critic_1_layer_sizes:
            self.critic_1_layers.append(Linear(input_size, output_size))
            input_size = output_size

        # LSTM critic module
        # if critic_2_extra_input is not None:
        #     assert type(critic_2_extra_input) is int, "critic_2_extra_input must be an integer."
        #     self.critic_2_lstm = LSTM(input_size=128 + critic_2_extra_input, hidden_size=128)
        # else:
        #     self.critic_2_lstm = LSTM(input_size=128, hidden_size=128)
        # self.critic_2_FC = Linear(128, 1)
        # self.h0 = None
        # self.c0 = None      # Initialize initial hidden and cell state for LSTM critic

        # A backup FC Critic 2 module
        self.critic_2_layers = nn.ModuleList()
        input_size = 128 + critic_2_extra_input if critic_2_extra_input is not None else 128
        for output_size in critic_1_layer_sizes:
            self.critic_2_layers.append(Linear(input_size, output_size))
            input_size = output_size

    def encoding(self, x):
        # Check input size
        assert (self.grayscale and x.shape[1:] == (4, 52, 52)) or (not self.grayscale and x.shape[1:] == (12, 52, 52)), \
            "Input x must have shape (N, 4, 52, 52)" if self.grayscale else "Input x must have shape (N, 12, 52, 52)"

        # Forward propogate
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))

        # Reshape result
        x = x.view(-1, 1024)

        x = self.encoder_FC1(x)     # shape: (N, 128)

        return x

    def actor(self, x, action_query=None):

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

    def critic_1(self, x):

        # Forward propogation
        for layer in self.critic_1_layers[:-1]:
            x = F.elu(layer(x))

        # Compute the value
        v = self.critic_1_layers[-1](x)

        return v

    def critic_2(self, x):
        """
            Input x should be the returned tensor from encoding()
        :param x:
        :return:
        """
        if self.use_lstm:
            assert len(x.shape) == 3, "Input x must have shape (seq_len, N, input_size). x now has {}".format(x.shape)

            batch_size = x.shape[1]

            # If h0 and c0 are None, then initialize
            if self.h0 is None and self.c0 is None:
                self.h0 = torch.zeros((1, batch_size, 128), dtype=torch.float32, device=self.device)
                self.c0 = torch.zeros((1, batch_size, 128), dtype=torch.float32, device=self.device)

            output, (self.h0, self.c0) = self.critic_2_lstm(x, (self.h0, self.c0))

            v = self.critic_2_FC(output)

        else:
            # Forward propogation
            for layer in self.critic_2_layers[:-1]:
                x = F.elu(layer(x))

            # Compute the value
            v = self.critic_2_layers[-1](x)

        return v

    def reset_critic_2(self):
        self.h0, self.c0 = None, None

    def forward(self, x, i_episode=None, action_query=None):
        """

        :param x:
        :param i_episode:   to tell LSTM critic which episode it is in for more accurate intrinsic value prediction
        :param action_query:
        :return:
        """
        # A single pass through both actor and critic

        # Obtain encoding vector
        x = self.encoding(x)

        # Pass through actor network
        action, log_prob = self.actor(x, action_query)

        # Pass through critic network 1
        value_1 = self.critic_1(x)

        # Pass through critic network 2
        if i_episode is not None:
            assert type(i_episode) is int, "i_episode must be an integer."
            x = torch.cat([x, torch.ones((x.shape[0], 1), dtype=torch.float32, device=self.device)],dim=1 )

        value_2 = self.critic_2(x.unsqueeze(dim=1) if self.use_lstm else x)    # Spare the batch dimension.
                                                        # Batch dimension of encoding vector x is the seq_len dimension of LSTM input

        return action, log_prob, value_1, value_2


