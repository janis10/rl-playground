# We use PyTorch to implement the neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import os


def initialize_fc_layer(layer, weight_scale=1.0, bias=0.0):
    """
    Initializes a fully connected layer.

    Arguments:
      layer - torch.nn.<layer> module
      w_scale - float
    Outputs:
      Initialized layer
    """
    # 'nn.init.orthogonal_' fills the 'layer.weight.data' (Tensor) with a (semi) orthogonal matrix,
    # see Exact solutions to the nonlinear dynamics of learning in deep linear neural networks - Saxe, A. et al. (2013).
    nn.init.orthogonal_(layer.weight.data, gain=weight_scale)
    # layer.weight.data.mul_(weight_scale)
    # 'nn.init.constant_' fills the 'layer.bias.data' (Tensor) with the 'bias' value.
    nn.init.constant_(layer.bias.data, bias)
    return layer


class Actor(nn.Module):
    # Chosen architecture:
    # input -> fc1 -> relu -> fc2 -> relu -> fc3 -> softmax -> output

    # Initialize
    def __init__(
        self, state_size, action_size, checkpoint_path=None, hidden_0=256, hidden_1=128
    ):
        super(Actor, self).__init__()
        # We initialize 3 fully connected layers.
        self.fc1 = initialize_fc_layer(nn.Linear(state_size, hidden_0))
        self.fc2 = initialize_fc_layer(nn.Linear(hidden_0, hidden_1))
        self.fc3 = initialize_fc_layer(nn.Linear(hidden_1, action_size))

        self.checkpoint_path = checkpoint_path

    # Forward propagation
    def forward(self, x, action=None):
        # Input x
        # -> fc1 -> relu
        x = F.relu(self.fc1(x))
        # -> fc2 -> relu
        x = F.relu(self.fc2(x))
        # -> fc3 -> softmax
        probs = F.softmax(self.fc3(x), dim=1)

        # Create Categorical distribution based on the
        # probabilities out of the softmax.
        dist = distributions.Categorical(probs)

        # If no action provided, sample randomly
        # based on the distribution from the nn output.
        if action is None:
            action = dist.sample()

        # Compute the log-probability density/mass function
        # evaluated at the 'action' value.
        log_prob = dist.log_prob(action)

        return action, log_prob, dist.entropy()

    # Load from checkpoint
    def load(self):
        if self.checkpoint_path is not None and os.path.isfile(self.checkpoint_path):
            self.load_state_dict(torch.load(self.checkpoint_path))

    # Save to checkpoint
    def checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)


class Critic(nn.Module):
    # Chosen architecture:
    # input -> fc1 -> relu -> fc2 -> relu -> fc3 -> output

    # Initialize
    def __init__(self, state_size, checkpoint_path=None, hidden_0=256, hidden_1=128):
        super(Critic, self).__init__()

        self.fc1 = initialize_fc_layer(nn.Linear(state_size, hidden_0))
        self.fc2 = initialize_fc_layer(nn.Linear(hidden_0, hidden_1))
        self.fc3 = initialize_fc_layer(nn.Linear(hidden_1, 1))

        self.checkpoint_path = checkpoint_path

    # Forward propagation
    def forward(self, x):
        # Input x
        # -> fc1 -> relu
        x = F.relu(self.fc1(x))
        # -> fc2 -> relu
        x = F.relu(self.fc2(x))
        # -> fc3
        value = self.fc3(x)
        return value

    # Load from checkpoint
    def load(self):
        if self.checkpoint_path is not None and os.path.isfile(self.checkpoint_path):
            self.load_state_dict(torch.load(self.checkpoint_path))

    # Save to checkpoint
    def checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)
