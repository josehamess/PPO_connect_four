import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# network for taking action
class Actor(nn.Module):
    def __init__(self, n_actions, drop_out):
        super(Actor, self).__init__()
        self.n_actions = n_actions
        self.drop_out = drop_out

        self.actor = nn.Sequential(
                nn.Conv2d(1, 24, 3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(24, 64, 3),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, 3),
                nn.LeakyReLU(),
                nn.Conv2d(128, 256, 4),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.LeakyReLU(),
                nn.Linear(256, self.n_actions),
                nn.Softmax(dim=-1)
                )
    
    def forward(self, state, action_mask):

        # multiply action dist by action mask to remove probs
        # for actions that are not available
        dist = self.actor(state.float()) * action_mask

        # renormalise probabilities with removed actions
        # add small value to avoid div by zero error
        dist = dist / torch.sum(dist) + 1e-7

        # create torch distribution
        dist = Categorical(dist)
        
        return dist


# network for estimating value of states
class Critic(nn.Module):
    def __init__(self, drop_out):
        super(Critic, self).__init__()
        self.drop_out = drop_out

        self.actor = nn.Sequential(
                nn.Conv2d(1, 24, 3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(24, 64, 3),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, 3),
                nn.LeakyReLU(),
                nn.Conv2d(128, 256, 4),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.LeakyReLU(),
                nn.Linear(256, 1)
                )
    
    def forward(self, state):
        value = self.actor(state.float())
        return value
