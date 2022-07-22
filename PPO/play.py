from pettingzoo.classic import connect_four_v3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# own modules
from helper_functions import choose_action, reshape_image
from networks import Actor


def main():

    # select model to play against
    AI = torch.load(f'models/model_in_use/actor')

    # initialise gym env
    env = connect_four_v3.env()
    env.reset()
    done = False

    # change size of pygame screen
    pygame.display.set_mode((400, 400))

    # select which player user is
    user = f'player_{np.random.randint(0, 2)}'
    print('')
    print(f'You are {user}')
    print('')

    # iterate over agents
    for agent in env.agent_iter():

        # get obs
        observation, _, done, _ = env.last()

        # get player to input action
        if agent == user:
            action = int(input('choose move:'))

        # AI makes move
        else:
            reshaped_observation = reshape_image(observation['observation'])
            action, _ = choose_action(  reshaped_observation, 
                                        observation['action_mask'], 
                                        AI
                                    )

        if done:

            # determine who won
            if agent == user:
                print('Game over : user wins')
            else:
                print('Game over : AI wins')

            # ask user if they want to play again
            play_again = int(input('Press 1 to play again or 0 to close:'))

            if play_again == 0:
                sys.exit()
            else:
                env.reset()
                main()

        env.step(action)

        env.render()


if __name__ == '__main__':
    main()
