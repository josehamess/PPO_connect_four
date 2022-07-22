import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# helper functions for use in main file


def clean_memory():

    ### returns a clean memory dictionary ###

    # initialise agent memory dict
    agent_memory = {}

    # fill memory dict with memory lists
    agent_memory['probs'] = []
    agent_memory['values'] = []
    agent_memory['obs'] = []
    agent_memory['rewards'] = []
    agent_memory['actions'] = []
    agent_memory['action_masks'] = []
    agent_memory['dones'] = []

    return agent_memory


def choose_action(observation, action_mask, actor):
    
    ### returns the log probability of taking the action as well as the acion ###

    # get action distribution from actor
    dist = actor(torch.tensor(observation), torch.tensor(action_mask))

    # sample an action from distribution
    action = dist.sample()

    # get the log prob of taking that action
    prob = torch.squeeze(dist.log_prob(action)).item()

    # remove gradient from action and reduce dims
    action = torch.squeeze(action).item()

    return action, prob


def advantage_calc(values, rewards, dones, lambda_, gamma):

    ### returns a list containing the advantages for each timestep 0 to T - 1 ###

    advantages = []

    # iterate over each timestep
    for i in range(len(rewards) - 1):

        # initialise advantage for timestep t and discount
        A_t = 0
        discount = 1.0

        # at each timestep iterate to the end of episode or T - 1 timesteps
        for j in range(i, len(rewards) - 1):

            # calculate TD error
            # if done is true then no t + 1 val will exist so TD error is simply
            # reward at t minus value at t
            if dones[j] == True:
                TD_error = rewards[j] - values[j]
            
            # if not done then can take value from t + 1 into the future
            else:
                TD_error = rewards[j] + (gamma * values[j + 1]) - values[j]
            
            # add to advantage for iter i
            A_t += TD_error * discount

            # update discount with smoothing and discount gamma
            discount *= (lambda_ * gamma)
            
            # if the episode is finished then break advantage update
            # it would not make sense to continue with discounted values 
            # at the start of the next episode
            if dones[j] == True:
                break
        
        # update advantage list
        advantages.append(A_t)
    
    # complete advantage list with zero at end
    advantages.append(0.0)
    
    return advantages


def batch_builder(probs, values, obs, rewards, actions, advantages, action_masks, batch_size):

    ### returns a list of dictionaries where each dictionary contains lists of ###
    ### memories of length batch size ###

    #get random indices
    indices = list(range(0, len(rewards)))
    np.random.shuffle(indices)

    batches = []

    #iterate as many times as full batches can be created
    for i in range(0, len(rewards), batch_size):

        #make sure batch size is correct size else continue
        if len(rewards) - i < batch_size:
            continue
        
        #collect all information in a batch dictionary
        batch_indices = indices[i:i + batch_size]
        batch_dict = {  'observations': np.expand_dims(np.array([obs[index] for index in batch_indices]), 1), 
                        'actions':      [actions[index] for index in batch_indices], 
                        'probs':        [probs[index] for index in batch_indices], 
                        'values':       [values[index] for index in batch_indices], 
                        'rewards':      [rewards[index] for index in batch_indices], 
                        'masks':        [action_masks[index] for index in batch_indices],
                        'advantages':   [advantages[index] for index in batch_indices]}

        #append to batches list
        batches.append(batch_dict)
    
    return batches


def train(actor, critic, actor_optimiser, critic_optimiser, batches, clip, c1):

    ### returns the actor and critic model after an iteration of training on batches ###

    #iterate through batches
    for batch in batches:
        
        #calculate critic loss using mean MSE
        current_values = torch.squeeze(critic(torch.tensor(batch['observations'])))
        returns = torch.squeeze(torch.tensor(batch['advantages']) + torch.tensor(batch['values']))
        critic_loss = (returns - current_values) ** 2
        critic_loss = critic_loss.mean()

        #calculate actor loss using clipped probs ratio and advantages
        dist = actor(torch.tensor(batch['observations']), torch.tensor(batch['masks']))
        new_probs = dist.log_prob(torch.tensor(batch['actions']))
        prob_ratio = new_probs.exp() / torch.tensor(batch['probs']).exp()
        unclipped_loss = torch.tensor(batch['advantages']) * prob_ratio
        clipped_loss = torch.clamp(prob_ratio, 1 - clip, 1 + clip) * torch.tensor(batch['advantages'])
        # actor loss should be negative as we are trying to maximise this value but the Adam
        # optimiser aims to minimise loss
        actor_loss = -torch.min(unclipped_loss, clipped_loss).mean()

        #add losses together
        total_loss = actor_loss + (c1 * critic_loss)

        #update weights
        actor_optimiser.zero_grad()
        critic_optimiser.zero_grad()
        total_loss.backward()
        actor_optimiser.step()
        critic_optimiser.step()
    
    return actor, critic


def reshape_image(observation):

    ### returns the observation array of dims (batches, 1, 7, 6) to put into networks ###

    #pad observation to make square
    padded_observation = np.pad(observation, [(1, 1), (1, 0), (0, 0)])

    #transpose to get channel first
    transposed_observation = np.transpose(padded_observation, (2, 0, 1))

    #compress observation to one channel and make opponents positions negative
    compressed_observation = transposed_observation[0, :, :] + (-1 * transposed_observation[1, :, :])

    #reshape to add batch size to the front
    reshaped_observation = np.reshape(compressed_observation, 
                                    (1, 1, compressed_observation.shape[0],
                                    compressed_observation.shape[1]))
    
    return reshaped_observation


def rand_action_picker(n_actions, observation):

    ### returns a random action from the action space ###

    # add one to the action space so can differentiate from non valid actions
    action_space = np.array(range(1, n_actions + 1))

    # mask action space with action mask
    action_space_masked = np.array(observation['action_mask']) * action_space

    # remove actions that appear as zero then minus 1 from remaining actions
    action_space_masked = np.array([x for x in action_space_masked if x > 0]) - 1

    # pick a random action from the remaining valid actions
    action = action_space_masked[np.random.randint(0, len(action_space_masked))]

    return action
  

# this function may not be completely necessary but is still used to add some
# stochasticity to the creation of new models
def algo_hyperparam_init(limits):

    ### returns a dictionary with the algorithms hyperparameters ###
    
    #initialise hyperparameters
    algo_hyperparameters = {}

    #iterate over hyperparameters
    for param in limits:

        #select a hyperparameter value randomly from the limits provided
        algo_hyperparameters[param] = np.random.uniform(limits[param][0], limits[param][1])

    return algo_hyperparameters
