# some code is taken from the openai spinup implementations -  https://github.com/openai/spinningup

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import numpy as np
import gym
from gym.spaces import Discrete, Box
from torch.distributions.categorical import Categorical
import scipy.signal


# making the policy netowork
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


# open-ai implementation
# def discount_cumsum(x, discount):
# 	return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def discount_cumsum(rews, discount):
    vals = []
    n = len(rews)
    for i in reversed(range(n)):
        if len(vals) == 0:
            vals.append(rews[i])
        else:
            vals.insert(0, rews[i] + discount * vals[0])
    return torch.as_tensor(vals, dtype=torch.float32)


def gae_lambda_adv(rews, vals, gamma: float, lam: float):
    """
    The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state
    """
    rews = np.append(rews, 0)
    vals = np.append(vals, 0)

    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    return discount_cumsum(deltas, gamma * lam)


def calc_qvals(rews, gamma: float):
    """
        Calculates the q-value q(s,a), i.e. total discounted reward, for each
        step s and action a of a trajectory.
        :param gamma: discount factor.
        :return: A list of q-values, the same length as the number of
        experiences in this Experience.
        """
    qvals = []
    n = len(rews)
    for i in reversed(range(n)):
        if len(qvals) == 0:
            qvals.append(rews[i])
        else:
            qvals.insert(0, rews[i] + gamma * qvals[0])
    # ========================
    return torch.as_tensor(qvals, dtype=torch.float32)


def train(env_name='CartPole-v0', hidden_sizes=[128,128], lr=1e-2, epochs=50, batch_size=5000, render=False, gamma=0.99,
          lam=0.95, adv='vnl'):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    logits_net_actions = mlp([obs_dim] + hidden_sizes + [n_acts], activation=nn.ReLU)
    logits_net_states = mlp([obs_dim] + hidden_sizes + [1])
    print("Net Actions:\n", logits_net_actions)
    print("Net States:\n", logits_net_states)

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net_actions(obs)
        return Categorical(logits=logits)  # logits mean similar to softmax

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        with torch.no_grad():
            logp = get_policy(obs).sample().item()
        return logp

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, acts, weights):
        logp = get_policy(obs).log_prob(acts)
        return - (logp * weights).mean()

    def compute_weight_loss(obs, rtgs):
        weights = logits_net_states(obs)

    def compute_value_state(obs):
        with torch.no_grad():
            values = logits_net_states(obs)
        return values

    def compute_advantage_loss(obs, rewards):
        values = logits_net_states(obs).squeeze()

        criterion = nn.MSELoss()
        return criterion(values, rewards)

    # make optimizer
    optimizer_policy = Adam(logits_net_actions.parameters(), lr=lr)
    optimizer_advantage = SGD(logits_net_states.parameters(), lr=lr, momentum=0.9)

    # running one epoch of training
    def train_one_epoch(epoch_num):

        batch_obs = []
        batch_actions = []
        batch_weights = []
        batch_rewards = []
        batch_lens = []
        batch_rtgs1 = []
        batch_rtgs2 = []
        batch_episode_rewards = []
        batch_adv = []

        obs = env.reset()
        done = False
        episode_rewards = []
        episode_vals = []

        # render first episode of each epoch
        finished_rendering_this_epoch = False
        while True:

            # rendering
            if (not finished_rendering_this_epoch and epoch_num % 10 == 0) and render:
                env.render()

            # observe the env
            batch_obs.append(obs.copy())

            # value
            value_state = compute_value_state(torch.as_tensor(obs, dtype=torch.float32))
            # batch_weights.append(value_state)
            episode_vals.append(value_state)

            # calculate step
            action = get_action(torch.as_tensor(obs, dtype=torch.float32))

            # take a step and observe env
            obs, reward, done, _ = env.step(action)

            # add it to trajactory
            batch_actions.append(action)
            episode_rewards.append(reward)

            if done:
                # calculate total rewards
                episode_length, episode_reward = len(episode_rewards), sum(episode_rewards)
                # batch_rewards.append(episode_reward)
                batch_rewards += episode_rewards
                batch_lens.append(episode_length)
                batch_episode_rewards.append(episode_reward)

                batch_weights += episode_vals

                # calculate weights
                # the weight for each logprob(a|s) is R(tau)
                # batch_weights += [episode_reward] * episode_length # no reward to go
                batch_rtgs1 += list(reward_to_go(episode_rewards))
                batch_rtgs2 += list(calc_qvals(episode_rewards, gamma))

                # for GAE
                vals = torch.as_tensor(episode_vals, dtype=torch.float32)
                rews = torch.as_tensor(episode_rewards, dtype=torch.float32)
                batch_adv += list(gae_lambda_adv(rews, vals, gamma, lam))

                obs, done, episode_rewards, episode_vals = env.reset(), False, [], []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # brake from loop
                if len(batch_obs) > batch_size:
                    break

        ## optimize policy
        optimizer_policy.zero_grad()

        vals = torch.as_tensor(batch_weights, dtype=torch.float32)
        obs = torch.as_tensor(batch_obs, dtype=torch.float32)

        ## reward for value state
        # rtgs = torch.as_tensor(batch_rtgs1, dtype=torch.float32) # rewards without discount
        rtgs = torch.as_tensor(batch_rtgs2, dtype=torch.float32) # rewards with discount

        # regular advantage
        # weights = rtgs - vals

        ## GAE-Lambda advantage
        weights = torch.as_tensor(batch_adv, dtype=torch.float32)

        # calculate loss
        loss_policy = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                   acts=torch.as_tensor(batch_actions, dtype=torch.int32),
                                   weights=weights)
        loss_policy.backward()
        optimizer_policy.step()

        ## optimize advantage
        optimizer_advantage.zero_grad()
        loss_advantage = compute_advantage_loss(obs=obs, rewards=rtgs)

        loss_advantage.backward()
        optimizer_advantage.step()

        return loss_policy, loss_advantage, batch_episode_rewards, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss_policy, batch_loss_advantage, batch_rewards, batch_lens = train_one_epoch(i)
        print('epoch: %3d \t loss_policy: %.3f \t loss_advantage: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss_policy, batch_loss_advantage, np.mean(batch_rewards), np.mean(batch_lens)))

    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', '--bs', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()
    print('\nUsing vanille formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr, batch_size=args.batch_size,  epochs=args.epochs)
