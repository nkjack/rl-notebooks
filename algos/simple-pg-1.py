# some code is taken from the openai spinup implementations -  https://github.com/openai/spinningup
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from torch.distributions.categorical import Categorical

# making the policy netowork
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
	layers = []
	for j in range(len(sizes)-1):
		act = activation if j < len(sizes) - 2 else output_activation
		layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
	return nn.Sequential(*layers)

def reward_to_go(rews):
	n = len(rews)
	rtgs = np.zeros_like(rews)
	for i in reversed(range(n)):
		rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
	return rtgs

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):
	# make environment, check spaces, get obs / act dims
	env = gym.make(env_name)
	assert isinstance(env.observation_space, Box), "This example only works for envs with continuous state spaces."
	assert isinstance(env.action_space, Discrete), "This example only works for envs with discrete action spaces."

	obs_dim = env.observation_space.shape[0]
	n_acts = env.action_space.n

	logits_net = mlp([obs_dim]+hidden_sizes+[n_acts])

	# make function to compute action distribution
	def get_policy(obs):
		logits = logits_net(obs)
		return Categorical(logits=logits) # logits mean similar to softmax

	# make action selection function (outputs int actions, sampled from policy)
	def get_action(obs):
		return get_policy(obs).sample().item()

	# make loss function whose gradient, for the right data, is policy gradient
	def compute_loss(obs, acts, weights):
		logp = get_policy(obs).log_prob(acts)
		return - (logp * weights).mean()

	
	# make optimizer
	optimizer = Adam(logits_net.parameters(), lr=lr)

	# running one epoch of training
	def train_one_epoch():

		batch_obs = []
		batch_actions = []
		batch_weights = []
		batch_rewards = []
		batch_lens = []

		obs = env.reset()
		done = False
		episode_rewards = []

		# render first episode of each epoch
		finished_rendering_this_epoch = False

		while True:

			# rendering
			if (not finished_rendering_this_epoch) and render:
				env.render()

			# observe the env
			batch_obs.append(obs.copy())

			# calculate step
			action = get_action(torch.as_tensor(obs, dtype=torch.float32))

			# take a step and observe env
			obs, reward, done,  _ = env.step(action)

			# add it to trajactory
			batch_actions.append(action)
			episode_rewards.append(reward)

		
			if done:
				# calculate total rewards
				episode_length, episode_reward = len(episode_rewards), sum(episode_rewards)
				batch_rewards.append(episode_reward)
				batch_lens.append(episode_length)

				# calculate weights 
				# the weight for each logprob(a|s) is R(tau)
				# batch_weights += [episode_reward] * episode_length # no reward to go
				batch_weights += list(reward_to_go(episode_rewards))

				obs, done, episode_rewards = env.reset(), False, []
				
				# won't render again this epoch
				finished_rendering_this_epoch = True

				# brake from loop
				if len(batch_obs) > batch_size:
					break

		# load batch
		# restart optimizer
		optimizer.zero_grad()
		# calculate loss
		loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
							acts=torch.as_tensor(batch_actions, dtype=torch.int32),
							weights=torch.as_tensor(batch_weights, dtype=torch.float32))
		# backprop
		loss.backward()
		# optimizer step
		optimizer.step()
		return loss, batch_rewards, batch_lens

	# training loop
	for i in range(epochs):
		batch_loss, batch_rewards, batch_lens = train_one_epoch()
		print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
			(i, batch_loss, np.mean(batch_rewards), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', '--bs', type=int, default=5000)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr, batch_size=args.batch_size)
