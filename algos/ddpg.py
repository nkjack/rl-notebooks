import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import numpy as np
import gym
from gym.spaces import Discrete, Box
from torch.distributions.categorical import Categorical
from typing import List
import scipy.signal
from copy import deepcopy
import time

"""
inspired by the openAI implementation, some code is copied from:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch
"""

"""
- select action using some policy on state and clip function
- create a replay buffer
- we need to to train the DQN so we calculate y targets and calc the mse. 
- To maximize the policy we need to maximze the Q-value, so we do it as well.
- update networks with polyak

pseudo-code:

Input:
    - policy network - p_network
    - Q-function network - q_network
    - empty replay buffer D - replay_buffer
    - target for policy + Q-function - p_target, q_target
    
Repeat
    observe state
    select action = clip(mu(s), a_low, a_high), where eps ~ N
    new_state, reward, is_done, _ =  env.step(action)
    
    replay_buffer.insert(state, action, reward, new_state, is_done)
    
    if is_done:
        _ = env.reset()
    
    is it time to update ?
        for update in num_of_updates:
            
            states, actions, rewards, n_states, is_dones = replay_buffer.random_batch()
            
            # compute targets
            y = rewards + delta * ( 1 - is_dones ) * q_target(new_states, p_target(new_states))
            
            # update Q-function by one step gradient descent
            loss_n = F.MSE(q_network(states, actions), y)
            
            # update policy by one step of gradient ascent
            loss_p = torch.mean(q_network(states, q_policy(states))
            
            # optimize
            
            # udpate target networks with
            q_target = polyak * q_target + (1 - polyak) * q_network
            p_target = polyak * p_target + (1 - polyak) * p_network
        
        end for
    end if
until convergence            


"""


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# making the policy netowork
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hid_sizes: List[int], activation):
        super(Actor, self).__init__()
        self.pi = mlp([obs_dim] + hid_sizes + [action_dim], activation, nn.Tanh)

    def forward(self, obs):
        return self.pi(obs)


class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_sizes: List[int], activation):
        super(DQN, self).__init__()
        self.q = mlp([obs_dim + act_dim] + hid_sizes + [1], activation)

    def forward(self, obs, act):
        qvals = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(qvals, -1) # Critical to ensure q has right shape.


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super(ActorCritic, self).__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit_high = action_space.high[0]
        act_limit_low = action_space.low[0]

        self.actor = Actor(obs_dim, act_dim, hidden_sizes, activation)
        self.DQN = DQN(obs_dim, act_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            return self.actor(obs) #.numpy()


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        """
        store (s,a,r,s',d)
        """
        self.states_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.states2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, o, a, r, n_o, d):
        self.states_buf[self.ptr] = o
        self.actions_buf[self.ptr] = a
        self.rewards_buf[self.ptr] = r
        self.states2_buf[self.ptr] = n_o
        self.done_buf[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.states_buf[idxs],
                      obs2=self.states2_buf[idxs],
                      act=self.actions_buf[idxs],
                      rew=self.rewards_buf[idxs],
                      done=self.done_buf[idxs])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def ddpg(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
         max_ep_len=1000, save_freq=1):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.DQN(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.DQN(o2, ac_targ.actor(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.DQN(o, ac.actor(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.actor.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.DQN.parameters(), lr=q_lr)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in ac.DQN.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.DQN.parameters():
            p.requires_grad = True

        # Record things
        # logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    ### IMPORTANT
    def get_action(o, noise_scale):
        a = ac.step(torch.as_tensor(o, dtype=torch.float32))
        a += torch.as_tensor(noise_scale * np.random.randn(act_dim), dtype=torch.float32)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):

                # rendering
                test_env.render()

                action = list(get_action(o, 0))
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(action)
                ep_ret += r
                ep_len += 1
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            print(f'test: ep_ret {ep_ret}, ep_len {ep_len}')

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = list(get_action(o, act_noise)) # Added the list cause was tensor
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            print(f'ep_ret {ep_ret} ep_len {ep_len}')
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling - where the actual training is happening
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs):
            #     logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            # logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            # logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            # logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('QVals', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            # logger.log_tabular('Time', time.time() - start_time)
            # logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--start_steps', type=int, default=10000)
    args = parser.parse_args()

    print('\nUsing ddpg formulation.\n')

    ddpg(lambda : gym.make(args.env), actor_critic=ActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs, start_steps=args.start_steps)


# example
## python ddpg.py --env LunarLanderContinuous-v2 --start_steps 500