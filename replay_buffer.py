import numpy as np
from collections import deque
import random
import torch

class ReplayBuffer:
	'''
	Class to hold the replay buffer
	'''
	def __init__(self, capacity):
		self.capacity = capacity
		self.actions_buffer = deque(maxlen=capacity)
		self.global_reward_buffer = deque(maxlen=capacity)
		self.old_local_obs_buffer = deque(maxlen=capacity)
		self.old_global_obs_buffer = deque(maxlen=capacity)
		self.new_local_obs_buffer = deque(maxlen=capacity)
		self.new_global_obs_buffer = deque(maxlen=capacity)
		self.local_rewards_buffer = deque(maxlen=capacity)

	def __len__(self):
		return len(self.buffer)

	def append(self, experience):
		'''
		Function to add to buffer

		Attributes
		----------
		- experience : torch tensor [actions, global_reward, old_local_obs, old_global_obs, new_local_obs
									new_global_obs, local_rewards, greedy_action] 
		'''
		self.actions_buffer.append(experience[0])
		self.global_reward_buffer.append(experience[1])
		self.old_local_obs_buffer.append(experience[2])
		self.old_global_obs_buffer.append(experience[3])
		self.new_local_obs_buffer.append(experience[4])
		self.new_global_obs_buffer.append(experience[5])
		self.local_rewards_buffer.append(experience[6])

	def sample(self, batch_size):
		'''
		Function to sample a batch of batch_size samples from the buffer

		Attributes
		----------
		- batch_size : the size of the mini batch

		Returns
		-------
		- actions : actions taken by each light
		- global_reward : tensor holding the global reward [batch_size]
		- old_local_obs : tensor holding local obs for each light [batch_size, num_lights, obs_size]
		- old_global_obs : tensor holding global observations [batch_size, obs_size]
		- new_local_obs : tensor holding new local obs for each light [batch_size, num_lights, obs_size]
		- new_global_obs : tensor holding the next global observations [batch_size, obs_size]
		- local_rewards : tensor holding the local reward from each light [batch_size, num_lights]
		- greedy_actions : the greedy action taken by the individual lights
		'''
		cur_capacity = len(self.actions_buffer)
		replace = cur_capacity < batch_size
		indices = np.random.choice(cur_capacity, batch_size, replace=replace)
		
		actions = torch.stack([self.actions_buffer[idx] for idx in indices])
		global_reward = torch.Tensor([self.global_reward_buffer[idx] for idx in indices])
		old_local_obs = torch.stack([self.old_local_obs_buffer[idx] for idx in indices])
		old_global_obs = torch.stack([self.old_global_obs_buffer[idx] for idx in indices])
		new_local_obs = torch.stack([self.new_local_obs_buffer[idx] for idx in indices])
		new_global_obs = torch.stack([self.new_global_obs_buffer[idx] for idx in indices])
		local_rewards = torch.stack([self.local_rewards_buffer[idx] for idx in indices])


		return (actions, global_reward, old_local_obs, old_global_obs, new_local_obs,
			new_global_obs, local_rewards)
