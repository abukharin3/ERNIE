
"""Multi-agent traffic light"""
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from Env import TrafficGridEnv, TrafficAction
from replay_buffer import ReplayBuffer
from Algorithms.qcombo import QCOMBO
from Algorithms.coma import COMA
from collections import deque
import argparse
import os
import time
from eval_configs import eval1, eval2, eval3, eval4, eval5, eval6, eval7, eval8, eval_4x4, eval_6x6_0, eval_6x6_1,eval_6x6_2,eval_6x6_3,eval_6x6_4, train_2x2, eval_custom_0, eval_custom_1
import random

parser = argparse.ArgumentParser(description='RL Experiment.')
parser.add_argument('--alg', type=str, default='qcombo', choices=['qcombo','coma'])
parser.add_argument('--nrow', type=int, default=2,
					help='n_row of environment')
parser.add_argument('--ncol', type=int, default=2, 
					help='n_col of environment')

class LightTrainer:
	'''
	Class to train flow traffic lights from a partially observed MDP
	Attributes
	----------
	- num_lights : the number of traffic lights in the network
	- model : the model we are using for the Q-network
	'''
	def __init__(self, alg, N_ROWS, N_COLUMNS, config, saving_dir=None, seed=0):
		self.alg = alg
		self.N_ROWS = N_ROWS
		self.N_COLUMNS = N_COLUMNS
		self.num_lights = N_COLUMNS * N_ROWS
		self.config = config
		self.seed = s

		if saving_dir is None:
			self.saving_dir = "results/{}{}_policy{}x{}_{}_{}".format(self.alg.name, self.config.alg.perturb,self.N_ROWS, self.N_COLUMNS, self.alg.lam,
				self.config.alg.perturb_alpha)
		else:
			self.saving_dir = saving_dir
		os.makedirs(self.saving_dir, exist_ok=True)

	def process_local_observation(self, obs, last_change, n_lights = None):
		'''
		Function to combine environment observation, last light change,
		and observation area into torch tensors for use in local Q-network
		Attributes
		----------
		- obs : Dictionary of observation from each light
		- last_change : tensor containing the last change of each light [num_lights, 1]
		Returns
		-------
		- obs_tensor : a tensor containing observations for each light [num_lights, obs_size]
		'''
		if n_lights is None:
			n_lights = self.num_lights

		last_change = np.ndarray.flatten(last_change)
		vals = list(obs.values())
		obs_tuple = []
		for i in range(n_lights):
			one_hot = np.zeros(self.num_lights)
			one_hot[i % self.num_lights] = 1
			local_obs = vals[i]
			obs_arr = np.concatenate([local_obs, [last_change[i]], one_hot])
			obs_tensor = torch.from_numpy(obs_arr).float()
			obs_tuple.append(obs_tensor)

		obs_tensor = torch.stack(obs_tuple)
		return obs_tensor

	def process_global_observation(self, obs, last_change):
		'''
		Function to combine environment observation, last light change,
		and observation area into torch tensors for use in local Q-network
		Attributes
		----------
		- obs : Dictionary of observation from each light
		- last_change : tensor containing the last change of each light [num_lights, 1]
		Returns
		-------
		- obs_tensor : a tensor containing the global observations 
		'''
		last_change = np.ndarray.flatten(last_change)
		vals = list(obs.values())
		obs_arr = np.concatenate(vals)
		obs_arr = np.concatenate([obs_arr, last_change])
		obs_tensor = torch.from_numpy(obs_arr).float()

		return obs_tensor

	def random_action(self):
		'''
		Function to choose random action
		'''
		decision = np.random.randint(0,2)
		return decision

	def enumerate_action(self, action, tensor = False):
		'''
		Helper function to enumerate an action for the global Q function
		Attributes
		----------
		- action : array of 0 and 1's containing action of each light
		Returns
		-------
		- enum : int in 0 through 2 ^ num_lights
		'''

		coeff = [2.0 ** (self.num_lights - 1 - i) for i in range(self.num_lights)]
		coeff = torch.Tensor(coeff)
		enum = torch.matmul(action.float(), coeff)

		return enum

	def train_episodes(self, n_episodes, save_reward=False):
		'''
		Function to train agent over multiple episodes
		'''
		print(self.saving_dir)
		# Initialize replay buffer
		replay = ReplayBuffer(self.config.main.replay_capacity)
		for episode in range(n_episodes):
			# Initialize environment
			flow = self.config.env.train_parameters
			my_env = TrafficGridEnv(N_ROWS=self.N_ROWS, N_COLUMNS=self.N_COLUMNS, flow=flow)
			env = my_env.make_env()

			# Get initial observation from environment
			obs = env.reset()
			last_change = env.last_change

			# Store previous observation and actions
			old_local_obs = self.process_local_observation(obs, last_change)
			old_global_obs = self.process_global_observation(obs, last_change)
			old_actions = torch.zeros(self.num_lights)

			# Initialize action
			actions = TrafficAction(act=0, N_ROWS=self.N_ROWS, N_COLUMNS=self.N_COLUMNS)
			is_training = False
			r = []

			start_iter = 1
			end_iter = self.config.main.train_iters + 1	

			for i in range(start_iter, end_iter):
				# For first 1000 timesteps, take random actions
				if i <= 1000:
					for j in range(self.num_lights):
						if i % 10 == 0:
							action = self.random_action()
							actions["center{}".format(j)] = action
						else:
							action = 0
							actions["center{}".format(j)] = action

				if i % 100 == 0:
					print("Episode: ", episode, "Iter: ", i)

				if i >= 1000 or episode > 0:
					is_training =  True



				obs, _, done, info = env.step(actions)
				last_change = env.last_change
				rewards = env.compute_individual_reward(actions, obs, last_change)
				
				
				'''
				For each transition, we must add to the replay buffer:
				- actions : the action taken by each light [num_lights]
				- global_reward : the global reward of the transition (scalar)
				- old_local_obs : the local observation of each light [num_lights, obs_size]
				- old_global_obs : the global observation [obs_size]
				- new_local_obs : the local observation of each light [num_lights, obs_size]
				- new_global_obs : the global observation [obs_size]
				- local_rewards : the reward at each light [num_lights]
				- greedy_action : the greedy action that would be taken by the lights in the new state (int)
				'''
				actions_tensor = torch.Tensor([int(actions["center{}".format(i)]) for i in range(self.num_lights)])
				global_reward = sum(rewards.values())

				new_local_obs = self.process_local_observation(obs, last_change) 
				new_global_obs = self.process_global_observation(obs, last_change)

				local_rewards = torch.Tensor(list(rewards.values()))
				experience = [actions_tensor, global_reward, old_local_obs, old_global_obs,
							new_local_obs, new_global_obs, local_rewards]
				replay.append(experience)


				# Train every fifth step
				if is_training and i % self.config.main.update_period == 0:
					self.alg.train_step(replay, summarize = False)

				# Make next decision
				old_actions = []
				if i >= 1000 or episode > 0:
					for j in range(self.num_lights):
						new_state_tensor = new_local_obs[j, :]
						action = self.alg.choose_action(new_state_tensor)

						actions["center{}".format(j)] = action
						old_actions.append(action)
					
				# Update observations
				old_local_obs = new_local_obs
				old_global_obs = new_global_obs
				old_actions = actions_tensor
				r.append(global_reward)

			r = np.array(r)
			if save_reward:	
				np.save(f"{self.saving_dir}/training_reward_{episode}.npy", r)

	def eval(self, render=False, eval_dict=None, perturb=False, perturb_size = 0, speed = 35, n_lights = 4):
		'''
		Function to evaluate model on SUMO environment
		Attributes
		----------
		- N_ROWS : the number of rows in the traffic grid
		- N_COLUMNS : the number of columns in the traffic grid
		'''
		flow=700
		# Initialize environment
		if n_lights == 4:
			my_env = TrafficGridEnv(N_ROWS=2, N_COLUMNS=2, render=False, eval_dict=eval_dict,
									speed=speed)
		elif n_lights == 16:
			my_env = TrafficGridEnv(N_ROWS=4, N_COLUMNS=4, render=False, eval_dict=eval_dict,
									speed=speed)
		env = my_env.make_env()

		# Get initial observation from environment
		obs = env.reset()
		last_change = env.last_change

		# Store previous observation and actions
		old_local_obs = self.process_local_observation(obs, last_change, n_lights = n_lights)
		old_global_obs = self.process_global_observation(obs, last_change)
		old_actions = torch.zeros(self.num_lights)


		# Initialize action
		actions = TrafficAction(act=0, N_ROWS=self.N_ROWS, N_COLUMNS=self.N_COLUMNS)

		# Run simulation
		r = []
		for i in range(1, self.config.main.eval_iters + 1):
			if i % 100 == 0:
				print(i)
			if i <= 1000:
				for j in range(n_lights):
					if i % 9 == 0:
						action = self.random_action()
						actions["center{}".format(j)] = action
					else:
						action = 0
						actions["center{}".format(j)] = action


			obs, _, done, info = env.step(actions)
			last_change = env.last_change
			rewards = env.compute_individual_reward(actions, obs, last_change)

			actions_tensor = torch.Tensor([int(actions["center{}".format(i)]) for i in range(n_lights)])
			global_reward = sum(rewards.values())

			new_local_obs = self.process_local_observation(obs, last_change, n_lights = n_lights) 
			new_global_obs = self.process_global_observation(obs, last_change)

			local_rewards = torch.Tensor(list(rewards.values()))

			# Make next decision
			old_actions = []
			if i >= 1000:
				for j in range(n_lights):
					new_state_tensor = new_local_obs[j, :]

					if perturb:
						# Perturb new_state_tensor
						new_state_tensor = new_state_tensor + torch.normal(torch.zeros(new_state_tensor.shape), torch.ones_like(new_state_tensor) * perturb_size)

					action = self.alg.choose_action(new_state_tensor)
					actions["center{}".format(j)] = action
					old_actions.append(action)

			# Update observations
			old_local_obs = new_local_obs
			old_global_obs = new_global_obs
			old_actions = actions_tensor
			r.append(global_reward)
		
		r = np.array(r)

		np.save(f"{self.saving_dir}/eval_reward_{perturb}_{perturb_size}_{speed}_{n_lights}.npy", r)


	def eval_marl(self, perturb_size = 1e-1, render=False, eval_dict=None, eval_num=0, seed=0, cutoff = 0.05):
		'''
		Function to evaluate an agent trained on the 2x2 grid on a 4x4 grid
		Attributes
		----------
		- N_ROWS : the number of rows in the traffic grid
		- N_COLUMNS : the number of columns in the traffic grid
		'''
		n_lights = 4
		flow=700
		# Initialize environment
		my_env = TrafficGridEnv(N_ROWS=2, N_COLUMNS=2,render=render, eval_dict=eval_dict)
		env = my_env.make_env()

		# Get initial observation from environment
		obs = env.reset()
		last_change = env.last_change

		# Store previous observation and actions
		old_local_obs = self.process_local_observation(obs, last_change, n_lights = n_lights) # Change to include MF Action
		old_global_obs = self.process_global_observation(obs, last_change)
		old_actions = torch.zeros(self.num_lights)

		# Initialize action
		actions = TrafficAction(act=0, N_ROWS=self.N_ROWS, N_COLUMNS=self.N_COLUMNS)

		# Run simulation
		is_training = False
		r = []

		log_path = self.saving_dir
		header = 'timestamp,time,'
		header += ','.join(['reward_{}'.format(idx) for idx in range(self.num_lights)] + ['global_reward'])
		header += '\n'
		start_time = time.time()
		with open(os.path.join(log_path, 'eval_log_{}.csv'.format(flow)), 'w') as f:
			f.write(header)

		for i in range(1, self.config.main.eval_iters + 1):
			if i % 100 == 0:
				print(i)
			if i <= 1000:
				for j in range(n_lights):
					if i % 9 == 0:
						action = self.random_action()
						actions["center{}".format(j)] = action
					else:
						action = 0
						actions["center{}".format(j)] = action


			obs, _, done, info = env.step(actions)
			last_change = env.last_change
			rewards = env.compute_individual_reward(actions, obs, last_change)

			actions_tensor = torch.Tensor([int(actions["center{}".format(i)]) for i in range(n_lights)])
			global_reward = sum(rewards.values())

			new_local_obs = self.process_local_observation(obs, last_change, n_lights=n_lights)
			new_global_obs = self.process_global_observation(obs, last_change)

			# Find mean field actions if the algorithm is a mean field algorithm
			if self.alg.mean_field:
				new_local_obs, new_actions = self.alg.get_mf_action(old_actions=old_actions,
					new_local_obs=new_local_obs)

			local_rewards = torch.Tensor(list(rewards.values()))

			# Train every fifth step
			if i % self.config.main.log_period == 0:
				s =  '{},{}' .format(i, time.time() - start_time)
				for idx in range(self.num_lights):
					s += ',{}'.format(rewards['center{}'.format(idx)])
				s += ',{}'.format(global_reward)
				s += '\n'
				with open(os.path.join(log_path, 'eval_log_{}.csv'.format(flow)), 'a') as f:
					f.write(s)
				# print(i, global_reward)



			# Make next decision
			old_actions = []
			if i >= 1000:
				for j in range(n_lights):
					if self.alg.mean_field:
						action = new_actions[j]
					elif self.alg.share_obs:
						action = self.alg.local_agents[j].choose_action(new_global_obs)
					else:
						new_state_tensor = new_local_obs[j, :]
						# Perturb new_state_tensor
						new_state_tensor = new_state_tensor + torch.normal(torch.zeros(new_state_tensor.shape), torch.ones_like(new_state_tensor) * perturb_size)
						action = self.alg.choose_action(new_state_tensor)
						if np.random.uniform(0, 1) < cutoff:
							action = 1 - action
					actions["center{}".format(j)] = action
					old_actions.append(action)
				to_print = [int(actions["center{}".format(k)]) for k in range(self.num_lights)]
			# Update observations
			old_local_obs = new_local_obs
			old_global_obs = new_global_obs
			old_actions = actions_tensor
			r.append(global_reward)
			# self.tfboard_writer.add_scalar('GlobalReward/eval', global_reward, i)
		
		r = np.array(r)

		print(f"{self.saving_dir}/eval_reward_perturbed_agent"+"_" + str(cutoff) + str(seed) + "_.npy")
		np.save(f"{self.saving_dir}/eval_reward_perturbed_agent"+"_" + str(cutoff) + str(seed) + "_.npy", r)	

	def save(self, subdir=""):
		"""
		Save the agent
		Attributes
		----------
		- subdir : sub directory to save the algo
		"""
		self.alg.save(f"{self.saving_dir}/{subdir}")

	def load(self, dir=None):
		"""
		Load the agent
		Attributes
		----------
		- dir : directory to load the algo
		"""
		if dir is None:
			dir = self.saving_dir
		self.alg.load(dir)


if __name__ == "__main__":
	args = parser.parse_args()

	# Choose algorithm
	if args.alg == 'qcombo':
		from Algorithms.configs import config_qcombo
		config = config_qcombo.get_config()
		alg = QCOMBO(n_rows = config.env.n_rows, n_cols = config.env.n_cols, config = config)
	elif args.alg == 'coma':
		from Algorithms.configs import config_coma
		config = config_coma.get_config()
		alg = COMA(n_rows = config.env.n_rows, n_cols = config.env.n_cols, config = config)

	for s in range(5):
		print("\n", "QCOMBO", s, "\n")
		config.alg.perturb = False
		# Set random seed
		torch.manual_seed(s)
		np.random.seed(s)
		random.seed(s)

		save_dir = "results/QCOMBO{}".format(s)
		trainer = LightTrainer(alg=alg, N_ROWS=config.env.n_rows, N_COLUMNS=config.env.n_cols, config=config,
				saving_dir=save_dir, seed=s)
		trainer.train_episodes(3, save_reward=True)
		trainer.save()
		trainer.eval(eval_dict=train_2x2, speed=35, perturb=True, perturb_size=1e-1)
