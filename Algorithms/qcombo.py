import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
import os
from copy import deepcopy
import datetime
import higher

class QCOMBO:
	'''
	Implementation of QCOMBO algorithm
	'''
	def __init__(self, n_rows, n_cols, config):
		'''
		Attributes
		----------
		- n_rows : the number of rows in the grid
		- n_cols : the number of columns in the grid
		- lr : the learning rate
		- discount : the discount factor in the TD target
		'''
		self.name = "QCOMBO"
		self.n_rows = n_rows
		self.n_cols = n_cols
		self.num_lights = n_rows * n_cols
		self.discount = config.alg.discount
		self.qcombo_lam = config.alg.qcombo_lam # Regularization coefficient
		self.lam = config.alg.lam
		self.exploration_rate = config.alg.exploration_rate
		self.config = config

		# Initialize the local and global networks
		self.local_net = LocalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount, config=config)
		self.global_net = GlobalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount, config=config)

		# initialize the target networks
		self.local_target_net = LocalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount, config=config)
		self.global_target_net = GlobalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount, config=config)
		self.local_target_net.eval()
		self.global_target_net.eval()

		# Inititalize second global net for finite difference methods
		self.global_copy_net = GlobalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount, config=config)

	def get_greedy(self, local_obs):
		'''
		Helper function to get the greedy action under the local neural network

		Attributes
		----------
		- local_obs : the local observation of each traffic light [batch_size, num_lights, input_count]

		Returns
		-------
		- greedy_actions : the greedy actions that would be taken by each light [batch_size]
		'''
		actions_list = []
		for i in range(self.num_lights):
			state_tensor = local_obs[:, i, :]
			Q = self.local_net(state_tensor) # [batch_size, 2]
			actions = torch.argmax(Q, dim=1) # [batch_size]
			actions_list.append(actions)

		greedy_local_actions = torch.stack(actions_list, dim=1) # [batch_size, num_lights]
		binary_coeff = torch.Tensor([2 ** (self.num_lights - i - 1) for i in range(self.num_lights)]) # [num_lights]
		global_action = torch.matmul(greedy_local_actions, binary_coeff.long()) # [batch_size]
		return global_action

	def get_reg_loss(self, global_obs, local_obs, actions, copy_net = False):
		'''
		Function to get the regularization loss as described in QCOMBO

		Attributes
		----------
		- global_obs : the global observation [batch_size, obs_size]
		- local_obs : the local observation [batch_size, num_lights, obs_size]
		- actions : the actions taken by the agent [batch_size, num_lights]

		Returns
		-------
		- loss : the regularization loss
		'''
		# First enumerate the global actions
		binary_coeff = torch.Tensor([2 ** (self.num_lights - i - 1) for i in range(self.num_lights)])
		global_actions = torch.matmul(actions, binary_coeff)


		if copy_net:
			global_Q = self.global_copy_net(global_obs) # [batch_size, 2 ** num_lights]	
		else:
			global_Q = self.global_net(global_obs) # [batch_size, 2 ** num_lights]

		global_Q_taken = global_Q[torch.arange(global_obs.shape[0]), global_actions.long()]

		local_Q = torch.zeros(global_obs.shape[0])
		for i in range(self.num_lights):
			local_obs_tensor = local_obs[:, i, :]
			local_actions = actions[:, i]
			Q = self.local_net(local_obs_tensor) # [batch_size, 2]
			Q_taken = Q[torch.arange(global_obs.shape[0]), local_actions.long()] # [batch_size]
			local_Q += Q_taken
		local_Q /= self.num_lights
		loss = self.local_net.loss_function(local_Q, global_Q_taken)
		return loss

	def get_adv_reg_loss(self, state_tensor, perturbed_state_tensor):
		'''
		Function to get the regularization part of the loss function based on adversarial perturbation

		Parameters
		----------
		- state_tensor : the non-perturbed state tensor
		- perturbed_state_tensor : the perturbed state tensor

		Returns
		-------
		- reg_loss : the regularization loss
		'''
		normal_Q = self.global_net(state_tensor)
		perturbed_Q = self.global_net(perturbed_state_tensor)
		reg_loss = torch.norm(normal_Q - perturbed_Q, p="fro")

		return reg_loss



	def train_step(self, replay, summarize=True):
		'''
		Function to train the agent via stochastic gradient descent

		Attributes
		----------
		- replay : the replay buffer holding the last 1000 examples
		- summarize : whether or not to summarize the training progress
		'''
		import datetime
		start = datetime.datetime.now()
		for i in range(self.config.alg.num_minibatches):
			actions, global_reward, old_local_obs, old_global_obs, new_local_obs, new_global_obs, local_rewards = \
									replay.sample(self.config.alg.minibatch_size)

			individual_loss = self.local_net.get_loss(
				old_state=old_local_obs, new_state=new_local_obs, actions=actions, rewards=local_rewards)
			greedy_actions = self.get_greedy(new_local_obs)
			global_loss = self.global_net.get_loss(old_global_state=old_global_obs,
												   new_global_state=new_global_obs, reward=global_reward,
												   actions=actions,
												   greedy_actions=greedy_actions)
			# Get the regularization loss
			reg_loss = self.get_reg_loss(global_obs=old_global_obs, local_obs=old_local_obs,
										 actions=actions)

			loss = individual_loss + global_loss + self.qcombo_lam * reg_loss
			if self.config.alg.perturb:

				if self.config.alg.stackelberg:
					perturbation = torch.normal(torch.zeros_like(old_global_obs), torch.ones_like(old_global_obs) * 1e-3).detach()
					perturbation = self.unroll_perturb(old_global_obs, perturbation)

					perturbed_tensor = old_global_obs + perturbation * torch.abs(old_global_obs.detach())

					adv_reg_loss = self.get_adv_reg_loss(old_global_obs, perturbed_tensor)
					loss = loss + self.lam * adv_reg_loss

					# Set gradients to zero
					self.local_net.optimizer.zero_grad()
					self.global_net.optimizer.zero_grad()

					loss.backward()
					self.local_net.optimizer.step()
					self.global_net.optimizer.step()

				else:
					# Get the regularization loss for a smooth policy

					perturbed_tensor = old_global_obs + torch.normal(torch.zeros_like(old_global_obs), torch.ones_like(old_global_obs) * 1e-3)
					perturbed_tensor.requires_grad = True
					obs_grad = torch.zeros(perturbed_tensor.shape)

					for k in range(self.config.alg.perturb_num_steps):
						# Calculate adversarial perurbation
						distance_loss = torch.norm(self.global_net(old_global_obs) - self.global_net(perturbed_tensor), p="fro")
						grad = torch.autograd.grad(outputs=distance_loss, inputs=perturbed_tensor, grad_outputs=torch.ones_like(loss), retain_graph=True, create_graph=True)[0]
						obs_grad = grad
						perturbed_tensor = perturbed_tensor + self.config.alg.perturb_alpha * grad * torch.abs(old_global_obs.detach())

					adv_reg_loss = self.get_adv_reg_loss(old_global_obs, perturbed_tensor)
					loss = loss + self.lam * adv_reg_loss

					# Set gradients to zero
					self.local_net.optimizer.zero_grad()
					self.global_net.optimizer.zero_grad()

					loss.backward()
					self.local_net.optimizer.step()
					self.global_net.optimizer.step()
			else:
				# Set gradients to zero
				self.local_net.optimizer.zero_grad()
				self.global_net.optimizer.zero_grad()

				loss.backward()
				self.local_net.optimizer.step()
				self.global_net.optimizer.step()

	def unroll_perturb(self, old_global_obs, perturbation, loss_scale=1e-5):
		'''
		Stackelberg adv training
		'''
		self.global_net.perturbation = nn.Parameter(perturbation)
		opt = optim.SGD([self.global_net.perturbation], lr=self.config.alg.perturb_alpha)
		logit = self.global_net(old_global_obs).detach().clone()

		with higher.innerloop_ctx(
			self.global_net, opt, copy_initial_weights=True) as (fmodel, diffopt):
				for k in range(self.config.alg.perturb_num_steps):
					perturbed_tensor = old_global_obs.detach().clone() + fmodel.perturbation

					loss = -1 * torch.norm(self.global_net(old_global_obs) - self.global_net(perturbed_tensor), p="fro") * loss_scale
					diffopt.step(loss)

				perturbation_save = fmodel.perturbation.detach().clone()

				# compute Stackelberg gradient
				perturbed_tensor = old_global_obs.detach().clone() + fmodel.perturbation
				loss = -1 * torch.norm(logit - fmodel(perturbed_tensor), p="fro") * loss_scale
				fmodel.perturbation.retain_grad()
				loss.backward()

				# Copy gradients to model
				param_dict = {}
				for name, param in fmodel.named_parameters():
					param_dict[name] = param.grad.detach().clone()
				for name, param in self.global_net.named_parameters():
					param.grad = param_dict[name]

		return perturbation_save


	def choose_action(self, new_state):
		'''
		Function to choose the next action based off of the previous state

		Attributes
		----------
		- new_state : the state the agent is currently in
		- epsilon : the probability that the agent explores

		Returns
		-------
		- decision in {0, 1}
		'''
		if np.random.uniform() < self.exploration_rate:
			decision = np.random.randint(0, 2)
			self.exploration_rate *= self.config.alg.anneal_exp
		else:
			Q = self.local_net(new_state)
			decision = int(torch.argmax(Q))

		return decision

	def update_targets(self):
		'''
		Function to update the target networks to the current policy networks
		'''
		self.local_target_net.load_state_dict(self.local_net.state_dict())
		self.global_target_net.load_state_dict(self.global_net.state_dict())

	def save(self, dir, model_id=None):
		'''
		Function to save the model

		Parameters
		----------
		- dir : the directory to save the model in
		'''
		self.local_net.save(dir, model_id)
		self.global_net.save(dir, model_id)

	def load(self, dir, model_id=None):
		'''
		Function to load a model

		Parameters
		----------
		- dir : the directory to load the model from
		'''
		self.local_net.load(dir, model_id)
		self.global_net.load(dir, model_id)

class LocalNet(nn.Module):
	'''
	Local neural network to carry out the individual part of QCOMBO
	'''
	def __init__(self, n_rows, n_cols, lr, discount, config):
		'''
		Attributes
		----------
		- n_rows : the number of rows in the grid
		- n_cols : the number of columns in the grid
		- lr : the learning rate
		- discount : the discount factor in the TD target
		'''
		super(LocalNet, self).__init__()
		self.num_lights = n_rows * n_cols
		self.n_rows = n_rows
		self.n_cols = n_cols
		self.lr = lr
		self.discount = discount
		self.config = config

		# Initialize the input and output counts
		self.input_count = 18 + self.num_lights
		self.output_count = 2

		# Initialize the neural network layers
		self.fc1 = nn.Linear(in_features=self.input_count, out_features=64)
		self.fc2 = nn.Linear(in_features=64, out_features=64)
		self.fc3 = nn.Linear(in_features=64, out_features=2)

		# Initialize the loss function
		self.loss_function = nn.MSELoss()

		# Initialize the optimzer
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

	def forward(self, x):
		'''
		Forward pass of the neural network

		Attributes
		----------
		- x : the model input [batch_size, input_count]

		Returns
		-------
		- Q : the predicted Q function [batch_size, output_count]
		'''
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		Q = self.fc3(y2)

		return Q

	def get_loss(self, old_state, new_state, actions, rewards):
		'''
		Function to get the loss (TD error)

		Attributes
		----------
		- old_state : the old observation of the agent [batch_size, num_lights, input_count]
		- new_state : new observations of the agent [batch_size, num_lights, input_count]
		- actions : tensor of agent actions [batch_size, num_lights]
		- rewards : tensor of rewards recieved by the agent [batch_size, num_lights]

		Returns
		-------
		- loss : the loss of the Q-network
		'''
		total_loss = 0
		for i in range(self.num_lights):
			old_state_tensor = old_state[:, i, :]
			new_state_tensor = new_state[:, i, :]
			action = actions[:, i]
			reward = rewards[:, i]
			old_Q = self.forward(old_state_tensor) # [batch_size, output_count]
			Q_taken = old_Q[torch.arange(self.config.alg.minibatch_size), action.long()] # [batch_size]

			new_Q = self.forward(new_state_tensor) # [batch_size, 2]
			max_Q = torch.max(new_Q, dim=1)[0] # [batch_size]

			target = reward + self.discount * max_Q
			loss = self.loss_function(Q_taken, target)
			total_loss += loss

		return total_loss

	def save(self, dir, model_id=None):
		torch.save(self.state_dict(), os.path.join(dir, 'QCOMBO_local_{}.pt'.format(model_id)))

	def load(self, dir, model_id=None):
		self.load_state_dict(torch.load(os.path.join(dir, 'QCOMBO_local_{}.pt'.format(model_id))))


class GlobalNet(nn.Module):
	'''
	Global Q-network in QCOMBO algorithm
	'''
	def __init__(self, n_rows, n_cols, lr, discount, config):
		'''
		Attributes
		----------
		- n_rows : the number of rows in the grid
		- n_cols : the number of columns in the grid
		- lr : the learning rate
		- discount : the discount factor in the TD target
		'''
		super(GlobalNet, self).__init__()
		self.num_lights = n_rows * n_cols
		self.n_rows = n_rows
		self.n_cols=n_cols
		self.lr = lr
		self.discount = discount
		self.config = config

		# Initialize the input and output counts
		self.input_count = 18 * self.num_lights
		self.output_count = 2 ** self.num_lights

		# Initialize the network parameters
		self.fc1 = nn.Linear(in_features=self.input_count, out_features=64)
		self.fc2 = nn.Linear(in_features=64, out_features=64)
		self.fc3 = nn.Linear(in_features=64, out_features=self.output_count)

		# Initialize the loss function
		self.loss_function = nn.MSELoss()

		# Initialize the optimizer
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

	def forward(self, x):
		'''
		Forward pass of the neural network

		Attributes
		----------
		- x : the model input [batch_size, input_count]

		Returns
		-------
		- Q : the predicted Q function [batch_size, output_count]
		'''
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		Q = self.fc3(y2)

		return Q

	def get_loss(self, old_global_state, new_global_state, reward, actions, greedy_actions):
		'''
		Function to get the global part of the QCOMBO loss

		Attributes
		----------
		- old_global_state : the old global observation [batch_size, input_count]
		- new_global_state : the new global observation [batch_size, num_lights, input_count]
		- reward : the global reward recieved [batch_size]
		- actions : actions taken by the agent [batch_size, num_lights]
		- greedy_actions : the greedy actions taken by the lights [batch_size]

		Returns
		-------
		- loss : the global loss function
		'''

		# First enumerate the global actions
		binary_coeff = torch.Tensor([2 ** (self.num_lights - i - 1) for i in range(self.num_lights)])
		global_action = torch.matmul(actions, binary_coeff)
		# Calculate the TD approximation
		old_Q = self.forward(old_global_state)
		Q_taken = old_Q[torch.arange(old_global_state.shape[0]), global_action.long()]
		# Calculate the TD target
		new_Q = self.forward(new_global_state) # [batch_size, 2 ** num_lights]

		greedy_Q = new_Q[torch.arange(old_global_state.shape[0]), greedy_actions.long()] # [batch_size]
		target = reward + self.discount * greedy_Q

		loss = self.loss_function(Q_taken, target)
		return loss

	def save(self, dir, model_id=None):
		torch.save(self.state_dict(), os.path.join(dir, 'QCOMBO_global_{}.pt'.format(model_id)))

	def load(self, dir, model_id=None):
		self.load_state_dict(torch.load(os.path.join(dir, 'QCOMBO_global_{}.pt'.format(model_id))))

