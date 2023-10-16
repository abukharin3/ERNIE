import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim

import os

class Actor(nn.Module):
	'''
	Class to implement the actor neural network that parametrizes the policy
	'''
	def __init__(self, n_rows, n_cols, lr = 0.05, discount=0.95):
		'''
		Attributes
		----------
		n_rows : the number of rows in the grid
		n_cols: the number of columns in the grid
		lr : the learning rate
		discount : the discount factor
		'''
		super(Actor, self).__init__()
		self.num_lights = n_rows * n_cols
		self.share_obs = False
		self.discount = discount
		self.share_obs = False
		# Declare the input and output counts
		self.input_count = 18 + self.num_lights
		self.output_count = 2

		# Initialize Actor network
		self.fc1 = nn.Linear(in_features=self.input_count, out_features=64)
		self.fc2 = nn.Linear(in_features=64, out_features=64)
		self.fc3 = nn.Linear(in_features=64, out_features=self.output_count)

		# Initialize the optimizer
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

	def forward(self, s):
		'''
		Forward pass of the neural network
		Attributes
		----------
		- s : the input to the neural network [batch_size, input_count]
		Returns
		-------
		- pi : the policy at state s [batch_size, 2]
		'''
		dim = len(list(s.size())) - 1

		y1 = F.relu(self.fc1(s))
		y2 = F.relu(self.fc2(y1))
		pi = F.softmax(self.fc3(y2), dim=dim)
		
		return pi

	def get_loss(self, actions, probs, advantages):
		'''
		Function to get the loss function of the actor network
		Attributes
		----------
		- actions : the actions taken by each agent [batch_size, num_lights]
		- probs : the probability taken for each action [batch_size, num_lights, action_shape]
		- advantages : the COMA advantages [batch_size, num_lights, 1]
		'''

		batch_size = actions.size()[0]
		actions = actions.int().numpy()
		actions_one_hot = np.zeros([batch_size, self.num_lights, 2])
		grid = np.indices((batch_size, self.num_lights))
		actions_one_hot[grid[0], grid[1], actions] = 1

		actions_one_hot = torch.from_numpy(actions_one_hot)
		probs = torch.sum(torch.mul(actions_one_hot, probs), dim=2, keepdim=True)
		log_probs = torch.log(probs + 1e-15)

		loss = torch.mul(log_probs, advantages)
		loss = torch.sum(loss)
		return -loss.mean()


		



class Critic(nn.Module):
	'''
	Class to implement the critic network that approximates the Q function
	'''
	def __init__(self, n_rows, n_cols, lr=0.05, discount=0.95):
		'''
		Attributes
		----------
		n_rows : the number of rows in the grid
		n_cols: the number of columns in the grid
		lr : the learning rate
		discount : the discount factor
		'''
		super(Critic, self).__init__()
		self.num_lights = n_rows * n_cols
		self.discount = discount
		self.lr = lr

		# Declare the input and output counts
		self.input_count = (18 + self.num_lights) * self.num_lights + self.num_lights
		self.output_count = 1

		# Initialize the critic network (approximates the Q functioon)
		self.fc1 = nn.Linear(in_features=self.input_count, out_features=256)
		self.fc2 = nn.Linear(in_features=256, out_features=256)
		self.fc3 = nn.Linear(in_features=256, out_features=self.output_count)

		# Initialize loss function
		self.loss_function = nn.MSELoss()

		# Initialize the optimizer
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

	def forward(self, obs, actions):
		'''
		Forward pass of the neural network
		Attributes
		----------
		- obs : the input to the neural network [batch_size, input_count]
		- actions : the actions each agent takes [batch_size, 1]
		Returns
		-------
		- Q : the estimates Q function [batch_size, num_lights, 2]
		'''

		obs_shape = obs.size()[1]
		joint_obs = obs.view(-1, self.num_lights * obs_shape)
		joint_actions = actions.view(-1, self.num_lights).float()

		x = torch.cat((joint_obs, joint_actions), dim=1)
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		Q = self.fc3(y2)

		return Q.repeat_interleave(self.num_lights, dim=0) 
	
	def get_baseline(self, obs, actions, probs):
		'''
		Counterfactual baseline for COMA
		Attributes
		----------
		- obs : the input to the neural network [batch_size, num_lights, input_count]
		- actions : the actions each agent takes [batch_size, num_lights, 1]
		- probs : the policies of the agents [batch_size, num_lights, 2]
		Returns
		-------
		- A : the counterfactual advantage [batch_size, num_lights, 1]
		'''
		batch_size = actions.size()[0]
		obs_shape = obs.size()[2]

		obs = obs.view(-1, obs_shape)
		actions = actions.view(-1, 1)

		# Q : [batch_size, num_lights, 1]
		Q = self.forward(obs, actions).view(-1, self.num_lights, 1)

		action_shape = 2
		possible_actions = [0, 1]

		# baseline_obs : [batch_size * self.num_lights * (self.num_lights * num_actions), obs_shape]
		# Contains the obs of agent 1, ..., n repeated self.num_lights * num_actions times for each training sample
		baseline_obs = obs.view(-1, self.num_lights, obs_shape).repeat_interleave(self.num_lights * action_shape, dim=0)
		baseline_obs = baseline_obs.view(-1, obs_shape)

		# baseline_actions : [batch_size * (self.num_lights * 2), self.num_lights]
		# Contains counterfactual actions (joint for now) 
		baseline_actions = actions.view(-1, self.num_lights).repeat_interleave(self.num_lights * action_shape, dim=0)

		baseline_probs = torch.zeros(batch_size * self.num_lights * action_shape, 1)

		# Assign actions
		for i, action in enumerate(baseline_actions):
			batch_index = i % (self.num_lights * action_shape)
			batch_num = i // (self.num_lights * action_shape)
			agent_index = batch_index // action_shape
			action_index = batch_index % action_shape
			baseline_action = possible_actions[action_index]
			action[agent_index] =  baseline_action

			# Ensure that the probs are changed too
			baseline_probs[i][0] = probs[batch_num][agent_index][baseline_action]
		
		baseline_actions = baseline_actions.view(-1, 1)

		baseline_values = self.forward(baseline_obs, baseline_actions).view(-1, action_shape, self.num_lights, 1)
		# self.forward repeats Q values self.num_lights times, so we only need the first
		baseline_values = baseline_values[:, :, 0, :]
		baseline_probs = baseline_probs.view(-1, action_shape, 1)

		baseline_expectation = torch.sum(torch.mul(baseline_probs, baseline_values), dim=1)
		baseline_expectation = baseline_expectation.view(-1, self.num_lights, 1)

		return Q - baseline_expectation


	def get_target(self, rewards, new_obs, new_actions):
		'''
		Function to get the TD target for the Q function
		Arguments
		---------
		- rewards : the reward recieved at each step [batch_size, num_lights]
		- new_obs : the obs landed in at each step [batch_size, num_lights, input_count]
		- new_actions : the actions taken by the agents with new_obs [batch_size, num_lights, 1]
		Returns
		-------
		- target : the TD target [batch_size]
		'''
		input_shape = new_obs.size()[2]
		new_obs = new_obs.view(-1, input_shape)
		new_actions = new_actions.view(-1, 1)

		new_Q = self.forward(new_obs, new_actions).view(-1, self.num_lights, 1)
		
		rewards = torch.unsqueeze(rewards, dim=2)
		target = rewards + self.discount * new_Q

		return target

	def get_loss(self, old_local_obs, local_actions, new_local_obs, new_local_actions, local_rewards):
		'''
		Function to get the loss of the critic network
		Arguments
		---------
		- old_local_obs : the previous state each agent was in [batch_size, num_lights, num_obs]
		- local_rewards : the reward recieved by each agent [batch_size, num_lights]
		- local_actions : the actions taken by each agent [batch_size, num_lights]
		- new_local_obs : the next state the agent is in [batch_size, num_lights, num_obs]
		- new_local_actions : the actions taken by agents with new_local_obs [batch_size, num_lights, 1]
		Returns
		-------
		- avg_loss : the average loss of each light
		'''
		
		target = self.get_target(local_rewards, new_local_obs, new_local_actions)
		target = target.detach()

		input_shape = old_local_obs.size()[2]
		old_local_obs = old_local_obs.view(-1, input_shape)
		local_actions = local_actions.view(-1, 1)

		output = self.forward(old_local_obs, local_actions).view(-1, self.num_lights, 1)

		loss = self.loss_function(target, output)
		return loss.mean()





class COMA(nn.Module):
	'''
	Class to implement Independent Actor Critic Algorithm for traffic light
	control. We use parameter sharing for the actor and critic functions
	'''

	def __init__(self, n_rows, n_cols, config):
		'''
		Attributes
		----------
		n_rows : the number of rows in the grid
		n_cols: the number of columns in the grid
		exploration_rate : the rate at which the algorithm explores
		'''
		self.name = "COMA"
		super(COMA, self).__init__()
		self.num_lights = n_rows * n_cols
		self.mean_field = False
		self.coma = True
		self.discount = config.alg.discount
		self.exploration_rate = config.alg.exploration_rate
		self.share_obs = False

		# Initialize actor and critic networks
		self.actor = Actor(n_rows, n_cols, discount=self.discount, lr=config.actor.lr)
		self.critic = Critic(n_rows, n_cols, discount=self.discount, lr=config.critic.lr)

		self.config = config

	def train_step(self, replay, summarize=False):
		'''
		Function to train the model with 100 minibatches of size 30
		Attributes
		----------
		- replay : the current replay buffer
		- summarize : Whether or not to summarize the training results
		'''
		for i in range(self.config.alg.num_minibatches):
			actions, global_reward, old_local_obs, old_global_obs, new_local_obs, new_global_obs, local_rewards = replay.sample(self.config.alg.minibatch_size)
			
			# First train the critic function
			# View the new policy in shape [batch_size * num_lights, action_shape]
			new_pi = self.actor.forward(new_local_obs).view(-1, 2)
			
			# All inputs are of shape [batch_size, num_lights, ...]
			new_actions = torch.multinomial(new_pi, 1).view(-1, self.num_lights, 1)

			# Need to store grads for perturbation loss later
			old_local_obs.requires_grad = True
			critic_loss = self.critic.get_loss(old_local_obs=old_local_obs, 
				local_actions=actions, new_local_obs=new_local_obs, new_local_actions=new_actions, 
				local_rewards=local_rewards)


			# Given that COMA is a policy gradient method, we perturb the critic loss here (to follow Q-Learning methods)
			if self.config.alg.perturb_critic:
				# Get the regularization loss for a smooth policy
				perturbed_tensor = torch.normal(old_global_obs, torch.ones_like(old_global_obs) * 1e-3)
				perturbed_tensor.requires_grad = True
				for i in range(self.config.alg.perturb_num_steps):
					# Gradient of loss wrt the old observation
					
					obs_grad = torch.autograd.grad(outputs=critic_loss, inputs=old_local_obs, grad_outputs=torch.ones_like(critic_loss), retain_graph=True)[0]
					
					# project gradient onto ball
					obs_grad = torch.clamp(input=obs_grad, min=-self.config.alg.perturb_radius,
											max=self.config.alg.perturb_radius)
					perturbed_tensor = perturbed_tensor + self.config.alg.perturb_alpha * obs_grad

				adv_reg_loss = self.get_adv_reg_loss(old_local_obs, perturbed_tensor, actions, self.critic)
				critic_loss = critic_loss + self.config.alg.lam * adv_reg_loss

			self.critic.optimizer.zero_grad()

			critic_loss.backward()
			self.critic.optimizer.step()

			# Calculate the advantage function
			probs = self.actor.forward(old_local_obs)
			advantages = self.critic.get_baseline(old_local_obs, actions, probs)

			actor_loss = self.actor.get_loss(actions, probs, advantages)

			self.actor.optimizer.zero_grad()

			# Given that COMA is a policy gradient method, we can also perturb the actor
			if self.config.alg.perturb_actor:
				if self.config.alg.stackelberg:
					# Stack observations since we employ parameter sharing
					obs_i = old_local_obs.reshape([self.config.env.n_rows * self.config.env.n_cols * self.config.alg.minibatch_size, old_local_obs.shape[-1]])
					perturbed_tensor = obs_i.detach() + torch.normal(torch.zeros_like(obs_i.detach()), torch.ones_like(obs_i.detach()) * 1e-3)
					perturbed_tensor.requires_grad = True
					obs_grad = torch.zeros(perturbed_tensor.shape)

					# Initialize d_delta_d_theta
					d_delta_d_theta = {}
					for i, param in enumerate(self.actor.parameters()):
						d_delta_d_theta[i] = torch.zeros([perturbed_tensor.shape[1]] + list(param.shape))

					for h in range(self.config.alg.perturb_num_steps):
						# Calculate adversarial perurbation
						distance_loss = torch.norm(self.actor(perturbed_tensor) - self.actor(obs_i), p="fro")
						grad = torch.autograd.grad(outputs=distance_loss, inputs=perturbed_tensor, grad_outputs=torch.ones_like(distance_loss), retain_graph=True, create_graph=True)[0]

						# Get Hessian information via finite difference method. First make perturbed observation
						noise = torch.normal(torch.zeros(perturbed_tensor.shape), torch.ones(perturbed_tensor.shape)) * 1e-3
						noise_tensor = perturbed_tensor.detach() + noise
						noise_tensor.requires_grad = True

						# noisy_loss = individual_loss + noisy_global_loss + self.qcombo_lam * noisy_reg_loss
						noisy_loss = torch.norm(self.actor(obs_i) - self.actor(noise_tensor), p="fro")
						grad2 = torch.autograd.grad(outputs=noisy_loss, inputs=noise_tensor, grad_outputs=torch.ones_like(noisy_loss), 
							                        retain_graph=True, create_graph=True, allow_unused=True)[0]

						# Calculate Hessian
						stacked_grad = torch.stack([grad for i in range(grad.shape[1])])
						stacked_grad2 = torch.stack([grad2 for i in range(grad.shape[1])])
						hessian = stacked_grad - stacked_grad2.T
						hessian = hessian.mean(1) / 1e-3

						# Calculate Jacobian
						stackelberg_grads = {}
						for k, param in enumerate(self.actor.parameters()):
							stackelberg_grads[k] = []
						self.actor.optimizer.zero_grad()
						for j in range(grad.shape[1]):
							grad_j = grad[:, j].mean()
							grad_j.backward(retain_graph=True)
							for k, param in enumerate(self.actor.parameters()):
								stackelberg_grads[k].append(param.grad)
						for k, param in enumerate(self.actor.parameters()):
							stackelberg_grads[k] = torch.stack(stackelberg_grads[k])

						obs_grad = obs_grad + grad
						obs_grad = torch.clamp(input=obs_grad, min=-self.config.alg.perturb_radius,
											   max=self.config.alg.perturb_radius)
						perturbed_tensor = perturbed_tensor.detach() + self.config.alg.perturb_alpha * obs_grad * torch.abs(obs_i.detach())

						for k, param in enumerate(self.actor.parameters()):
							d_delta_d_theta[k] = d_delta_d_theta[k] + stackelberg_grads[k] + torch.matmul(d_delta_d_theta[k].T, hessian).T

						# Update gradients
						self.actor.optimizer.zero_grad()

						# First get the smooth loss, and backward the grad
						smooth_loss = torch.norm(self.actor(obs_i) - self.actor(perturbed_tensor), p=2)
						actor_loss = actor_loss + self.config.alg.lam * smooth_loss
						actor_loss.backward(retain_graph=True)
						for k, param in enumerate(self.actor.parameters()):
							# Calculate leader follower interaction
							smooth_partial = torch.autograd.grad(outputs=smooth_loss, inputs = obs_grad, grad_outputs = torch.ones_like(smooth_loss),
							                                     retain_graph=True)[0].mean(0) # obs_grad.shape
							leader_follower = torch.matmul(d_delta_d_theta[k].T, smooth_partial.unsqueeze(1)).squeeze().T * self.config.actor.follower_lr / self.config.actor.lr
							param.grad = param.grad + self.config.alg.lam * leader_follower

						self.actor.optimizer.step()
				else:
					# Get the regularization loss for a smooth policy
					perturbed_tensor = torch.normal(old_local_obs.detach(), torch.ones_like(old_local_obs.detach()) * 1e-3)
					perturbed_tensor.requires_grad = True
					for i in range(self.config.alg.perturb_num_steps):
						# Gradient of loss wrt the old observation
						
						obs_grad = torch.autograd.grad(outputs=actor_loss, inputs=old_local_obs, grad_outputs=torch.ones_like(actor_loss), retain_graph=True)[0]
						
						# project gradient onto ball
						obs_grad = torch.clamp(input=obs_grad, min=-self.config.alg.perturb_radius,
												max=self.config.alg.perturb_radius)
						perturbed_tensor = perturbed_tensor + self.config.alg.perturb_alpha * obs_grad

					adv_reg_loss = self.get_adv_reg_loss(old_local_obs, perturbed_tensor, actions, self.actor)
					actor_loss = actor_loss + self.config.alg.lam * adv_reg_loss

					actor_loss.backward()
					self.actor.optimizer.step()

		if summarize:
			print("Actor Loss: ", actor_loss)
			print("Critic Loss: ", critic_loss)
				

	def get_adv_reg_loss(self, obs, perturbed_obs, actions, network):
		'''
		Function to get the regularization part of the loss function based on adversarial perturbation

		Parameters
		----------
		- obs : the non-perturbed obs tensor [batch_size, num_lights, obs_shape]
		- perturbed_obs : the perturbed obs tensor [batch_size, num_lights, obs_shape]
		- actions : the actions taken by each agent [batch_size, num_lights]
		- network : the network that we are applying the regularizer to [nn.Module]
		Returns
		-------
		- reg_loss : the regularization loss
		'''
		# print(obs.shape, perturbed_obs.shape)
		perturbed_obs = perturbed_obs.detach()
		if network == self.critic:
			obs_shape = obs.size()[2]
			actions = actions.view(-1, 1)
			normal = network(obs.view(-1, obs_shape), actions)
			perturbed = network(perturbed_obs.view(-1, obs_shape), actions)
		elif network == self.actor:
			normal = network(obs)
			perturbed = network(perturbed_obs)
		else:
			raise Exception

		reg_loss = torch.norm(normal - perturbed, p="fro")
		
		return reg_loss

	def choose_action(self, new_state):
		'''
		Function to choose the next action of the agent given a state
		Attributes
		----------
		- new_state : the state the agent is currently in [num_obs]
		Returns
		-------
		- decision : int in {0, 1}
		'''
		if np.random.uniform() < self.exploration_rate:
			decision = np.random.randint(0, 2)
			self.exploration_rate *= self.config.alg.anneal_exp
		else:
			pi = self.actor.forward(new_state)
			if np.random.uniform() < pi[0]:
				decision = 0
			else:
				decision = 1
		
		return decision
	
	def save(self, dir, model_id=None):
		torch.save(self.state_dict(), os.path.join(dir, 'COMA_{}.pt'.format(model_id)))

	def load(self, dir, model_id=None):
		self.load_state_dict(torch.load(os.path.join(dir, 'COMA_{}.pt'.format(model_id))))
