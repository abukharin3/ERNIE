# ERNIE: A Robust MARL Algorithm
Repository for the paper Robust Multi-Agent Reinforcement Learning via Adversarial Regularization: Theoretical Foundation and Stable Algorithms, accepted to NeurIPS 2023.

The simplest version of ERNIE can be implemented as follows:
```python
perturbed_tensor = old_global_obs + torch.normal(torch.zeros_like(old_global_obs), torch.ones_like(old_global_obs) * 1e-3)
perturbed_tensor.requires_grad = True

for k in range(self.config.alg.perturb_num_steps):
	# Calculate adversarial perurbation
	distance_loss = torch.norm(self.global_net(old_global_obs) - self.global_net(perturbed_tensor), p="fro")
	grad = torch.autograd.grad(outputs=distance_loss, inputs=perturbed_tensor, grad_outputs=torch.ones_like(loss), retain_graph=True, create_graph=True)[0]
	perturbed_tensor = perturbed_tensor + self.config.alg.perturb_alpha * grad * torch.abs(old_global_obs.detach())

adv_reg_loss = torch.norm(old_global_obs, perturbed_tensor)
```
This loss can simply be added to your algorithm's training loss. Note that here ERNIE is applied to the global policy.

To train policies in the traffic light control environment, first install follow the installation instructions at https://flow-project.github.io. Then run the command
```
python train_policy.py
```
