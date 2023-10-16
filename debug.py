import numpy as np
import matplotlib.pyplot as plt

def ma(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

fig, ax = plt.subplots()

############################################################
#
#                   Plot Training Reward
#
############################################################

reward = []
for s in range(5):
	x = np.load("results/QCOMBO{}/training_reward_4.npy".format(s))
	reward.append(x)

reward = np.array(reward)
avg = ma(reward.mean(0), 50)
std = ma(reward.std(0), 50)

length = len(avg)
ax.fill_between(np.arange(length), avg - 1.96 * std, avg + 1.96 * std, alpha=0.3)
plt.plot(ma(reward.mean(0), 50), label = "QCOMBO")


for (lam, alpha) in [(1e-1, 1e-2)]:#, (5e-1, 1e-2)]:#, (1e-1, 1e-1)]:
	reward = []
	for s in range(5):
		x = np.load("results/QCOMBO_adv{}{}{}/training_reward_4.npy".format(0, lam, alpha))
		reward.append(x)
	reward = np.array(reward)
	avg = ma(reward.mean(0), 50)
	std = ma(reward.std(0), 50)

	length = len(avg)
	ax.fill_between(np.arange(length), avg - 1.96 * std, avg + 1.96 * std, alpha=0.3)
	plt.plot(ma(reward.mean(0), 50), label="QCOMBO_adv{}{}".format(lam, alpha))


for (lam, alpha) in [(1e-1, 1e-2)]:# (1e-1, 1e-1)]:
	reward = []
	for s in range(5):
		x = np.load("results/QCOMBO_adv{}{}{}{}/training_reward_4.npy".format(0, lam, alpha, True))
		reward.append(x)
	reward = np.array(reward)
	avg = ma(reward.mean(0), 50)
	std = ma(reward.std(0), 50)

	length = len(avg)
	ax.fill_between(np.arange(length), avg - 1.96 * std, avg + 1.96 * std, alpha=0.3)
	plt.plot(ma(reward.mean(0), 50), label="QCOMBO_Stackelberg{}{}".format(lam, alpha))

plt.legend()
plt.title("Training Reward")
plt.show()

############################################################
#
#                   Plot Eval Reward
#
############################################################

reward = []
for s in range(5):
	x = np.load("results/QCOMBO{}/eval_reward_{}_{}_{}.npy".format(s, True, 1e-1, 35))
	reward.append(x)

reward = np.array(reward)
avg = ma(reward.mean(0), 50)
std = ma(reward.std(0), 50) / 5

length = len(avg)
ax.fill_between(np.arange(length), avg - 1.96 * std, avg + 1.96 * std, alpha=0.3)
plt.plot(ma(reward.mean(0), 50), label = "QCOMBO")


for (lam, alpha) in [(1e-1, 1e-2)]:#, (5e-1, 1e-2), (1e-1, 1e-1)]:
	reward = []
	for s in range(5):
		x = np.load("results/QCOMBO_adv{}{}{}/eval_reward_{}_{}_{}.npy".format(s, lam, alpha, True, 1e-1, 35))
		reward.append(x)
	reward = np.array(reward)
	avg = ma(reward.mean(0), 50)
	std = ma(reward.std(0), 50) / 5

	length = len(avg)
	ax.fill_between(np.arange(length), avg - 1.96 * std, avg + 1.96 * std, alpha=0.3)
	plt.plot(ma(reward.mean(0), 50), label="QCOMBO_adv{}{}".format(lam, alpha))

for (lam, alpha) in [(1e-1, 1e-2)]:#, (1e-1, 1e-1)]:
	reward = []
	for s in range(5):
		x = np.load("results/QCOMBO_adv{}{}{}{}/eval_reward_{}_{}_{}_{}.npy".format(s, lam, alpha, True, True, 1e-1, 35, 4))
		reward.append(x)
	reward = np.array(reward)
	avg = ma(reward.mean(0), 50)
	std = ma(reward.std(0), 50) / 5

	length = len(avg)
	ax.fill_between(np.arange(length), avg - 1.96 * std, avg + 1.96 * std, alpha=0.3)
	plt.plot(ma(reward.mean(0), 50), label="QCOMBO_stack{}{}".format(lam, alpha))

plt.legend()
plt.title("Eval")
plt.show()


# ############################################################
# #
# #                   Plot Eval Reward
# #
# ############################################################

# reward = []
# for s in range(5):
# 	x = np.load("results/QCOMBO{}/eval_reward_{}_{}_{}.npy".format(s, False, 0, 40))
# 	reward.append(x)
# reward = np.array(reward)[:, 1000:]
# avg = ma(reward.mean(0), 50)
# std = ma(reward.std(0), 50)

# length = len(avg)
# #ax.fill_between(np.arange(length), avg - 1.96 * std, avg + 1.96 * std, alpha=0.3)
# plt.plot(ma(reward.mean(0), 50), label = "QCOMBO")


# for (lam, alpha) in [(5e-1, 1e-2), (1e-1, 1e-2), (1e-1, 1e-1)]:
# 	reward = []
# 	for s in range(5):
# 		x = np.load("results/QCOMBO_adv{}{}{}/eval_reward_{}_{}_{}.npy".format(s, lam, alpha, False, 0, 40))
# 		reward.append(x)
# 	reward = np.array(reward)[:, 1000:]
# 	avg = ma(reward.mean(0), 50)
# 	std = ma(reward.std(0), 50)

# 	length = len(avg)
# 	#ax.fill_between(np.arange(length), avg - 1.96 * std, avg + 1.96 * std, alpha=0.3)
# 	plt.plot(ma(reward.mean(0), 50), label="QCOMBO_adv{}{}".format(lam, alpha))

# plt.legend()
# plt.show()