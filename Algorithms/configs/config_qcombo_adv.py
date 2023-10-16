from .configdict import ConfigDict

def get_config():
    
    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.discount = .95
    config.alg.exploration_rate = .05
    config.alg.anneal_exp = .99
    config.alg.lam = 0.5
    config.alg.qcombo_lam = 1.7
    config.alg.perturb_alpha = 0.001
    config.alg.perturb_num_steps = 1
    config.alg.perturb_radius = 1
    config.alg.num_minibatches = 100
    config.alg.minibatch_size = 30
    config.alg.perturb = True
    config.alg.stackelberg = False
    config.alg.normal_perturb = True
    config.alg.stackelberg_update = 1
    # config.alg.target_update_period = 50 # Update target network every 1000 steps

    config.critic = ConfigDict()
    config.critic.lr = 1e-4
    config.critic.follower_lr = 1e-4
    config.critic.size = [256, 256]
    
    config.main = ConfigDict()
    config.main.dir = 'results/QCOMBOtune_test1'
    config.main.name = 'adv'
    config.main.train_iters = 3000
    config.main.eval_iters = 5000
    config.main.log_period = 1 # Log every _ steps
    config.main.eval_period = 400 # Eval every _ steps
    config.main.update_period = 30 # Run an update every _ steps
    config.main.replay_capacity = 30000
    config.main.collect_eval_reward = 200
    
    config.env = ConfigDict()
    config.env.n_rows = 2
    config.env.n_cols = 2
    config.env.train_parameters = [700] * 8# Use for 2x2
    # config.env.train_parameters = [700] * 9# Use for 3x2
    # config.env.train_parameters = [700] * 12# Use for 3x3

    config.env.eval_parameters = [[20, 500, 20, 500, 600,900,600,900]]#, [600] * 8] #, [700 * 4, 700 * 4, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10],
                                   #     [700 * 8, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10]]

    return config