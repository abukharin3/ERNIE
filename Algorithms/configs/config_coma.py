from .configdict import ConfigDict

def get_config():
    
    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.discount = .95
    config.alg.exploration_rate = .05
    config.alg.anneal_exp = .99
    config.alg.lam = 1e-1
    config.alg.perturb_alpha = 1e-1
    config.alg.perturb_num_steps = 1
    config.alg.perturb_radius = 1
    config.alg.num_minibatches = 100
    config.alg.minibatch_size = 30
    config.alg.perturb_critic = False
    config.alg.perturb_actor = True
    config.alg.perturb = True
    config.alg.stackelberg = True

    config.actor = ConfigDict()
    config.actor.lr = 5e-5
    config.actor.follower_lr = 5e-5

    config.critic = ConfigDict()
    config.critic.lr = 5e-4
    config.critic.size = [256, 256]
    
    config.main = ConfigDict()
    config.main.dir = 'results/COMA_2x2_perturb_actor'
    config.main.train_iters = 3000
    config.main.eval_iters = 5000
    config.main.log_period = 1 # Log every _ steps
    # config.main.eval_period = 400 # Eval every _ steps
    config.main.update_period = 50 # Run an update every _ steps
    config.main.replay_capacity = 30000
    config.main.collect_eval_reward = 200
    
    config.env = ConfigDict()
    config.env.n_rows = 2
    config.env.n_cols = 2
    config.env.train_parameters = [700] * 8
    config.env.eval_parameters = [[20, 500, 20, 500, 600,600,600,600], [600] * 8, [800] * 8, [700 * 4, 700 * 4, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10],
                                    [700 * 8, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10]]

    return config