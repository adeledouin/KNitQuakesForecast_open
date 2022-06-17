"""An example of training DQN against OpenAI Gym Envs.
This script is an example of training a DQN agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported. For continuous action
spaces, A NAF (Normalized Advantage Function) is used to approximate Q-values.
To solve CartPole-v0, run:
    python train_dqn_gym.py --env CartPole-v0
To solve Pendulum-v0, run:
    python train_dqn_gym.py --env Pendulum-v0
"""

import argparse
import os
import sys
import pandas as pd

import gym
import numpy as np
import torch.optim as optim
from gym import spaces

import torch
import gym_KnitCity
import gym_KnitCity.envs
from gym_KnitCity.envs.env_KnitCity_raw import KnitLab as KnitLabRaw

import pfrl
from pfrl import experiments, explorers
from pfrl import nn as pnn
from pfrl import q_functions, replay_buffers, utils
from pfrl.agents.dqn import DQN

import logging

logging.basicConfig(format='| %(levelname)s | %(asctime)s | %(message)s', level=logging.INFO)

from classConfig import ConfigData, ConfigPred
import Config_data
import Config_pred
from classPlot import ClassPlot
from Datas.classStat import Histo, Stat
from Config_env import def_damage


parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default="results",
                    help=("Directory path to save output files.If it does not exist, it will be created."))
parser.add_argument("p", type=str, help="data path_from_root")
parser.add_argument("--env", type=str, default="01")
parser.add_argument("--env_transfert", type=str, default="01")
parser.add_argument("--raw", action="store_true")
parser.add_argument("--tau_ref", type=int, default=4)
parser.add_argument("--results", type=str, default="_suppaper_5023")
parser.add_argument("--sub_set", type=str, default="d_10_100")
parser.add_argument("--model", type=str, default="1_2_1")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--max-nb-episode", type=int, default=1)
parser.add_argument("--max-nb-eval", type=int, default=190)

parser.add_argument("--replay-start-size", type=int, default=25000)
parser.add_argument("--target-update-interval", type=int, default=10000)

parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--n-hidden-channels", type=int, default=64) #16)
parser.add_argument("--n-hidden-layers", type=int, default=2) #3)
parser.add_argument("--update-interval", type=int, default=1)

parser.add_argument("--seed", type=int, default=1, help="Random seed [0, 2 ** 32)")
parser.add_argument("--final-exploration-steps", type=int, default=10 ** 4)
parser.add_argument("--start-epsilon", type=float, default=1.0)
parser.add_argument("--end-epsilon", type=float, default=0.1)
parser.add_argument("--noisy-net-sigma", type=float, default=None)
parser.add_argument("--demo", action="store_true", default=False)
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--prioritized-replay", action="store_true")
parser.add_argument("--target-update-method", type=str, default="hard")
parser.add_argument("--soft-update-tau", type=float, default=1e-2)
parser.add_argument("--minibatch-size", type=int, default=None)
parser.add_argument("--render-train", action="store_true")
parser.add_argument("--render-eval", action="store_true")
parser.add_argument("--monitor", action="store_true")
parser.add_argument("--reward-scale-factor", type=float, default=1.0)
parser.add_argument("--actor-learner", action="store_true",
                    help="Enable asynchronous sampling with asynchronous actor(s)")  # NOQA
parser.add_argument("--num-envs", type=int, default=1,
                    help=("The number of environments for sampling (only effective with --actor-learner enabled)"))  # NOQA

args = parser.parse_args()

results_type = pd.Series(args.results)
if results_type.str.contains('newReward').all():
    from gym_KnitCity.envs.env_KnitCity import KnitCity_newReward as KnitCity, KnitLab
else:
    from gym_KnitCity.envs.env_KnitCity import KnitCity, KnitLab
if results_type.str.contains('5023').all():
    from Config_models_5023 import model
# elif results_type.str.contains('1000').all():
#     from Config_models_1000 import model
# elif results_type.str.contains('2004').all():
#     from Config_models_2004 import model
# else:
#     from Config_models_3009 import model

## args supp
remote = True
ref_tricot = 'knit005_'
n_exp = 'mix_'
version_data = 'v1'

NAME_EXP = ref_tricot + n_exp + version_data
config_data = ConfigData(args.path_from_root, Config_data.exp[NAME_EXP])

global_path_load = '/{}/KnitQuakesForecast/'.format(config_data.path_from_root) + config_data.ref + '/input/' \
                                                                               'version%d/' % (
                            config_data.version_work)
global_path_save = '/{}/KnitQuakesForecast/'.format(config_data.path_from_root) + config_data.ref + '/output/' \
                                                                               'version%d/' % (
                            config_data.version_work)

histo = Histo(config_data)
plot = ClassPlot(remote, histo)

m = model[args.model]
tau_ref = args.tau_ref
damage = def_damage[args.sub_set]

if not args.raw:
    ENV_NAME = 'env-KnitCity-v{}'.format(args.env)
    ENV_NAME_EVAL = 'env-KnitCity-comp-v{}'.format(args.env)
else:
    ENV_NAME = 'env-KnitCityRaw-v{}'.format(args.env)
    ENV_NAME_EVAL = 'env-KnitCityRaw-comp-v{}'.format(args.env)

print('on est en version {}'.format(m['version_pred']))


# Set a random seed used in PFRL
utils.set_random_seed(args.seed)
process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
assert process_seeds.max() < 2 ** 32

def clip_action_filter(a):
    return np.clip(a, action_space.low, action_space.high)

def make_env(idx=0, test=False):
    city = KnitCity(nb_learning_classe=m['nb_learning_classe'], l=1, sub_version=m['sub_version'], d=damage)
    simplet = KnitCity(nb_learning_classe=m['nb_learning_classe'], l=1, sub_version=m['sub_version'],
                       d=damage)
    delphes = KnitCity(nb_learning_classe=5, l=1, sub_version=m['sub_version'], d=damage)

    if not test:
        if not args.raw:
            lab = KnitLab(date=m['date'], version_pred=m['version_pred'], model=m['model'], trainsize=1000000,
                          sub_version=m['sub_version'],
                          nb_step=5000, nb_episode=300, tau_ref=tau_ref)
        else:
            lab = KnitLabRaw(date=m['date'], version_pred=m['version_pred'], model=m['model'], trainsize=1000000,
                             sub_version=m['sub_version'],
                             nb_step=5000, nb_episode=300, tau_ref=tau_ref)

        env = gym.make(ENV_NAME, city=city, lab=lab, simplet=simplet, delphes=delphes)
    else:
        if not args.raw:
            eval_lab = KnitLab(date=m['date'], version_pred=m['version_pred'], model=m['model'], trainsize=1000000,
                               sub_version=m['sub_version'],
                               nb_step=5000, nb_episode=300, tau_ref=tau_ref)
        else:
            eval_lab = KnitLabRaw(date=m['date'], version_pred=m['version_pred'], model=m['model'], trainsize=1000000,
                                  sub_version=m['sub_version'],
                                  nb_step=5000, nb_episode=300, tau_ref=tau_ref)

        env = gym.make(ENV_NAME_EVAL, city=city, lab=eval_lab, simplet=simplet, delphes=delphes)
    # Use different random seeds for train and test envs
    process_seed = int(process_seeds[idx])
    env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
    utils.set_random_seed(env_seed)
    env = pfrl.wrappers.CastObservationToFloat32(env)
    if args.monitor:
        env = pfrl.wrappers.Monitor(env, args.outdir)
    if isinstance(env.action_space, spaces.Box):
        utils.env_modifiers.make_action_filtered(env, clip_action_filter)
    if not test:
        # Scale rewards (and thus returns) to a reasonable range so that
        # training is easier
        env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)

    return env


env = make_env(test=False)
eval_env = make_env(test=True)

args.outdir = global_path_save + "Knit_City/results{}/{}/{}/env_v{}_reftau{}".format(
    args.results, args.sub_set, env.lab.version_pred if not args.raw else '{}_raw'.format(env.lab.version_pred), args.env, args.tau_ref)
print("Output files are saved in {}".format(args.outdir))

args.load = global_path_save + "Knit_City/results{}/{}/{}/env_v{}/best/".format(
    args.results, args.sub_set, env.lab.version_pred if not args.raw else '{}_raw'.format(env.lab.version_pred), args.env_transfert)


timestep_limit = env.spec.max_episode_steps
obs_space = env.observation_space
obs_size = obs_space.low.size
action_space = env.action_space

args.steps = env.lab.indexes.shape[
                 0] * env.lab.nb_step if args.max_nb_episode is None else np.min(np.array([args.max_nb_episode * env.lab.nb_step,
                                                                                 env.lab.starts_episodes.shape[
                                                                                     0] * env.lab.nb_step]))
args.eval_interval = 1 #10 * env.lab.nb_step
args.eval_n_runs = eval_env.lab.starts_episodes.shape[0] if args.max_nb_eval is None else np.min(np.array([args.max_nb_eval,
                                                                                 eval_env.lab.starts_episodes.shape[0]]))

print(
    'training for {} episodes of {} steps - {} steps tot'.format(
        env.lab.starts_episodes.shape[0] if args.max_nb_episode is None else np.min(np.array([args.max_nb_episode,
                                                                                 env.lab.starts_episodes.shape[
                                                                                     0]])), env.lab.nb_step, args.steps))
print('and eval on {} episodes'.format(args.eval_n_runs))

n_atoms = 51
v_max = env.city.class_risk[-1]
v_min = -env.city.class_risk[-1]

n_actions = action_space.n
q_func = q_functions.DistributionalFCStateQFunctionWithDiscreteAction(
    obs_size,
    n_actions,
    n_atoms,
    v_min,
    v_max,
    n_hidden_channels=args.n_hidden_channels,
    n_hidden_layers=args.n_hidden_layers,
)
# Use epsilon-greedy for exploration
explorer = explorers.LinearDecayEpsilonGreedy(
    args.start_epsilon,
    args.end_epsilon,
    args.final_exploration_steps,
    action_space.sample,
)

opt = torch.optim.Adam(q_func.parameters(), 1e-3)

# rbuf_capacity = 50000  # 5 * 10 ** 5
if args.minibatch_size is None:
    args.minibatch_size = 32
rbuf = replay_buffers.ReplayBuffer(10 ** 6)

agent = pfrl.agents.CategoricalDQN(
    q_func,
    opt,
    rbuf,
    gpu=args.gpu,
    gamma=args.gamma,
    explorer=explorer,
    replay_start_size=args.replay_start_size,
    target_update_interval=args.target_update_interval,
    update_interval=args.update_interval,
    minibatch_size=args.minibatch_size,
    target_update_method=args.target_update_method,
    soft_update_tau=args.soft_update_tau,
)

if args.load:
    print('load from {}'.format(args.load))
    agent.load(args.load)


agent, eval_stats_history, train_info_supp_history, eval_info_supp_history = experiments.train_agent_with_evaluation(
    agent=agent,
    env=env,
    steps=args.steps,
    eval_n_steps=None,
    eval_n_episodes=args.eval_n_runs,
    eval_interval=args.eval_interval,
    outdir=args.outdir,
    eval_env=eval_env,
    train_max_episode_len=timestep_limit,
)

np.save(args.outdir + '/stats.npy', eval_stats_history)
np.save(args.outdir + '/train_info_supp.npy', train_info_supp_history)
np.save(args.outdir + '/info_supp.npy', eval_info_supp_history)

### ploooooots
color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9']
symbole = ['0', '1', '2', '3', '4']
import matplotlib.patches as mpatches

stats_labels = ['steps', 'episodes', 'elapsed',
                'mean', 'median', 'stev', 'max', 'min', 'average q', 'average loss', 'cumulative steps', 'n updates',
                'rlen']
nb_test = np.shape(eval_stats_history)[0]

fig, ax = plot.belleFigure('{}'.format(stats_labels[0]), 'mean R with $\Delta R$', nfigure=None)
for i in range(nb_test):
    ax.errorbar(eval_stats_history[i][0], eval_stats_history[i][3],
                yerr=np.array([[eval_stats_history[i][3] - eval_stats_history[i][7]],
                               [eval_stats_history[i][6] - eval_stats_history[i][3]]]),
                fmt='.', ecolor=color[0], label='class {}'.format(i))
save = args.outdir + '/mean_R_deltaR'
plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

fig, ax = plot.belleFigure('{}'.format(stats_labels[0]), '{} R'.format(stats_labels[3]), nfigure=None)
for i in range(nb_test):
    ax.plot(eval_stats_history[i][0], eval_stats_history[i][3],
            '.', color=color[0])
save = args.outdir + '/mean_R'
plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

R_test = np.zeros((nb_test, eval_env.lab.starts_episodes.shape[0]))
R_simplet = np.zeros((nb_test, eval_env.lab.starts_episodes.shape[0]))
R_delphes = np.zeros((nb_test, eval_env.lab.starts_episodes.shape[0]))
out_days_test = np.zeros((nb_test, eval_env.lab.starts_episodes.shape[0], 5))
out_days_simplet = np.zeros((nb_test, eval_env.lab.starts_episodes.shape[0], 5))
out_days_delphes = np.zeros((nb_test, eval_env.lab.starts_episodes.shape[0], 5))
for i in range(nb_test):
    for j in range(eval_env.lab.starts_episodes.shape[0]):
        R_test[i, j] = eval_info_supp_history[i]['info_cities'][j][0]['cost']
        R_simplet[i, j] = eval_info_supp_history[i]['info_cities'][j][1]['cost']
        R_delphes[i, j] = eval_info_supp_history[i]['info_cities'][j][2]['cost']
        out_days_test[i, j, :] = eval_info_supp_history[i]['info_cities'][j][0]['out_events']
        out_days_simplet[i, j, :] = eval_info_supp_history[i]['info_cities'][j][1]['out_events']
        out_days_delphes[i, j, :] = eval_info_supp_history[i]['info_cities'][j][2]['out_events']

# print(out_days_test[1, :, :])
# print(out_days_simplet[1, 1, :])
# print(out_days_delphes[1, 1, :])
legend = ['test', 'simplet', 'delphes']
legend_properties = {'weight': 'normal'}

fig, ax = plot.belleFigure('{}'.format(stats_labels[0]), '${}$'.format('R'), nfigure=None)
for i in range(nb_test):
    ax.plot(eval_stats_history[i][0], np.mean(R_delphes[i, :]),
            '.', color=color[2])
    ax.plot(eval_stats_history[i][0], np.mean(R_simplet[i, :]),
            '.', color=color[1])
    ax.plot(eval_stats_history[i][0], np.mean(R_test[i, :]),
            '.', color=color[0])
patchs = [0 for _ in range(0, 3)]
for i in range(0, 3):
    patchs[i] = mpatches.Patch(color=color[i], label='${} set$'.format(legend[i]))
plot.plt.legend(prop=legend_properties, handles=patchs)
save = args.outdir + '/mean_R_allcities'
plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

for wich_decade in range(1, 6):
    fig, ax = plot.belleFigure('{}'.format(stats_labels[0]), 'days out supp for decade {}'.format(wich_decade), nfigure=None)
    for i in range(nb_test):
        ax.plot(eval_stats_history[i][0],
                np.mean(out_days_simplet[i, :, wich_decade - 1] - out_days_delphes[i, :, wich_decade - 1]),
                '.', color=color[1])
        ax.plot(eval_stats_history[i][0],
                np.mean(out_days_test[i, :, wich_decade - 1] - out_days_delphes[i, :, wich_decade - 1]),
                '.', color=color[0])
    patchs = [0 for _ in range(0, 2)]
    for i in range(0, 2):
        patchs[i] = mpatches.Patch(color=color[i], label='${} set$'.format(legend[i]))
    plot.plt.legend(prop=legend_properties, handles=patchs)
    save = args.outdir + '/out_days_decade{}'.format(wich_decade)
    plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)
