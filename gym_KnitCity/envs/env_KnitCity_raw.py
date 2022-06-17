import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random
from Datas.classSignal import SignalForce
from Datas.classEvent import ForceEvent
from classConfig import ConfigData, ConfigPred
import Config_data
import Config_pred
from dictdata import dictdata

import logging
logging.basicConfig(format='| %(levelname)s | %(asctime)s | %(message)s', level=logging.INFO)
import time

# Tom:

# from gym_KnitCity.envs import Function

def which_classe(y, nb_learning_classe=5):
    if nb_learning_classe == 2:
        classe = np.array([0, 1, 1, 1, 1])
    elif nb_learning_classe == 3:
        classe = np.array([0, 1, 1, 2, 2])
    elif nb_learning_classe == 4:
        classe = np.array([0, 1, 2, 3, 3])
    else:
        classe = np.array([0, 1, 2, 3, 4])

    return classe[y]


def which_decade(y, decade):
    for i in range(decade.size - 1):
        if i == decade.size - 2:
            if (y >= decade[i]) and (y <= decade[i + 1]):
                y_decade = i
                return y_decade
        else:
            if (y >= decade[i]) and (y < decade[i + 1]):
                y_decade = i
                return y_decade


def decade_to_onehot(decade):
    obs = np.zeros(5)
    obs[decade] = 1
    return obs.astype(int)


def onehot_to_decade(onehot):
    decade = np.where(onehot == 1)[0]
    # print(onehot, decade)
    return decade


class KnitCity():
    def __init__(self, nb_learning_classe, l, d, sub_version, hypothese_events='geometric_risk'):
        self.nb_learning_classe = nb_learning_classe
        self.l = l
        self.hypothese_events = hypothese_events

        reverse_classes = True if sub_version == '_reverse' else False
        self.d_rate(d, reverse_classes)
        self.decade = np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1])

        self.t = None
        self.death_per_decade = None
        self.out_events = None
        self.past_events = None
        self.life_cost = None
        self.days_out = None
        self.cost = None
        self.running_life_cost = None
        self.running_days_out = None
        self.running_cost = None
        self.where = None

    def d_rate(self, d, reverse):
        self.death_rate = d
        if self.nb_learning_classe == 2:
            if reverse:
                self.class_risk = np.array([d[3], d[4]])
            else:
                self.class_risk = np.array([d[0], d[4]])
        elif self.nb_learning_classe == 3:
            if reverse:
                self.class_risk = np.array([d[2], d[3], d[4]])
            else:
                self.class_risk = np.array([d[0], d[2], d[4]])
        elif self.nb_learning_classe == 4:
            if reverse:
                self.class_risk = np.array([d[0], d[2], d[3], d[4]])
            else:
                self.class_risk = np.array([d[0], d[1], d[2], d[4]])
        else:
            self.class_risk = self.death_rate

    def degat(self, df):
        return 1000

    def initialise(self):
        self.t = 0
        self.death_per_decade = np.array([0, 0, 0, 0, 0])
        self.past_events = np.array([0, 0, 0, 0, 0])
        self.out_events = np.array([0, 0, 0, 0, 0])
        self.life_cost = 0
        self.days_out = 0
        self.cost = 0
        self.running_life_cost = 0
        self.running_days_out = 0
        self.running_cost = 0
        self.where = 'in'

    def update_t(self):
        self.t = self.t + 1

    def update_nb_death(self, df_now):
        nb_death = self.death_rate[which_decade(df_now, self.decade)]
        if self.hypothese_events == 'risk_on_last':
            nb_death = self.degat(df_now)
        if self.where == 'in':
            self.death_per_decade[which_decade(df_now, self.decade)] = self.death_per_decade[
                                                                           which_decade(df_now, self.decade)] + nb_death
            self.life_cost = - nb_death
            self.running_life_cost = self.running_life_cost - nb_death
        else:
            self.life_cost = nb_death
            self.out_events[which_decade(df_now, self.decade)] = self.out_events[
                                                                     which_decade(df_now, self.decade)] + 1
            self.running_life_cost = self.running_life_cost + nb_death

        self.past_events[which_decade(df_now, self.decade)] = self.past_events[
                                                                  which_decade(df_now, self.decade)] + 1

    def update_days_out(self):
        if self.where == 'out':
            self.days_out = 1
            self.running_days_out = self.running_days_out + 1
        else:
            self.days_out = 0

    def update_cost(self):
        self.cost = self.life_cost - self.days_out * self.l
        self.running_cost = self.running_life_cost - self.running_days_out * self.l

    def update_where(self, action):
        if action == 0:
            self.where = 'in'
        else:
            self.where = 'out'

    def update_city(self, df_now):
        self.update_nb_death(df_now)
        self.update_days_out()
        self.update_cost()

class KnitCity_test():
    def __init__(self, nb_learning_classe, l, d, sub_version, hypothese_events='geometric_risk'):
        self.nb_learning_classe = nb_learning_classe
        self.l = l
        self.hypothese_events = hypothese_events

        reverse_classes = True if sub_version == '_reverse' else False
        self.d_rate(d, reverse_classes)
        self.decade = np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 1e1])

        self.t = None
        self.death_per_decade = None
        self.out_events = None
        self.past_events = None
        self.life_cost = None
        self.days_out = None
        self.cost = None
        self.running_life_cost = None
        self.running_days_out = None
        self.running_cost = None
        self.where = None

    def d_rate(self, d, reverse):
        self.death_rate = d
        if self.nb_learning_classe == 2:
            if reverse:
                self.class_risk = np.array([d[3], d[4]])
            else:
                self.class_risk = np.array([d[0], d[4]])
        elif self.nb_learning_classe == 3:
            if reverse:
                self.class_risk = np.array([d[2], d[3], d[4]])
            else:
                self.class_risk = np.array([d[0], d[2], d[4]])
        elif self.nb_learning_classe == 4:
            if reverse:
                self.class_risk = np.array([d[0], d[2], d[3], d[4]])
            else:
                self.class_risk = np.array([d[0], d[1], d[2], d[4]])
        else:
            self.class_risk = self.death_rate

    def degat(self, df):
        return 1000

    def initialise(self):
        self.t = 0
        self.death_per_decade = np.array([0, 0, 0, 0, 0])
        self.past_events = np.array([0, 0, 0, 0, 0])
        self.out_events = np.array([0, 0, 0, 0, 0])
        self.life_cost = 0
        self.days_out = 0
        self.cost = 0
        self.running_life_cost = 0
        self.running_days_out = 0
        self.running_cost = 0
        self.where = 'in'

    def update_t(self):
        self.t = self.t + 1

    def update_nb_death(self, df_now):
        nb_death = self.death_rate[which_decade(df_now, self.decade)]
        if self.hypothese_events == 'risk_on_last':
            nb_death = self.degat(df_now)
        if self.where == 'in':
            self.death_per_decade[which_decade(df_now, self.decade)] = self.death_per_decade[
                                                                           which_decade(df_now, self.decade)] + nb_death
            self.life_cost = - nb_death
            self.running_life_cost = self.running_life_cost - nb_death
        else:
            self.life_cost = 0 #nb_death
            self.out_events[which_decade(df_now, self.decade)] = self.out_events[
                                                                     which_decade(df_now, self.decade)] + 1
            self.running_life_cost = self.running_life_cost #+ nb_death

        self.past_events[which_decade(df_now, self.decade)] = self.past_events[
                                                                  which_decade(df_now, self.decade)] + 1

    def update_days_out(self):
        if self.where == 'out':
            self.days_out = 1
            self.running_days_out = self.running_days_out + 1
        else:
            self.days_out = 0

    def update_cost(self):
        self.cost = self.life_cost - self.days_out * self.l
        self.running_cost = self.running_life_cost - self.running_days_out * self.l

    def update_where(self, action):
        if action == 0:
            self.where = 'in'
        else:
            self.where = 'out'

    def update_city(self, df_now):
        self.update_nb_death(df_now)
        self.update_days_out()
        self.update_cost()

class KnitLab():
    def __init__(self, date, version_pred, model, trainsize, sub_version,
                 nb_step, nb_episode, tau_ref):

        self.config_data = ConfigData(Config_data.exp['knit005_mix_v2'])
        NAME_EXP = 'knit005_' + version_pred
        self.config_pred = ConfigPred(Config_pred.exp_scalar[NAME_EXP], self.config_data)

        self.date = date
        self.version_pred = version_pred
        self.model = model
        self.trainsize = trainsize
        self.nb_episode = nb_episode
        self.tau_ref = tau_ref
        self.nbclasses = self.config_pred.output_shape
        self.nb_step = nb_step
        self.seq_size = self.config_pred.seq_size
        self.sub_version = sub_version

    def set_lab_args(self, nb_eval, info_size, model_set, RL_set):

        self.nb_eval = nb_eval
        self.info_size = info_size
        self.model_set = model_set
        self.RL_set = RL_set

        self.signal_flu = SignalForce(self.config_data, 'flu_rsc', self.model_set)
        signalevent = ForceEvent(self.config_data, self.signal_flu.f, self.signal_flu.ext, self.signal_flu.t,
                                 'flu_rsc', self.model_set, Sm=False)

        if self.sub_version == '_reverse':
            fileName = self.config_pred.global_path_load + 'sequences_RL/' + '{}_reversed2_classes_edges_{}_futur_{}.npy'.format(
                self.config_pred.output_shape, self.config_pred.label_save,
                self.config_pred.futur)
        else:
            fileName = self.config_pred.global_path_load + 'sequences_RL/' + '{}_classes_edges_{}_futur_{}.npy'.format(
                self.config_pred.output_shape, self.config_pred.label_save,
                self.config_pred.futur)

        logging.info("load from {}".format(fileName))
        self.classes_edges = np.load(fileName)

        fileName = self.config_pred.global_path_load + 'sequences_RL/' + 'dict_sequence_reftau{}_{}_{}seqsize_{}step_{}futur.npy'.format(
            self.tau_ref, self.config_pred.input_data, self.config_pred.seq_size, self.config_pred.overlap_step, self.config_pred.futur)

        logging.info("load from {}".format(fileName))
        self.dict_sequence = np.load(fileName, allow_pickle=True).flat[0]

        if self.model_set == 'train' and self.RL_set == 'train':
            fileName = self.config_pred.global_path_load + 'sequences_RL/' + 'dict_all_value_reftau{}_for_class_{}_futur_{}.npy'.format(
                self.tau_ref, self.config_pred.label_save, self.config_pred.futur)

            logging.info("load from {}".format(fileName))
            dict_all_values = np.load(fileName, allow_pickle=True).flat[0]
            self.y_value = dict_all_values['train']
            self.y_target = None
            self.y_pred = None
            logging.info("load from {}".format(
                self.config_pred.global_path_load + 'sequences_RL/indexes_trainsetreftau{}_{}ep_{}steps.npy'.format(self.tau_ref,
                                                                                                               self.nb_episode,
                                                                                                               self.nb_step)))
            self.starts_episodes = np.load(
                self.config_pred.global_path_load + 'sequences_RL/indexes_trainsetreftau{}_{}ep_{}steps.npy'.format(self.tau_ref,
                                                                                                               self.nb_episode,
                                                                                                               self.nb_step))

        else:
            load_test_callback = self.config_pred.global_path_load + 'sequences_RL/test_callback_reftau{}_{}_{}_m{}_{}seq{}.npy'.format(
                self.tau_ref, self.date, self.version_pred, self.model, self.trainsize, self.sub_version)
            logging.info("load from {}".format(load_test_callback))
            callback = np.load(load_test_callback, allow_pickle=True).flat[0]
            self.y_value = callback['y_value']
            self.y_target = callback['y_target']
            self.y_pred = callback['y_pred']

            if self.RL_set == 'train':
                logging.info("load from {}".format(
                    self.config_pred.global_path_load + 'sequences_RL/indexes_testsetreftau{}_{}ep_{}steps.npy'.format(
                        self.tau_ref, self.nb_episode,
                        self.nb_step)))
                self.starts_episodes = np.load(
                    self.config_pred.global_path_load + 'sequences_RL/indexes_testsetreftau{}_{}ep_{}steps.npy'.format(
                        self.tau_ref, self.nb_episode,
                        self.nb_step))
            elif self.RL_set == 'test':
                logging.info("load from {}".format(
                    self.config_pred.global_path_load + 'sequences_RL/indexes_evalsetreftau{}_{}eval_{}steps.npy'.format(
                        self.tau_ref, self.nb_eval,
                        self.nb_step)))
                self.starts_episodes = np.load(
                    self.config_pred.global_path_load + 'sequences_RL/indexes_evalsetreftau{}_{}eval_{}steps.npy'.format(
                        self.tau_ref, self.nb_eval,
                        self.nb_step))
            else:
                logging.info("load from {}".format(
                    self.config_pred.global_path_load + 'sequences_RL/indexes_compsetreftau{}_{}eval_{}steps.npy'.format(
                        self.tau_ref, self.nb_eval,
                        self.nb_step)))
                self.starts_episodes = np.load(
                    self.config_pred.global_path_load + 'sequences_RL/indexes_compsetreftau{}_{}eval_{}steps.npy'.format(
                        self.tau_ref, self.nb_eval,
                        self.nb_step))

        self.df = self.df_tab(signalevent.index_df_tt, signalevent.df_tt)

        self.running_index = 0

        self.where_start_episode = int(
            np.where((self.dict_sequence[self.model_set][:, 0] == self.starts_episodes[self.running_index, 0]) & (
                    self.dict_sequence[self.model_set][:, 1] == self.starts_episodes[self.running_index, 1]))[0])

        self.knowledge = np.ones(self.seq_size + info_size) * np.NaN
        self.god_knowledge = np.ones(self.seq_size + info_size) * np.NaN

        self.df_now = None
        self.df_next = None

    # ------------------------------------------
    def get_class_label(self, value):
        if value < self.classes_edges[0]:
            self.classes_edges[0] = value
        elif value > self.classes_edges[-1]:
            self.classes_edges[-1] = value + 1

        classe_value, _ = np.histogram(value, bins=self.classes_edges)
        classe_label = np.where(classe_value == 1)[0][0]

        return classe_label

    # ------------------------------------------
    def df_tab(self, index, df):
        df_tab = np.zeros_like(index)

        where_df = np.where(index == 1)
        for i in range(where_df[0].size):
            df_tab[where_df[0][i], where_df[1][i]] = df[i]

        return df_tab

    # ------------------------------------------
    def update_indexes(self):
        self.running_index = self.running_index + 1

        self.where_start_episode = int(
            np.where((self.dict_sequence[self.model_set][:, 0] == self.starts_episodes[self.running_index, 0]) & (
                    self.dict_sequence[self.model_set][:, 1] == self.starts_episodes[self.running_index, 1]))[0])

    # ------------------------------------------
    def update_knowledge(self, index):
        cycle = self.dict_sequence[self.model_set][index, 0]
        t = self.dict_sequence[self.model_set][index, 1]

        raw = self.signal_flu.f[cycle, t: t + self.seq_size]
        if self.info_size == 0:
            self.knowledge = raw
            self.god_knowledge = raw
        elif self.info_size == 1:
            self.knowledge = np.concatenate((raw, np.array([int(self.y_pred[index])]).astype(int) if self.y_pred is not None else None))
            self.god_knowledge = np.concatenate((raw, np.array([self.get_class_label(self.y_value[index])]).astype(int)))
        else:
            self.knowledge = np.concatenate((raw, self.y_pred[int(index - self.info_size + 1): int(index + 1)].astype(
                int) if self.y_pred is not None else None))
            self.god_knowledge = np.concatenate((raw, np.array([self.get_class_label(self.y_value[i]) for i in
                                           range(index - self.info_size + 1, index + 1)]).astype(int)))
            # self.god_knowledge = self.y_target[int(index - self.info_size + 1): int(index + 1)].astype(int)

    # ------------------------------------------
    def update_df_now(self, index):
        cycle = self.dict_sequence[self.model_set][index, 0]
        t = self.dict_sequence[self.model_set][index, 1]

        self.df_now = self.df[cycle, t + self.seq_size - 1]

    # ------------------------------------------
    def update_df_next(self, index):
        cycle = self.dict_sequence[self.model_set][index, 0]
        t = self.dict_sequence[self.model_set][index, 1]

        self.df_next = self.df[cycle, t + self.seq_size]

    # ------------------------------------------
    def update_lab(self, index):
        self.update_knowledge(index)
        self.update_df_now(index)
        self.update_df_next(index)


def decision(my_city, pred):
    # print(my_city.class_risk[pred], my_city.l)
    if my_city.class_risk[pred] >= my_city.l:
        action = 1
    else:
        action = 0

    return action

def real_classe_futur(df_next, decade, nb_decade=5):
    return int(which_classe(which_decade(df_next, decade), nb_decade))


class DecisionalKnitCityRaw(gym.Env):
    """
        """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,  city, lab, simplet, delphes, lab_args,  decade, dt, knowledge_type,
                 ini_pos=[0, 0], recup_info=True):

        self.viewer = None
        self.city = city
        self.lab = lab
        self.lab.set_lab_args(lab_args['nb_eval'], lab_args['info_size'], lab_args['model_set'], lab_args['RL_set'])
        self.simplet = simplet
        self.delphes = delphes
        self.decade = decade
        self.ini_pos = ini_pos
        self.knowledge_type = knowledge_type
        self.recup_info = recup_info
        self.dt = dt  # 1

        self.nbclasses, self.knowledge, self.df_now = self.__recup_info_lab()

        self.low_state = np.array([-5 for _ in range(self.knowledge.size)]).astype(np.float)
        self.high_state = np.array([5 for _ in range(self.knowledge.size)]).astype(np.float)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float)
        self.state = None

        self.reward_episode = []

        self.__seed()
        self.reset()

    ###--------------------------------------------###
    def create_lab(self):

        self.lab = KnitLab(self.lab_arg['date'], self.lab_arg['version_pred'], self.lab_arg['model'],
                           self.lab_arg['trainsize'], self.lab_arg['sub_version'], self.lab_arg['info_size'],
                           self.lab_arg['nb_step'], self.lab_arg['nb_episode'], self.lab_arg['nb_eval'],
                           self.lab_arg['tau_ref'], self.lab_arg['model_set'], self.lab_arg['RL_set'])



        ###--------------------------------------------###

    ###--------------------------------------------###
    def __recup_info_lab(self):

        nbclasses = np.size(self.lab.classes_edges) - 1
        knowledge = self.lab.knowledge
        df_now = self.lab.df_now

        return nbclasses, knowledge, df_now

    ###--------------------------------------------###
    def __recup_info_city(self, city):
        dict_on_city = dictdata()

        dict_on_city.add('life_cost', city.running_life_cost)
        dict_on_city.add('days_out', city.running_days_out)
        dict_on_city.add('cost', city.running_cost)
        dict_on_city.add('death_per_decade', city.death_per_decade)
        dict_on_city.add('events', city.past_events)
        dict_on_city.add('out_events', city.out_events)

        return dict_on_city

        ###--------------------------------------------###

    ###--------------------------------------------###
    def get_info_eval(self):
        end_delphes = self.__recup_info_city(self.delphes)
        end_simplet = self.__recup_info_city(self.simplet)
        end_city = self.__recup_info_city(self.city)

        return np.array([end_city, end_simplet, end_delphes])

        ###--------------------------------------------###

    ###--------------------------------------------###
    def __update_reward_episode(self, reward):
        self.reward_episode.append(reward)

    ###--------------------------------------------###
    def __seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

        ###--------------------------------------------###

    ###--------------------------------------------###
    def _get_ob(self):
        s = self.state
        return s  # -self.nb_ressort]

    ###--------------------------------------------###
    def reset(self):

        if self.lab.RL_set == 'train':
            logging.info("{} | episode {}".format(self.lab.RL_set, self.lab.running_index))
            logging.debug(
                "running index = {} | index_0 = {}".format(self.lab.running_index, self.lab.where_start_episode))
        if np.size(self.reward_episode) != 0:
            self.__update_reward_episode(self.traj_reward[-1] + self.traj_reward_delphes[-1])

        fist_index = self.lab.where_start_episode
        self.lab.update_lab(fist_index)
        self.city.initialise()
        self.simplet.initialise()
        self.delphes.initialise()

        if self.knowledge_type == 'NN':
            self.state = np.asarray(self.lab.knowledge)
        else:
            self.state = np.asarray(self.lab.god_knowledge)

            if self.recup_info:
                self.traj_where = np.array([self.city.where])
            self.traj_death = np.array([self.city.life_cost])
            self.traj_days_out = np.array([self.city.days_out])
            self.traj_death_per_decade = self.city.death_per_decade
            self.traj_reward = [self.city.cost]
            self.traj_reward_delphes = [self.delphes.cost]
            if np.size(self.reward_episode) == 0:
                self.__update_reward_episode(0)

        return self._get_ob()

            ###--------------------------------------------###

    ###--------------------------------------------###
    def step(self, action):
        ''' n pas dasn un épisode'''

        reward = 0.
        ### action for delphes for ground state
        action_delphes = decision(self.delphes,
                                  real_classe_futur(self.lab.df_next, self.decade))
        self.delphes.update_where(action_delphes)
        self.delphes.update_t()

        ### action for simplet town for ground state
        # action_simplet = decision(self.simplet, int(self.state[-1]))
        if self.knowledge_type == 'NN':
            action_simplet = decision(self.simplet, int(self.lab.knowledge[-1]) if self.lab.info_size != 0 else 0)
        else:
            action_simplet = decision(self.simplet, int(self.lab.knowledge[-1]) if self.lab.info_size != 0 else 0)

        self.simplet.update_where(action_simplet)
        self.simplet.update_t()

        ### update pas de tps
        self.city.update_where(action)
        self.city.update_t()

        running_index = self.lab.where_start_episode + self.city.t
        self.lab.update_lab(running_index)
        self.delphes.update_city(self.lab.df_now)
        self.simplet.update_city(self.lab.df_now)
        self.city.update_city(self.lab.df_now)

        if self.knowledge_type == 'NN':
            self.state = np.asarray(self.lab.knowledge)
        else:
            self.state = np.asarray(self.lab.god_knowledge)

        reward = self.city.cost

        if self.city.t == self.lab.nb_step:
            done = True
            if self.knowledge_type == 'god':
                logging.debug(
                    "out days {} | nb days out = {}".format(self.city.out_events, self.city.running_days_out))
                logging.debug("events {}".format(self.city.past_events))
                logging.debug("cost = {} | cost delphes = {} | delta = {} ".format(self.city.running_cost,
                                                                                  self.delphes.running_cost, np.abs(
                        self.delphes.running_cost - self.city.running_cost)))
                logging.debug('Check Delphes')
                logging.debug(
                    "out days {} | nb days out = {}".format(self.delphes.out_events, self.delphes.running_days_out))
                logging.debug("events {}".format(self.delphes.past_events))
            else:
                logging.debug(
                    "out days {} | nb days out = {}".format(self.city.out_events, self.city.running_days_out))
                logging.debug("events {}".format(self.city.past_events))
                logging.debug("simplet days out {} | nb days out = {}".format(self.simplet.out_events,
                                                                             self.simplet.running_days_out))
                logging.debug("cost = {} | cost delphes = {} | delta = {} ".format(self.city.running_cost,
                                                                                  self.delphes.running_cost, np.abs(
                        self.delphes.running_cost - self.city.running_cost)))
                logging.debug('Check Delphes')
                logging.debug(
                    "out days {} | nb days out = {}".format(self.delphes.out_events, self.delphes.running_days_out))
                logging.debug("events {}".format(self.delphes.past_events))

            if self.lab.running_index != self.lab.starts_episodes.shape[0] - 1:
                self.lab.update_indexes()
            else:
                if self.lab.RL_set == 'test':
                    self.lab.running_index = 0
                    self.lab.where_start_episode = int(
                        np.where((self.lab.dict_sequence[self.lab.model_set][:, 0] == self.lab.starts_episodes[
                            self.lab.running_index, 0]) & (
                                         self.lab.dict_sequence[self.lab.model_set][:, 1] ==
                                         self.lab.starts_episodes[
                                             self.lab.running_index, 1]))[0])

        else:
            done = False

        return (self._get_ob(), reward, done, {})
