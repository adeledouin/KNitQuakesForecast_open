from gym.envs.registration import register
# import Config_env as Config_env
import Config_env_raw as Config_env_raw
import Config_env
import gym
import logging

logging.basicConfig(format='| %(levelname)s | %(asctime)s | %(message)s', level=logging.INFO)

env_dict = gym.envs.registration.registry.env_specs.copy()
## raw data

for env in env_dict:
    for i in range(1, 3):
        if 'env-KnitCityRaw-v0{}'.format(i) in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
        if 'env-KnitCityRaw-v0{}-eval'.format(i) in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
        if 'env-KnitCityRaw-v0{}-comp'.format(i) in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]

v_list_raw = ['0{}'.format(i) for i in range(1, 8)]

def my_register_raw(iid):
   register(
               id='env-KnitCityRaw-v{}'.format(iid),
               entry_point='gym_KnitCity.envs:DecisionalKnitCityRaw',
               max_episode_steps=25000,
               kwargs=Config_env_raw.exp['{}'.format(iid)]
               )

   register(
       id='env-KnitCityRaw-eval-v{}'.format(iid),
       entry_point='gym_KnitCity.envs:DecisionalKnitCityRaw',
       max_episode_steps=25000,
       kwargs=Config_env_raw.exp['{}-eval'.format(iid)]
   )

   register(
       id='env-KnitCityRaw-comp-v{}'.format(iid),
       entry_point='gym_KnitCity.envs:DecisionalKnitCityRaw',
       max_episode_steps=25000,
       kwargs=Config_env_raw.exp['{}-comp'.format(iid)]
   )

logging.info('registering raw')
for iid in v_list_raw:
    my_register_raw(iid)

## dans results supp paper

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    for i in range(1, 6):
        for j in [0, 1]:
            if 'env-KnitCity-v{}{}'.format(j, i) in env:
                print("Remove {} from registry".format(env))
                del gym.envs.registration.registry.env_specs[env]
            if 'env-KnitCity-eval-v{}{}'.format(j, i) in env:
                print("Remove {} from registry".format(env))
                del gym.envs.registration.registry.env_specs[env]
            if 'env-KnitCity-comp-v{}{}'.format(j, i) in env:
                print("Remove {} from registry".format(env))
                del gym.envs.registration.registry.env_specs[env]

v_list = ['0{}'.format(i) for i in range(1, 7)] + ['1{}'.format(i) for i in range(1, 7)]

def my_register_suppaper(iid):
   register(
               id='env-KnitCity-v{}'.format(iid),
               entry_point='gym_KnitCity.envs:DecisionalKnitCity',
               max_episode_steps=25000,
               kwargs=Config_env.exp['{}'.format(iid)]
               )

   register(
       id='env-KnitCity-eval-v{}'.format(iid),
       entry_point='gym_KnitCity.envs:DecisionalKnitCity',
       max_episode_steps=25000,
       kwargs=Config_env.exp['{}-eval'.format(iid)]
   )

   register(
       id='env-KnitCity-comp-v{}'.format(iid),
       entry_point='gym_KnitCity.envs:DecisionalKnitCity',
       max_episode_steps=25000,
       kwargs=Config_env.exp['{}-comp'.format(iid)]
   )

logging.info('registering : suppaper env')
for iid in v_list:
    my_register_suppaper(iid)
