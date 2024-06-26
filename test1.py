
from rleplus.examples.bbright.env import BBrightEnv

from pythermalcomfort.models import pmv, pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments

import numpy as np

env = BBrightEnv({"output": "/tmp/tests_output"}, reward_type="human", nhumans=1)

# input variables
tdb = 27  # dry bulb air temperature, [$^{\circ}$C]
tr = 25  # mean radiant temperature, [$^{\circ}$C]
rh = 50  # relative humidity, [%]

vr = 0.1
met = 1.1
clo = 1.4

obs, _ = env.reset()
# print(obs)
pmvs = []
rewards = []

#get the variables from the environment
var_dict = env.get_variables()
var_keys = var_dict.keys()
# create a dict with keys from variable keys and empty arrays for the observations
obs_dict = {key: [] for key in var_keys}

print(vr, met, clo)

for i in range(12):
    obs, rew, done, _, _ = env.step(24.0)
    # append the observations to the dict
    for key, val in zip(var_keys, obs):
        obs_dict[key].append(val)
    # collect rewards
    rewards.append(rew)
    # calculate and collect the PMV index
    _pmv = pmv(tdb=obs_dict["air_tmp"][-1], tr=obs_dict["rad_tmp"][-1], vr=vr, rh=obs_dict["air_hum"][-1], met=met, clo=clo)
    pmvs.append(_pmv)
    # print the datetime
    # print(obs[-1])
    
    if done:
        break
# print(obs, rew, done)
# print the observations
for key, val in obs_dict.items():
    print(key, val)
# print the rewards
print("rewards: ", rewards)
# print the PMV index
print("pmvs: ", pmvs)

env.close()
