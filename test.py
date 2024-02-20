
from rleplus.examples.amphitheater.env import AmphitheaterEnv

from pythermalcomfort.models import pmv, pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments

import numpy as np

env = AmphitheaterEnv({"output": "/tmp/tests_output"})

# input variables
tdb = 27  # dry bulb air temperature, [$^{\circ}$C]
tr = 25  # mean radiant temperature, [$^{\circ}$C]
v = 0.3  # average air speed, [m/s]
rh = 50  # relative humidity, [%]
activity = "Typing"  # participant's activity description
garments = ["Sweatpants", "T-shirt", "Shoes or sandals"]

met = met_typical_tasks[activity]  # activity met, [met]
icl = sum(
    [clo_individual_garments[item] for item in garments]
)  # calculate total clothing insulation

# calculate the relative air velocity
vr = v_relative(v=v, met=met)
# calculate the dynamic clothing insulation
clo = clo_dynamic(clo=icl, met=met)

obs, _ = env.reset()
print(obs)
pmvs = []
# calculate PMV in accordance with the ASHRAE 55 2020
# results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard="ASHRAE")
# pmvs.append(results['pmv'])
for i in range(1000):
    obs, rew, done, _, _ = env.step(25)
    results = pmv_ppd(tdb=obs[1], tr=obs[1], vr=vr, rh=rh, met=met, clo=clo, standard="ASHRAE")
    pmvs.append(results['pmv'])
    if done:
        break
print(obs, rew, done)
print('indoor tempt:', obs[1])
print(np.array(pmvs).mean())
env.close()
