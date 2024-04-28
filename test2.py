
from rleplus.examples.amphitheater.env import AmphitheaterEnv

from pythermalcomfort.models import pmv, pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments

from model.human import Human

import numpy as np

env = AmphitheaterEnv({"output": "/tmp/tests_output"})

# termal comfort parameters
vr = 0.3  # average air speed, [m/s]
rh = 50  # relative humidity, [%]

# Create human object with the default parameters
humans = [Human()]

# the training variables
epocs = 1000
silent = False

# get initial observation
obs, _ = env.reset()

#training loop
for i in range(epocs):
    print('Epoc:', i)
    # get the dry bulb air temperature
    tdb = obs[1]
    # get the mean radiant temperature
    tr = obs[1]

    # iterate over humans
    for human in humans:
        # calculate pmv value for the current human
        temp_pmv = human.calcpmv(tdb, tr, vr, rh)

        # get probability of complaint
        prob = human.calcprobability(temp_pmv)

        # generate random number between 0 and 1
        rand = np.random.rand()

        # check if the human complains
        complaint = rand < prob



        