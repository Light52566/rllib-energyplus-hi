
from rleplus.examples.amphitheater.env import AmphitheaterEnv

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
print_every = 100
no_complaint_threshold = 4

# get initial observation
obs, _ = env.reset()

#training loop
for i in range(epocs):
    if i % print_every == 0:
        print('Epoc:', i)

    # get the dry bulb air temperature
    tdb = obs[1]
    # get the mean radiant temperature
    tr = obs[1]

    # no complaint counter
    no_complaint = 0

    # cumulative reward for timestep
    cum_reward = 0

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

        if complaint:
            cum_reward += -1
        else:
            no_complaint += 1
        
        if no_complaint == no_complaint_threshold:
            cum_reward += 1
            no_complaint = 0
        
        if not silent and i % print_every == 0:
            print('Temp:', tdb, tr, ',PMV:', temp_pmv, ',Prob:', prob, ',Complaint:', complaint, ',Cumulative reward:', cum_reward)

    # step the environment
    obs, rew, done, _, _ = env.step(25)

env.close()
print("Done!")
        
    





        