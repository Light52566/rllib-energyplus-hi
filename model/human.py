import numpy as np
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments


class Human:
    def __init__(self) -> None:
        # pmv parameters
        self.icl = 1.1 # total clothing insulation, [clo]
        self.met = 1.4  # activity metabolic rate, [met]

        # interaction parameters
        self.dist_skew = 0.0  # skewness of the probability distribution
        self.dist_loc = 0.0  # location of the probability distribution
        self.dist_scale = 1 # scale of the probability distribution

    # Function that uniformly samples the pmv values varying the temperature
    # min_tdb: min dry bulb air temperature, [°C]
    # max_tdb: max dry bulb air temperature, [°C]
    # step_tdb: step of the dry bulb air temperature, [°C]
    # tr: mean radiant temperature, [°C]
    # v: average air speed, [m/s]
    # rh: relative humidity, [%]
    def uniform_sample_pmv(self, min_tdb = 10.0, max_tdb = 40.0, step_tdb = 0.5, tr = 25, v = 0.1, rh =50) -> dict:
        pmvs = {"pmv": [], "tdb": []}
        vr = v_relative(v=v, met=self.met)
        clo = clo_dynamic(clo=self.icl, met=self.met)
        for tdb in np.arange(min_tdb, max_tdb, step_tdb):
            results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=self.met, clo=clo, standard="ASHRAE")
            pmvs["pmv"].append(results['pmv'])
            pmvs["tdb"].append(tdb)
        return pmvs
    
