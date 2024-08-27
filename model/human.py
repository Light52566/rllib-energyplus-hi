import numpy as np
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments


class Human:
    def __init__(self, icl:float=1.1, met:float=1.4, 
                 exp_a:float=1.0, exp_b:float=2.0, exp_c:float=0.9, exp_d:float=2.7) -> None:
        # pmv parameters
        self.icl = icl # total clothing insulation, [clo]
        self.met = met  # activity metabolic rate, [met]


        # interaction parameters
        # self.dist_skew = 0.0  # skewness of the probability distribution
        # self.dist_loc = 0.0  # location of the probability distribution
        # self.dist_scale = 1 # scale of the probability distribution

        # interaction probability parameters
        self.prob_func = "exp" # Probability function to use. Options: "sigmoid", "exp"

        # P(pmv) = exp(ax-b) + exp(-cx-d)
        self.exp_a = exp_a
        self.exp_b = exp_b
        self.exp_c = exp_c
        self.exp_d = exp_d
        self.normalizer = 0.1

        # P(pmv) = 1 / (1 + exp(-k1 * (pmv - T1))) + 1 / (1 + exp(-k2 * (pmv - T2))
        self.k1 = 6.6 # Steepness parameter for the rising side of the probability curve.
        self.T1 = 1.2 # pmv at which the rising side of the curve transitions from low to high probability.
        self.k2 = -2.6 # Steepness parameter for the falling side of the probability curve.
        self.T2 = -1.6 # pmv at which the falling side of the curve transitions from high to low probability.

    def calcpmv(self, tdb: float, tr: float, v: float, rh: float) -> float:
        """
        Calculate the Predicted Mean Vote (PMV) based on the input variables.

        Parameters:
        - tdb: Dry bulb air temperature, [°C]
        - tr: Mean radiant temperature, [°C]
        - v: Average air speed, [m/s]
        - rh: Relative humidity, [%]

        Returns:
        - pmv: Predicted Mean Vote.
        """
        vr = v_relative(v=v, met=self.met)
        clo = clo_dynamic(clo=self.icl, met=self.met)
        results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=self.met, clo=clo, standard="ASHRAE")
        return results['pmv']
    
    def temp2pmv(self, min_tdb = 10.0, max_tdb = 40.0, step_tdb = 0.5, tr = 25, v = 0.1, rh =50) -> dict:
        """ Uniformly samples the pmv values varying the temperature
        min_tdb: min dry bulb air temperature, [°C]
        max_tdb: max dry bulb air temperature, [°C]
        step_tdb: step of the dry bulb air temperature, [°C]
        tr: mean radiant temperature, [°C]
        v: average air speed, [m/s]
        rh: relative humidity, [%]
        """
        pmvs = {"pmv": [], "tdb": []}
        vr = v_relative(v=v, met=self.met)
        clo = clo_dynamic(clo=self.icl, met=self.met)
        for tdb in np.arange(min_tdb, max_tdb, step_tdb):
            results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=self.met, clo=clo, standard="ASHRAE")
            pmvs["pmv"].append(results['pmv'])
            pmvs["tdb"].append(tdb)
        return pmvs
    
    def calcprobability(self, pmv: float, ) -> float:
        """
        Calculate the probability of complaint based on the current pmv.

        Parameters:
        - pmv: Current pmv.

        Returns:
        - probability: Probability of complaint.
        """
        if self.prob_func == "exp":
            probability = np.exp(self.exp_a * pmv - self.exp_b) + np.exp(-self.exp_c * pmv - self.exp_d)
        elif self.prob_func == "sigmoid":
            rising_side = 1 / (1 + np.exp(-self.k1 * (pmv - self.T1)))
            falling_side = 1 / (1 + np.exp(-self.k2 * (pmv - self.T2)))
            probability = rising_side + falling_side
        else:
            probability = 0.0
        # limit  probabilities between 0 and 1
        probability = max(0.0, min(1.0, self.normalizer * probability))
        
        return probability
    
    def temp2prob(self, min_tdb = 10.0, max_tdb = 40.0, step_tdb = 0.5, tr = 25, v = 0.1, rh =50) -> dict:
        """ Uniformly samples the probability of complaint varying the temperature
        min_tdb: min dry bulb air temperature, [°C]
        max_tdb: max dry bulb air temperature, [°C]
        step_tdb: step of the dry bulb air temperature, [°C]
        tr: mean radiant temperature, [°C]
        v: average air speed, [m/s]
        rh: relative humidity, [%]
        """
        probabilities = {"probability": [], "tdb": [], "pmv": []}
        pmvs = self.temp2pmv(min_tdb, max_tdb, step_tdb, tr, v, rh)
        for pmv, tdb in zip(pmvs["pmv"], pmvs["tdb"]):
            probability = self.calcprobability(pmv)
            probabilities["probability"].append(probability)
            probabilities["tdb"].append(tdb)
            probabilities["pmv"].append(pmv)
        return probabilities
    
    def pmv2prob(self, min_pmv = -3.0, max_pmv = 3.0, step_pmv = 0.1) -> dict:
        """ Uniformly samples the probability of complaint varying the pmv
        min_pmv: min pmv, [-]
        max_pmv: max pmv, [-]
        step_pmv: step of the pmv, [-]
        """
        probabilities = {"probability": [], "pmv": []}
        for pmv in np.arange(min_pmv, max_pmv, step_pmv):
            probability = self.calcprobability(pmv)
            probabilities["probability"].append(probability)
            probabilities["pmv"].append(pmv)
        return probabilities
    
    def setProbabilityFunction(self, prob_func: str) -> None:
        """
        Set the probability function to use.

        Parameters:
        - prob_func: Probability function to use. Options: "sigmoid", "exp"
        """
        self.prob_func = prob_func
    
