from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import gymnasium as gym
import numpy as np

from rleplus.env.energyplus import EnergyPlusEnv
from rleplus.env.utils import override

from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments

from model.human import Human



class BBrightEnv(EnergyPlusEnv):
    """B. Bright smartspace environment.

    This environment is based on an actual one room space called BRIGHT lab. 
    It is B. Amsterdam's newest event/office space in partnership with BRIGHT by COD.
    The weather file is a typical meteorological year (TMY) weather file for Amsterdam.

    HVAC: an ideal HVAC duel setpoint system for both heating and cooling.

    Target actuator: air temperature setpoint.
    """
    base_path = Path(__file__).parent
    pmv_dict = {}

    def __init__(self, env_config: Dict[str, Any], nhumans: int = 1):
        super().__init__(env_config)
        self.pmv_dict["v"] = 0.3
        self.pmv_dict["rh"] = 50
        self.pmv_dict["activity"] = "Typing"
        # self.pmv_dict["garments"] = ["Sweatpants", "T-shirt"]
        self.pmv_dict["met"] = met_typical_tasks[self.pmv_dict["activity"]]
        # self.pmv_dict["icl"] = sum([clo_individual_garments[item] for item in self.pmv_dict["garments"]])
        self.pmv_dict["vr"] = v_relative(v=self.pmv_dict["v"], met=self.pmv_dict["met"])
        # self.pmv_dict["clo"] = clo_dynamic(clo=self.pmv_dict["icl"], met=self.pmv_dict["met"])

        self.humans = [Human() for _ in range(nhumans)]

    @override(EnergyPlusEnv)
    def get_weather_file(self) -> Union[Path, str]:
        return self.base_path / "NLD_Amsterdam.062400_IWEC.epw"

    @override(EnergyPlusEnv)
    def get_idf_file(self) -> Union[Path, str]:
        return self.base_path / "BBright.idf"

    @override(EnergyPlusEnv)
    def get_observation_space(self) -> gym.Space:
        # observation space:
        # OAT, IAT, CO2, cooling setpoint, heating setpoint, fans elec, district heating
        low_obs = np.array([-40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        hig_obs = np.array([40.0, 40.0, 1e5, 30.0, 30.0, 1e8, 1e8])
        return gym.spaces.Box(low=low_obs, high=hig_obs, dtype=np.float32)

    @override(EnergyPlusEnv)
    def get_action_space(self) -> gym.Space:
        return gym.spaces.Discrete(100)

    @override(EnergyPlusEnv)
    def get_variables(self) -> Dict[str, Tuple[str, str]]:
        room_name = "Room_62b6a475-1bd9-45f0-8eb8-86bc21807113-0005a439"


        return {
            # # °C
            # "oat": ("Site Outdoor Air DryBulb Temperature", "Environment"),
            # # °C
            # "iat": ("Zone Mean Air Temperature", "TZ_Amphitheater"),
            # # ppm
            # "co2": ("Zone Air CO2 Concentration", "TZ_Amphitheater"),
            # # heating setpoint (°C)
            # "htg_spt": ("Schedule Value", "HTG HVAC 1 ADJUSTED BY 1.1 F"),
            # # cooling setpoint (°C)
            # "clg_spt": ("Schedule Value", "CLG HVAC 1 ADJUSTED BY 0 F"),

            # °C
            "zma": ("Zone Mean Air Temperature", room_name),
            # °C 
            "zhs": ("Zone Thermostat Heating Setpoint Temperature", room_name),
            # °C
            "zcs": ("Zone Thermostat Cooling Setpoint Temperature", room_name),
            # °C
            "sot": ("Site Outdoor Air Drybulb Temperature", u"Environment"),
             
        }

    @override(EnergyPlusEnv)
    def get_meters(self) -> Dict[str, str]:
        return {
            # HVAC elec (J)
            # "elec": "Electricity:HVAC",
            # District heating (J)
            # "dh": "Heating:DistrictHeatingWater",
        }

    @override(EnergyPlusEnv)
    def get_actuators(self) -> Dict[str, Tuple[str, str, str]]:
        actuator_key = 'Room_62b6a475-1bd9-45f0-8eb8-86bc21807113-0005a439'      
        component_type = 'Zone Temperature Control'
        cooling_control_type = 'Cooling Setpoint'
        heating_control_type = 'Heating Setpoint'
        
        return {
            # supply air temperature setpoint (°C)
            # "sat_spt": ("System Node Setpoint", "Temperature Setpoint", "Node 3")
            # heating setpoint (°C)
            "htg_spt": (component_type, heating_control_type, actuator_key),
            # cooling setpoint (°C)
            "clg_spt": (component_type, cooling_control_type, actuator_key)
        }

    @override(EnergyPlusEnv)
    def compute_reward(self, obs: Dict[str, float]) -> float:
        """A reward function that penalizes on human complaints and rewards no complaints."""
        # results = pmv_ppd(
        #     tdb=obs["iat"], tr=obs["iat"], vr=self.pmv_dict["vr"], rh=self.pmv_dict["rh"], met=self.pmv_dict["met"], clo=self.pmv_dict["clo"], standard="ASHRAE"
        # )
        # no complaint counter and threshold
        no_complaint = 0
        no_complaint_threshold = 4

        # cumulative reward for timestep
        step_cum_reward = 0

        # iterate over humans
        '''
        for human in self.humans:
            # calculate pmv value for the current human
            temp_pmv = human.calcpmv(obs["iat"], obs["iat"], self.pmv_dict["vr"], rh=self.pmv_dict["rh"])

            # get probability of complaint
            prob = human.calcprobability(temp_pmv)

            # generate random number between 0 and 1
            rand = np.random.rand()

            # check if the human complains
            complaint = rand < prob

            if complaint:
                step_cum_reward += -1
            else:
                no_complaint += 1
            
            if no_complaint == no_complaint_threshold:
                step_cum_reward += 1
                no_complaint = 0

        # reward = 1.0 - np.abs(results["pmv"])
        return step_cum_reward
        '''
        return 0

    @override(EnergyPlusEnv)
    def post_process_action(self, action: Union[float, List[float]]) -> Union[float, List[float]]:
        actual_range = (15.0, 30.0)

        return self._rescale(
            n=int(action), range1=(self.action_space.start, self.action_space.n), range2=actual_range  # noqa  # noqa
        )

    def _rescale(self, n: int, range1: Tuple[float, float], range2: Tuple[float, float]) -> float:
        delta1 = range1[1] - range1[0]
        delta2 = range2[1] - range2[0]
        return (delta2 * (n - range1[0]) / delta1) + range2[0]
