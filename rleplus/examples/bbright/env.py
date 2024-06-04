from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import gymnasium as gym
import numpy as np

from rleplus.env.energyplus import EnergyPlusEnv
from rleplus.env.utils import override

from pythermalcomfort.models import pmv, pmv_ppd

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

    def __init__(self, env_config: Dict[str, Any], reward_type: str = "pmv", nhumans: int = 1):
        super().__init__(env_config, reward_type=reward_type)
        self.pmv_dict["met"] = 1.1
        self.pmv_dict["vr"] = 0.1
        self.pmv_dict["clo"] = 1.4

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
        # out_tmp, air_tmp, opr_tmp, air_hum, htg_stp, clg_stp, eeq_htg, air_chg, rad_tmp
        low_obs = np.array([-40.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1e8, 0.0, 0.0])
        hig_obs = np.array([40.0, 40.0, 40.0, 100.0, 40.0, 40.0, 1e8, 10.0, 40.0])
        return gym.spaces.Box(low=low_obs, high=hig_obs, dtype=np.float32)

    @override(EnergyPlusEnv)
    def get_action_space(self) -> gym.Space:
        return gym.spaces.Discrete(100)

    @override(EnergyPlusEnv)
    def get_variables(self) -> Dict[str, Tuple[str, str]]:
        room_name = "Room_62b6a475-1bd9-45f0-8eb8-86bc21807113-0005a439"

        return {
            # °C
            "out_tmp": ("Site Outdoor Air Drybulb Temperature", u"Environment"),
            # °C
            "air_tmp": ("Zone Mean Air Temperature", room_name),
            # °C
            "opr_tmp": ("Zone Operative Temperature", room_name),
            # %
            "air_hum": ("Zone Air Relative Humidity", room_name),
            # °C 
            "htg_stp": ("Zone Thermostat Heating Setpoint Temperature", room_name),
            # °C
            "clg_stp": ("Zone Thermostat Cooling Setpoint Temperature", room_name),
            # always 0 so not worth reading
            # "ppl_cnt": ("Zone People Occupant Count", room_name),
            # causes error, not readable
            # "slr_rad": ("Zone Windows Total Transmitted Solar Radiation Rate", room_name),
            # 
            "eeq_htg": ("Zone Electric Equipment Total Heating Rate", room_name),
            # 
            "air_chg": ("Zone Infiltration Air Change Rate", room_name),
            # °C
            "rad_tmp": ("Zone Mean Radiant Temperature", room_name),  
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

        if self.reward_type == "zero":
            return 0.0
        elif self.reward_type == "pmv":
            # calculate the pmv value
            _pmv = pmv(tdb=obs["air_tmp"], tr=obs["rad_tmp"], vr=self.pmv_dict["vr"], rh=obs["air_hum"], met=self.pmv_dict["met"], clo=self.pmv_dict["clo"])
            # return negative distance of pmv from 0
            return -1*abs(_pmv)
        elif self.reward_type == "human":
            # no complaint counter and threshold
            no_complaint = 0
            no_complaint_threshold = 4

            # cumulative reward for timestep
            step_cum_reward = 0

            # iterate over humans
            
            for human in self.humans:
                # calculate pmv value for the current human
                temp_pmv = human.calcpmv(obs["air_tmp"], obs["rad_tmp"], self.pmv_dict["vr"], obs["air_hum"])

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
