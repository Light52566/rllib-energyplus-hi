from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import gymnasium as gym
import numpy as np

from rleplus.env.energyplus import EnergyPlusEnv
from rleplus.env.utils import override

from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments



class AmphitheaterEnv(EnergyPlusEnv):
    """University amphitheatre environment.

    This environment is based on an actual university amphitheatre in Luxembourg. The building model
    (calibrated against actual energy consumption) of this amphitheatre is available in the same folder.
    The weather file is a typical meteorological year (TMY) weather file.

    HVAC: an AHU with a heating hot water coil, and supply and exhaust air fans.

    Target actuator: supply air temperature setpoint.
    """
    base_path = Path(__file__).parent
    pmv_dict = {}

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.pmv_dict["v"] = 0.3
        self.pmv_dict["rh"] = 50
        self.pmv_dict["activity"] = "Typing"
        self.pmv_dict["garments"] = ["Sweatpants", "T-shirt"]
        self.pmv_dict["met"] = met_typical_tasks[self.pmv_dict["activity"]]
        self.pmv_dict["icl"] = sum([clo_individual_garments[item] for item in self.pmv_dict["garments"]])
        self.pmv_dict["vr"] = v_relative(v=self.pmv_dict["v"], met=self.pmv_dict["met"])
        self.pmv_dict["clo"] = clo_dynamic(clo=self.pmv_dict["icl"], met=self.pmv_dict["met"])

    @override(EnergyPlusEnv)
    def get_weather_file(self) -> Union[Path, str]:
        return self.base_path / "LUX_LU_Luxembourg.AP.065900_TMYx.2004-2018.epw"

    @override(EnergyPlusEnv)
    def get_idf_file(self) -> Union[Path, str]:
        return self.base_path / "model.idf"

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
        return {
            # °C
            "oat": ("Site Outdoor Air DryBulb Temperature", "Environment"),
            # °C
            "iat": ("Zone Mean Air Temperature", "TZ_Amphitheater"),
            # ppm
            "co2": ("Zone Air CO2 Concentration", "TZ_Amphitheater"),
            # heating setpoint (°C)
            "htg_spt": ("Schedule Value", "HTG HVAC 1 ADJUSTED BY 1.1 F"),
            # cooling setpoint (°C)
            "clg_spt": ("Schedule Value", "CLG HVAC 1 ADJUSTED BY 0 F"),
        }

    @override(EnergyPlusEnv)
    def get_meters(self) -> Dict[str, str]:
        return {
            # HVAC elec (J)
            "elec": "Electricity:HVAC",
            # District heating (J)
            "dh": "Heating:DistrictHeatingWater",
        }

    @override(EnergyPlusEnv)
    def get_actuators(self) -> Dict[str, Tuple[str, str, str]]:
        return {
            # supply air temperature setpoint (°C)
            "sat_spt": ("System Node Setpoint", "Temperature Setpoint", "Node 3")
        }

    @override(EnergyPlusEnv)
    def compute_reward(self, obs: Dict[str, float]) -> float:
        """A simple reward function that penalizes on thermal comfort."""
        results = pmv_ppd(
            tdb=obs["iat"], tr=obs["iat"], vr=self.pmv_dict["vr"], rh=self.pmv_dict["rh"], met=self.pmv_dict["met"], clo=self.pmv_dict["clo"], standard="ASHRAE"
        )

        reward = 1.0 - np.abs(results["pmv"])
        return reward

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
