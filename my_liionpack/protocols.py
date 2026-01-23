import numpy as np
import pybamm
from utils import *


experiment = pybamm.Experiment(
        [   
            # "Discharge at 0.1 C for 600 minutes or until 2.8 V",
            # "Rest for 180 minutes",
            # f"Charge at {current} A for {60*0.5/rate} minutes",
            # "Rest for 180 minutes",
            # f"Discharge at {current} A until 2.6 V",

            "Charge at 10 A for 600 minutes or until 3.6 V",
            "Rest for 180 minutes",
            *([f"Discharge at {10} A for 10 minutes",
            "Rest for 180 minutes"]*1)
            
        ],
        period="1 minute",
    )

protocol, terminations, step_types = generate_protocol_from_experiment(experiment)
print("Protocol and terminations generated from experiment.")
print(protocol[0],
      "----------------\n\n",
      terminations,
      "----------------\n\n",
      step_types)