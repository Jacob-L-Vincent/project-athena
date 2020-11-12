import os
import sys
module_path = os.path.abspath(os.path.join('../../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import os
from matplotlib import pyplot as plt

from utils.model import load_pool, load_lenet
from utils.file import load_from_json
from utils.metrics import error_rate, get_corrections
from models.athena import Ensemble, ENSEMBLE_STRATEGY
from tutorials.collect_raws import collect_raw_prediction


# load experiment configurations
trans_configs = load_from_json("../../src/configs/demo/athena-mnist.json")
model_configs = load_from_json("../../src/configs/demo/model-mnist.json")
data_configs = load_from_json("../../src/configs/demo/data-mnist.json")

output_dir = "../../results"

# collect the probabilities
collect_raw_prediction(trans_configs=trans_configs,
                       model_configs=model_configs,
                       data_configs=data_configs,
                       )











