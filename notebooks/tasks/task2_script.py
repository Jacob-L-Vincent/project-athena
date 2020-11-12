#######################################################
##    Basic playground script for staging task 2     ##
## created by Isaac Keohane isaackeohane95@gmail.com ##
#######################################################

import os
import sys
module_path = os.path.abspath(os.path.join('../../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import os
from matplotlib import pyplot as plt
import random

from utils.model import load_pool, load_lenet
from utils.file import load_from_json
from utils.metrics import error_rate, get_corrections
from models.athena import Ensemble, ENSEMBLE_STRATEGY
from tutorials.collect_raws import collect_raw_prediction
from scripts.ourFuncs_task2 import generate_subset


# load experiment configurations
trans_configs = load_from_json("../../src/configs/demo/athena-mnist.json")
model_configs = load_from_json("../../src/configs/demo/model-mnist.json")
data_configs = load_from_json("../../src/configs/demo/data-mnist.json")

output_dir = "../../results"

   
# generate subset indexes to grab benign samples
subset, subsetElse = generate_subset(totalSize=5,number=2,doRandom=False,
                    odir='default',doSave=False)


# collect the probabilities
collect_raw_prediction(trans_configs=trans_configs,
                       model_configs=model_configs,
                       data_configs=data_configs,
                       subset=subset,
                       use_subset=True,
                       use_logits=False)











