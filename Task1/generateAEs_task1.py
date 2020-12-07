import os
import sys


module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#############################

#from utils.data import subsampling
from utils.file import load_from_json

# load the configurations for the experiment
data_configs = load_from_json("configs/data-mnist.json")

# load the full-sized benign samples
bs_path = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
X_bs = np.load(data_configs.get(bs_path)

# load the corresponding true labels
tl_path = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
labels = np.load(tl_path)


######################################

from tutorials.craft_adversarial_examples import generate_ae
from utils.model import load_lenet
from utils.metrics import error_rate
from attacks.attack import generate

# loading experiment configurations
model_configs = load_from_json("configs/model-mnist.json")
data_configs = load_from_json("configs/data-mnist.json")
attack_configs = load_from_json("configs/attack-zk-mnist.json")

# load the targeted model
model_file = os.path.join(model_configs.get("dir"), model_configs.get("um_file"))
target = load_lenet(file=model_file, wrap=True)

# load the benign samples
data_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
data_bs = np.load(data_file)
# load the corresponding true labels
label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
labels = np.load(label_file)

# generate AEs
# let save=True and specify an output folder to save the generated AEs
generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_configs, save=True, 
            output_dir=data_configs.get("ae_dir"))

#################################################
from tutorials.eval_model import evaluate

# evaluate
outData = evaluate(trans_configs=trans_configs,
                   model_configs=model_configs,
                   data_configs=data_configs,
                   save=False,
                   output_dir=output_root)

outData.to_csv(os.path.join(data_configs.get("results_dir", "dataOut.csv"))








