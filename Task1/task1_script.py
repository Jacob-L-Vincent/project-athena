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
data_configs = load_from_json("../src/configs/demo/data-mnist.json")
output_root = "../results"


# load the full-sized benign samples
#file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
file = 'data/test_BS-mnist-clean.npy'
X_bs = np.load(file)

# load the corresponding true labels
#file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
file = 'data/test_Label-mnist-clean.npy'
labels = np.load(file)

# get random subsamples
# for MNIST, num_classes is 10
# files "subsamples-mnist-ratio_0.1-xxxxxx.npy" and "sublabels-mnist-ratio_0.1-xxxxxx.npy"
# will be generated and saved at "/results" folder, where "xxxxxx" are timestamps.
#subsamples, sublabels = subsampling(data=X_bs,
#                                    labels=labels,
#                                    num_classes=10,
#                                    filepath=output_root,
#                                    filename='mnist')

######################################

from tutorials.craft_adversarial_examples import generate_ae
from utils.model import load_lenet
from utils.metrics import error_rate
from attacks.attack import generate

# loading experiment configurations
model_configs = load_from_json("../src/configs/demo/model-mnist.json")
data_configs = load_from_json("../src/configs/demo/data-mnist.json")
attack_configs = load_from_json("../src/configs/demo/attack-zk-mnist.json")

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
# in this example, we generate AEs for 5 benign samples
#data_bs = data_bs[:20]
#labels = labels[:20]
# let save=True and specify an output folder to save the generated AEs
generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_configs, save=True, 
            output_dir="data/")

#################################################
from tutorials.eval_model import evaluate
# load experiment configurations
trans_configs = load_from_json("../src/configs/demo/athena-mnist.json")
model_configs = load_from_json("../src/configs/demo/model-mnist.json")
data_configs = load_from_json("../src/configs/demo/data-mnist.json")

output_dir = "results/"

# evaluate
outData = evaluate(trans_configs=trans_configs,
                   model_configs=model_configs,
                   data_configs=data_configs,
                   save=False,
                   output_dir=output_root)

outData.to_csv('results/dataOut.csv')








