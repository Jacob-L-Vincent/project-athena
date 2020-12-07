#########################################################
##    generate the ensemble predictions for task 2     ##
##  created by Isaac Keohane isaackeohane95@gmail.com  ##
#########################################################

import os
import sys
module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import os
from matplotlib import pyplot as plt

from utils.file import load_from_json
from utils.model import load_pool
from models.athena import Ensemble, ENSEMBLE_STRATEGY

# load experiment configurations
trans_configs = load_from_json("configs/athena-mnist.json")
model_configs = load_from_json("configs/model-mnist.json")
data_configs = load_from_json("configs/data-mnist.json")

output_dir = "models/"
save_output = True
verbose = 10

#####################################################
### setup the ensemble pool of weak defenses

# This wdList can be changed to a list of indexes of weak defenses in the 
#  athena-mnist.json file to get a custom set of weak defenses used in the
#   emsemble. Make sure to then set "customList" True and "useActi..." False
#    both set to False makes it use all the transformations in trans_configs
wdList = []

useActiveList = False
customList = False


# def the setup function
def setup_ensemble(trans_configs, model_configs, use_logits=False,
                   useActiveList=True, customList=False, wdList=[]):

    
    # load the pool and create the ensemble
    pool, _ = load_pool(trans_configs=trans_configs,
                        model_configs=model_configs,
                        active_list=useActiveList,
                        use_logits=use_logits,
                        wrap=True,
                        customList = customList,
                        custom_wds = wdList
                        )
    athena = Ensemble(classifiers=list(pool.values()),
                      strategy=ENSEMBLE_STRATEGY.MV.value)

    return athena

# run setup_ensemble to make an ensemble pool of weak defenses
athena = setup_ensemble(trans_configs=trans_configs,
                        model_configs=model_configs,
                        use_logits=False,
                        useActiveList=useActiveList,
                        customList=customList, wdList=wdList)

############################################################################
#3# generate and collect ensemble predictions probabilities 
# load data
x_bs = np.load(data_configs.get('bs_file'))
if(verbose>5): print("\nbenign sample data dimensions: {}\n".format(x_bs.shape))
totalNumData = x_bs.shape[0]

# grab predictions
preds = athena.predict(x=x_bs) # raw is False by default
#preds_raw = athena.predict(x=x_bs,raw=True)
if(verbose>5): print("\n>>> Shape of benign ensemble predictions: {}\n".format(preds.shape))

# save predictions
if save_output:
    np.save(os.path.join(output_dir,"ensemPredic_benignInput_probs.npy"),preds)


 


