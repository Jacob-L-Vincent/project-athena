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

from utils.model import load_pool, load_lenet
from utils.file import load_from_json
from utils.metrics import error_rate, get_corrections
from tutorials.collect_raws import collect_raw_prediction
from scripts.ourFuncs_task2 import generate_subset
from scripts.setup_ensemble import setup_ensemble


# load experiment configurations
trans_configs = load_from_json("../../src/configs/demo/athena-mnist.json")
model_configs = load_from_json("../../src/configs/demo/model-mnist.json")
data_configs = load_from_json("../../src/configs/demo/data-mnist.json")

output_dir = "../../ourDataFiles"
save_output = True

### setup the ensemble pool of weak defenses

# This wdList can be changed to a list of indexes of weak defenses in the 
#  athena-mnist.json file to get a custom set of weak defenses used in the
#   emsemble. Make sure to then set "customList" True and "useActi..." False
wdList=[]

useActiveList = True
customList = False

# run setup_ensemble to make an ensemble pool of weak defenses
athena = setup_ensemble(trans_configs=trans_configs,
                        model_configs=model_configs,
                        use_logits=False,
                        useActiveList=useActiveList,
                        customList=customList, wdList=wdList)

######################################################
#### collect the probabilities for the benign samples

# define the subset paramters
numberToSubset = 100
doRandom = True

bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
x_bs = np.load(bs_file)
print("benign sample data dimensions: {}".format(x_bs.shape))
totalNumData = x_bs.shape[0]
# generate subset indexes to grab benign samples
bs_subset, bs_subsetElse = generate_subset(totalSize=totalNumData,
                                           number=numberToSubset,doRandom=doRandom,
                                           odir='default',doSave=True)
x_bs = [x_bs[i] for i in bs_subset]
# grab predictions
preds = athena.predict(x=x_bs) # raw is False by default
print(">>> Shape of predictions: {}".format(preds.shape))

# save data for benign samples
if save_output:
    info_file = open(r"../../ourInfoSaves/infoFile_bs.txt","w")
    info_file.write("Info file for ensemble prediction of benign samples\n\n")
    info_file.write("numberToSubset: {}, doRandom: {}\n\n".format(
            numberToSubset, doRandom))
    info_file.write("useActiveList: {}\ncustomList: {}\nwdList: \n{}".format(
            str(useActiveList), str(customList), wdList) )
    info_file.close()
    
    np.save(output_dir+"/"+"ensemPredic_benign.npy",preds)







