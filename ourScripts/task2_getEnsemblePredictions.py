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
import pandas as pd
import os
from matplotlib import pyplot as plt

from utils.file import load_from_json
from scripts.ourFuncs_task2 import generate_subset
from scripts.setup_ensemble import setup_ensemble


# load experiment configurations
trans_configs = load_from_json("../src/configs/demo/athena-mnist_demoCopy.json")
model_configs = load_from_json("../src/configs/demo/model-mnist.json")
data_configs = load_from_json("../src/configs/demo/data-mnist.json")

output_dir = "../ourDataFiles"
save_output = True

### setup the ensemble pool of weak defenses

# This wdList can be changed to a list of indexes of weak defenses in the 
#  athena-mnist.json file to get a custom set of weak defenses used in the
#   emsemble. Make sure to then set "customList" True and "useActi..." False
#    both set to False makes it use all the transformations in trans_configs
wdList = []

useActiveList = False
customList = False

# run setup_ensemble to make an ensemble pool of weak defenses
athena = setup_ensemble(trans_configs=trans_configs,
                        model_configs=model_configs,
                        use_logits=False,
                        useActiveList=useActiveList,
                        customList=customList, wdList=wdList)

######################################################
#### collect the probabilities for the benign samples

# define the subset parameters
numberToSubset = 100
doRandom = True
totalNumData = 10000

# generate subset indexes to grab benign samples
subset, subsetElse = generate_subset(totalSize=totalNumData,doSave=True,
                                     number=numberToSubset,doRandom=doRandom,
                                     opath='default')

bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
x_bs = np.load(bs_file)
print("benign sample data dimensions: {}".format(x_bs.shape))
totalNumData = x_bs.shape[0]

x_bs = [x_bs[i] for i in subset]
# grab predictions
preds = athena.predict(x=x_bs) # raw is False by default
preds_raw = athena.predict(x=x_bs,raw=True)
print(">>> Shape of benign ensemble predictions: {}".format(preds.shape))

# save data for benign samples
if save_output:
    info_file = open(r"../ourInfoSaves/infoFile_bs.txt","w")
    info_file.write("Info file for ensemble prediction of benign samples\n\n")
    info_file.write("numberToSubset: {}, doRandom: {}\n\n".format(
            numberToSubset, doRandom))
    info_file.write("useActiveList: {}\ncustomList: {}\nwdList: \n{}".format(
            str(useActiveList), str(customList), wdList) )
    info_file.close()
    
    np.save(output_dir+"/"+"ensemPredic_benign.npy",preds)


###########################################################################
### generate and collect the probabilities for our advers. examples
ae_dir, ae_files = trans_configs.get('ae_dir'), trans_configs.get('ae_files')
ae_subset = [ae_files[i] for i in range(0,len(ae_files),3)]
results = pd.DataFrame(columns=[")





new_row = {"ae":fileName.replace(".npy",","), "UM":err_um, 
                   "BL":err_bl, "Ensemble":err_ens}
        results = results.append(new_row,ignore_index=True)



###############################################
### ran this once to generate subset of aversarial examples
###  I copied the results into wd_list so that it was static
num_trans = trans_configs.get('num_transformations')

dummy,_ = generate_subset(totalSize=num_trans,doSave=True,
                        number=30,doRandom=doRandom, 
                        opath=[r"../ourInfoSaves/wd_list_subset.npy",
                               r"../ourInfoSaves/wd_list_subsetElse.npy"] )
print(dummy)











