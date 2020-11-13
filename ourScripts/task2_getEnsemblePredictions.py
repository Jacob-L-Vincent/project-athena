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
from scripts.ourFuncs_task2 import generate_subset
from scripts.setup_ensemble import setup_ensemble


# load experiment configurations
trans_configs = load_from_json("../src/configs/demo/athena-mnist_demoCopy.json")
model_configs = load_from_json("../src/configs/demo/model-mnist.json")
data_configs = load_from_json("../src/configs/demo/data-mnist.json")

output_dir = "../ourDataFiles/ensembleOuts"
save_output = True
verbose = 10

######################################################
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
### generate subset indexes for exmaples and save info file

# define the subset parameters
numberToSubset = 10000
doRandom = False
totalNumData = 10000

# generate subset indexes to grab benign samples
subset, subsetElse = generate_subset(totalSize=totalNumData,doSave=True,
                                     number=numberToSubset,doRandom=doRandom,
                                     opath=[r"../ourInfoSaves/ensPred_subset.npy",
                                            r"../ourInfoSaves/ensPred_subsetElse.npy"])

# save info in a text file
if save_output:
    info_file = open(r"../ourInfoSaves/infoFile_ensPred.txt","w")
    info_file.write("Info file for ensemble predictions\n\n")
    info_file.write("numberToSubset: {}, doRandom: {}\nsubset:\n".format(
            numberToSubset, doRandom))
    info_file.write("{}\n\n".format(subset))
    info_file.write("useActiveList: {}\ncustomList: {}\nwdList: \n{}\n\n".format(
            str(useActiveList), str(customList), wdList) )
    info_file.write("dimensions of raw npy arrays: wd, input, class")
    info_file.close()
    

############################################################################
## generate and collect probabilities of benign samples
bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
x_bs = np.load(bs_file)
if(verbose>5): print("\nbenign sample data dimensions: {}\n".format(x_bs.shape))
totalNumData = x_bs.shape[0]

x_bs = [x_bs[i] for i in subset]
# grab predictions
preds = athena.predict(x=x_bs) # raw is False by default
preds_raw = athena.predict(x=x_bs,raw=True)
if(verbose>5): print("\n>>> Shape of benign ensemble predictions: {}\n".format(preds.shape))

if save_output:
    np.save(output_dir+"/"+"ensemPredic_benign_raw.npy",preds_raw)
    np.save(output_dir+"/"+"ensemPredic_benign.npy",preds)



###########################################################################
### generate and collect the probabilities for our advers. examples
ae_dir, ae_files = data_configs.get('ae_dir'), data_configs.get('ae_files')

for ae_file in ae_files:
    ae_file1 = os.path.join(ae_dir, ae_file)
    x_ae = np.load(ae_file1)
    x_ae = [x_ae[i] for i in subset]
    # grab predictions
    preds = athena.predict(x=x_ae) # raw is False by default
    preds_raw = athena.predict(x=x_ae,raw=True)
    
    if save_output:
        np.save(output_dir+"/"+"ensemPredic_raw_{}".format(ae_file),preds_raw)
        np.save(output_dir+"/"+"ensemPredic_{}".format(ae_file),preds)
    
    if(verbose>5): print("\n>>> Shape of ae ensemble {} predictions: {}\n".format(ae_file,preds.shape))


