"""
Code pieces for collecting raw values from WDs on the input(s).
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import sys
sys.path.append("../")

from utils.model import load_pool
from models.athena import Ensemble, ENSEMBLE_STRATEGY


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


