#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:33:59 2020

@author: isaac
"""


from utils.file import load_from_json

output_dir = ""
attack_configs = load_from_json("/home/isaac/working_directory/misc/project-athena/src/configs/demo/attack-zk-mnist.json")


num_attacks = attack_configs.get("num_attacks")
out_fnames = open('/home/isaac/working_directory/misc/project-athena/generated_aes.txt',"w+")
out_fnames.write('dir = "{}"'.format(output_dir)+"\n")
for id in range(num_attacks):
    key = "configs{}".format(id)
    out_fnames.write('        "{}.npy",'.format(attack_configs.get(key).get("description"))+"\n")
            
            
out_fnames.close()