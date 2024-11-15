# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import logging
import os
import sys
from pymic.util.parse_config import *
from pymic.net_run.agent_seg import  SegmentationAgent
from pymic.loss.loss_dict_seg import SegLossDict
from my_net2d import MyUNet2D 
from my_loss  import MyFocalDiceLoss

loss_dict = {'MyFocalDiceLoss':MyFocalDiceLoss}
loss_dict.update(SegLossDict)

def main():
    if(len(sys.argv) < 2):
        print('Number of arguments should be at least 3. e.g.')
        print('   python custom_run.py train config.cfg')
        exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", help="stage of train or test")
    parser.add_argument("cfg", help="configuration file")
    args = parser.parse_args()
    if(not os.path.isfile(args.cfg)):
        raise ValueError("The config file does not exist: " + args.cfg)
    config   = parse_config(args)
    config   = synchronize_config(config)

    log_dir  = config['training']['ckpt_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir)
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(args.stage), level=logging.INFO,
                            format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(args.stage), level=logging.INFO,
                            format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)

    agent  = SegmentationAgent(config, args.stage)
    # use custormized network and loss function
    mynet  = MyUNet2D(config['network'])
    agent.set_network(mynet)
    agent.set_loss_dict(loss_dict)
    agent.run()

if __name__ == "__main__":
    main()
