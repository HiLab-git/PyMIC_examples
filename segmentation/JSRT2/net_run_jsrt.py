# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
from pymic.util.parse_config import parse_config
from pymic.net_run.agent_seg import  SegmentationAgent
from my_net2d import MyUNet2D 
from my_loss  import MyFocalDiceLoss

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)

    # use custormized CNN and loss function
    mynet  = MyUNet2D(config['network'])
    myloss = MyFocalDiceLoss(config['training'])
    agent  = SegmentationAgent(config, stage)
    agent.set_network(mynet)
    agent.set_loss(myloss)
    agent.run()

if __name__ == "__main__":
    main()
