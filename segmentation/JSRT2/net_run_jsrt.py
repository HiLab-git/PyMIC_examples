# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
from pymic.util.parse_config import parse_config
from pymic.net_run.agent_seg import  SegmentationAgent
from pymic.loss.loss_dict_seg import SegLossDict
from my_net2d import MyUNet2D 
from my_loss  import MyFocalDiceLoss

loss_dict = {'MyFocalDiceLoss':MyFocalDiceLoss}
loss_dict.update(SegLossDict)

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python net_run_jsrt.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)

    agent  = SegmentationAgent(config, stage)
    # use custormized CNN and loss function
    mynet  = MyUNet2D(config['network'])
    agent.set_network(mynet)
    agent.set_loss_dict(loss_dict)
    agent.run()

if __name__ == "__main__":
    main()
