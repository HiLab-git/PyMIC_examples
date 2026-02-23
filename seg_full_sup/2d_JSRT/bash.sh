PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
# PyMICPATH=/home/disk4t/projects/PyMIC_project/Pypi_test/PyMIC-dev
export PYTHONPATH=$PYTHONPATH:$PyMICPATH

# python $PyMICPATH/pymic/net_run/train.py  config/canet.cfg
# python $PyMICPATH/pymic/net_run/train.py  config/unet_attention.cfg
# python $PyMICPATH/pymic/net_run/train.py  config/unetpp.cfg
# python $PyMICPATH/pymic/net_run/train.py  config/unet.cfg
# python $PyMICPATH/pymic/net_run/train.py  config/coplenet.cfg \
# --ckpt_dir model_test/copplenet --iter_max 500
# python $PyMICPATH/pymic/net_run/train.py  config/swinunet.cfg
# python $PyMICPATH/pymic/net_run/train.py  config/transunet.cfg
# python $PyMICPATH/pymic/net_run/train.py  config/unet_scse.cfg
python $PyMICPATH/pymic/util/evaluation_seg.py --cfg config/evaluation.cfg

# python $PyMICPATH/pymic/net_run/train.py  config/unet.cfg \
# -ckpt_dir model_test/unet -iter_max 500

# python $PyMICPATH/pymic/net_run/train.py  config/unet_scse.cfg \
# -ckpt_dir model_test/unet -iter_max 500

#pymic_run train config/unet.cfg
#pymic_run test config/unet.cfg
#pymic_eval_seg config/evaluation.cfg

