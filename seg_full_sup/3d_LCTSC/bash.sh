# PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
PyMICPATH=/home/disk4t/projects/PyMIC_project/Pypi_test/PyMIC-dev
export PYTHONPATH=$PYTHONPATH:$PyMICPATH

# python $PyMICPATH/pymic/net_run/train.py  config/unet.cfg
# python $PyMICPATH/pymic/net_run/predict.py  config/unet.cfg
# python $PyMICPATH/pymic/util/evaluation_seg.py -cfg config/evaluation.cfg

python $PyMICPATH/pymic/net_run/train.py  config/lcovnet.cfg \
-ckpt_dir model_test/lcovnet -iter_max 500

python $PyMICPATH/pymic/net_run/train.py  config/unet2d5.cfg \
-ckpt_dir model_test/unet2d5 -iter_max 500

#pymic_run train config/unet.cfg
#pymic_run test config/unet.cfg
#pymic_eval_seg config/evaluation.cfg

