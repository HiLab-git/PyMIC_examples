# PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
PyMICPATH=/home/disk4t/projects/PyMIC_project/Pypi_test/PyMIC-dev
export PYTHONPATH=$PYTHONPATH:$PyMICPATH

# python $PyMICPATH/pymic/net_run/train.py  config/unet.cfg
# python $PyMICPATH/pymic/net_run/predict.py  config/unet.cfg
# python $PyMICPATH/pymic/util/evaluation_seg.py -cfg config/evaluation.cfg

# python $PyMICPATH/pymic/net_run/train.py  config/unet3d.cfg \
# -ckpt_dir model_test/unet3d -iter_max 250

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_baseline.cfg \
-ckpt_dir model_test/unet2d_baseline -iter_max 400

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_cct.cfg \
-ckpt_dir model_test/unet2d_cct -iter_max 400

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_cps.cfg \
-ckpt_dir model_test/unet2d_cps -iter_max 400

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_mcnet.cfg \
-ckpt_dir model_test/unet2d_mcnet -iter_max 400

#pymic_run train config/unet.cfg
#pymic_run test config/unet.cfg
#pymic_eval_seg config/evaluation.cfg

#pymic_train config/unet2d_cps.cfg
# pymic_test config/unet2d_baseline.cfg
# pymic_test config/unet2d_cps.cfg
# pymic_test config/unet2d_cct.cfg
# pymic_test config/unet2d_mt.cfg
# pymic_test config/unet2d_mcnet.cfg
# pymic_test config/unet2d_urpc.cfg
# pymic_train config/unet2d_em.cfg
# pymic_test config/unet2d_em.cfg
# pymic_train config/unet2d_uamt.cfg
#pymic_eval_seg config/evaluation.cfg
