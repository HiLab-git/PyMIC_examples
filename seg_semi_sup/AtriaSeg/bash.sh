# PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
PyMICPATH=/home/disk4t/projects/PyMIC_project/Pypi_test/PyMIC-dev
export PYTHONPATH=$PYTHONPATH:$PyMICPATH


python $PyMICPATH/pymic/net_run/train.py  config/unet3d_r10_baseline.cfg \
-ckpt_dir model_test/unet3d_baseline -iter_max 200

python $PyMICPATH/pymic/net_run/train.py  config/unet3d_r10_urpc.cfg \
-ckpt_dir model_test/unet3d_urpc -iter_max 200

python $PyMICPATH/pymic/net_run/train.py  config/unet3d_r10_uamt.cfg \
-ckpt_dir model_test/unet2d_uamt -iter_max 200

python $PyMICPATH/pymic/net_run/train.py  config/unet3d_r10_cps.cfg \
-ckpt_dir model_test/unet2d_cps -iter_max 200



# pymic_train config/unet3d_r10_baseline.cfg
# pymic_test  config/unet3d_r10_baseline.cfg

# pymic_train config/unet3d_r10_em.cfg
# pymic_test  config/unet3d_r10_em.cfg

# pymic_train config/unet3d_r10_cps.cfg
# pymic_test  config/unet3d_r10_cps.cfg

# pymic_eval_seg -cfg config/evaluation.cfg

