
# PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
PyMICPATH=/home/disk4t/projects/PyMIC_project/Pypi_test/PyMIC-dev
export PYTHONPATH=$PYTHONPATH:$PyMICPATH


python $PyMICPATH/pymic/net_run/train.py  config/unet2d_baseline.cfg \
-ckpt_dir model_test/unet2d_baseline -iter_max 200

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_dmpls.cfg \
-ckpt_dir model_test/unet2d_dmpls -iter_max 200

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_ustm.cfg \
-ckpt_dir model_test/unet2d_ustm -iter_max 200

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_gcrf.cfg \
-ckpt_dir model_test/unet2d_ustm -iter_max 200

# pymic_train config/unet2d_baseline.cfg
# pymic_test  config/unet2d_baseline.cfg

# pymic_train config/unet2d_dmpls.cfg
# pymic_test  config/unet2d_dmpls.cfg

# pymic_train config/unet2d_ustm.cfg
# pymic_test  config/unet2d_ustm.cfg

# pymic_eval_seg -cfg config/evaluation.cfg
