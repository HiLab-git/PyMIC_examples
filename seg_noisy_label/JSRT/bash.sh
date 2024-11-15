
# PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
PyMICPATH=/home/disk4t/projects/PyMIC_project/Pypi_test/PyMIC-dev
export PYTHONPATH=$PYTHONPATH:$PyMICPATH

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_ce.cfg \
-ckpt_dir model_test/unet2d_ce -iter_max 400

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_nrdice.cfg \
-ckpt_dir model_test/unet2d_nrdice -iter_max 400

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_cot.cfg \
-ckpt_dir model_test/unet2d_cot -iter_max 400

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_dast.cfg \
-ckpt_dir model_test/unet2d_dast -iter_max 400

python $PyMICPATH/pymic/net_run/train.py  config/unet2d_trinet.cfg \
-ckpt_dir model_test/unet2d_trinet -iter_max 400

