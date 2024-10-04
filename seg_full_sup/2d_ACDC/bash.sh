# PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
PyMICPATH=/home/disk4t/projects/PyMIC_project/Pypi_test/PyMIC-dev
export PYTHONPATH=$PYTHONPATH:$PyMICPATH

# python $PyMICPATH/pymic/net_run/train.py config/unet_scse.cfg
# python $PyMICPATH/pymic/net_run/train.py config/unetpp.cfg


# python ../../../PyMIC/pymic/util/evaluation_seg.py -metric [dice,assd] -cls_num 4 -gt_dir /home/disk4t/data/heart/ACDC/preprocess/ -seg_dir result -name_pair ./config/data/image_test_gt_seg.csv
# python $PyMICPATH/pymic/net_run/predict.py config/unetpp.cfg

python $PyMICPATH/pymic/net_run/train.py config/unetpp.cfg \
-ckpt_dir model_test/unetpp -iter_max 500

python $PyMICPATH/pymic/net_run/train.py config/unet_scse.cfg \
-ckpt_dir model_test/unet_scse -iter_max 500

# python $PyMICPATH/pymic/util/evaluation_seg.py -cfg config/evaluation.cfg

# pymic_test config/unet.cfg
# pymic_eval_seg config/evaluation.cfg