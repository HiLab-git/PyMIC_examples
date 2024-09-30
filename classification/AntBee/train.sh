PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
# STUNetPATH=/home/x/projects/pubcode/STU-Net-main/nnUNet-1.7.1
export PYTHONPATH=$PYTHONPATH:$PyMICPATH
# #python ../../../PyMIC/pymic/net_run/train.py  config/train_test_ce2.cfg
# python $PyMICPATH/pymic/net_run/predict.py  config/train_test_ce2.cfg
python $PyMICPATH/pymic/util/evaluation_cls.py -cfg config/evaluation.cfg

#pymic_train config/train_test_ce2.cfg
# pymic_test config/train_test_ce2.cfg
# pymic_eval_cls config/evaluation.cfg
