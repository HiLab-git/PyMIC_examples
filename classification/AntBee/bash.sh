# PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
PyMICPATH=/home/disk4t/projects/PyMIC_project/Pypi_test/PyMIC-dev
export PYTHONPATH=$PYTHONPATH:$PyMICPATH

#python $PyMICPATH/pymic/net_run/train.py  config/train_test_ce1.cfg \
#-ckpt_dir model_test/resnet18_ce1 -iter_max 500
# python $PyMICPATH/pymic/net_run/predict.py  config/train_test_ce2.cfg
python $PyMICPATH/pymic/util/evaluation_cls.py -cfg config/evaluation.cfg

#pymic_train config/train_test_ce1.cfg \
#-ckpt_dir model_test/resnet18_ce1 -iter_max 500
#pymic_test config/train_test_ce1.cfg
#pymic_eval_cls -cfg config/evaluation.cfg
