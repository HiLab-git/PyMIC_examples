export PYTHONPATH=$PYTHONPATH:/home/x/projects/PyMIC_project/PyMIC
nnFormer_path=/home/x/projects/pubcode/transformer/nnFormer
export PYTHONPATH=$PYTHONPATH:$PyMIC_path:$nnFormer_path

#python ../../../PyMIC/pymic/net_run/train.py  config/train_test_ce2.cfg
python ../../../PyMIC/pymic/net_run/predict.py  config/train_test_ce2.cfg
python ../../../PyMIC/pymic/util/evaluation_cls.py config/evaluation.cfg

#pymic_train config/train_test_ce2.cfg
#pymic_run test config/train_test_ce2.cfg
#pymic_eval_cls config/evaluation.cfg
