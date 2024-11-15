# PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
PyMICPATH=/home/disk4t/projects/PyMIC_project/Pypi_test/PyMIC-dev
export PYTHONPATH=$PYTHONPATH:$PyMICPATH

# python $PyMICPATH/pymic/net_run/train.py  config/net_vitb16.cfg
# python $PyMICPATH/pymic/net_run/predict.py  config/net_vitb16.cfg
# python $PyMICPATH/pymic/net_run/predict.py  config/net_resnet18.cfg
# python $PyMICPATH/pymic/net_run/predict.py  config/net_vgg16.cfg
#python ../../../PyMIC/pymic/net_run/train.py test   config/net_resnet18.cfg
#python ../../../PyMIC/pymic/net_run/net_run.py train  config/net_vgg16.cfg
#python ../../../PyMIC/pymic/net_run/net_run.py test   config/net_vgg16.cfg
python $PyMICPATH/pymic/util/evaluation_cls.py -cfg config/evaluation.cfg

# python $PyMICPATH/pymic/net_run/train.py  config/net_vitb16.cfg -ckpt_dir model_test/vitb16 -iter_max 200
#python $PyMICPATH/pymic/net_run/train.py  config/net_resnet18.cfg \
#-ckpt_dir model_test/resnet18 -iter_max 200
#python $PyMICPATH/pymic/net_run/train.py  config/net_vgg16.cfg \
#-ckpt_dir model_test/vgg16 -iter_max 200
# python $PyMICPATH/pymic/net_run/predict.py  config/net_vitb16.cfg

#pymic_train config/net_vitb16.cfg \
#-ckpt_dir model_test/vitb16 -iter_max 200
#pymic_test config/net_vgg16.cfg
#pymic_eval_cls -cfg config/evaluation.cfg

