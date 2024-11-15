export PYMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
export PYTHONPATH=$PYTHONPATH:$PYMICPATH

# python custom_run.py train config/mynet.cfg
python custom_run.py test config/mynet.cfg

# pymic_eval_seg -cfg config/evaluation.cfg
python $PYMICPATH/pymic/util/evaluation_seg.py -cfg config/evaluation.cfg
