# export CUDA_VISIBLE_DEVICES=0
#pymic_train config/unet2d_cps.cfg
# pymic_test config/unet2d_baseline.cfg
# pymic_test config/unet2d_cps.cfg
# pymic_test config/unet2d_cct.cfg
# pymic_test config/unet2d_mt.cfg
# pymic_test config/unet2d_mcnet.cfg
# pymic_test config/unet2d_urpc.cfg
# pymic_test config/unet2d_em.cfg
pymic_test config/unet2d_uamt.cfg
#pymic_eval_seg config/evaluation.cfg

# python ../../../PyMIC/pymic/net_run_ssl/ssl_main.py train config/unet2d_cps.cfg
# python ../../../PyMIC/pymic/net_run_ssl/ssl_main.py test config/unet2d_cps.cfg
# python ../../../PyMIC/pymic/net_run/train.py config/unet2d_baseline.cfg
# python ../../../PyMIC/pymic/net_run/predict.py config/unet2d_cct.cfg
# python ../../../PyMIC/pymic/net_run/predict.py config/unet2d_cps.cfg
# python ../../../PyMIC/pymic/net_run/train.py config/unet2d_cct.cfg
# python ../../../PyMIC/pymic/net_run/predict.py config/unet2d_mt.cfg
# python $PYMICPATH/pymic/net_run/predict.py config/unet2d_uamt.cfg
# python ../../../PyMIC/pymic/net_run/predict.py config/unet2d_urpc.cfg
# python ../../../PyMIC/pymic/util/evaluation_seg.py -cfg config/evaluation.cfg

# python ../../../PyMIC/pymic/net/net2d/unet2d.py 