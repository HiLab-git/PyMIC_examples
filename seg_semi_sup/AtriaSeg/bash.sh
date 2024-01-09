pymic_train config/unet3d_baseline.cfg
pymic_test  config/unet3d_baseline.cfg

pymic_train config/unet3d_em.cfg
pymic_test  config/unet3d_em.cfg

pymic_train config/unet3d_cps.cfg
pymic_test  config/unet3d_cps.cfg

pymic_eval_seg -cfg config/evaluation.cfg

