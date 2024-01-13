pymic_train config/unet3d_r10_baseline.cfg
pymic_test  config/unet3d_r10_baseline.cfg

pymic_train config/unet3d_r10_em.cfg
pymic_test  config/unet3d_r10_em.cfg

pymic_train config/unet3d_r10_cps.cfg
pymic_test  config/unet3d_r10_cps.cfg

pymic_eval_seg -cfg config/evaluation.cfg

