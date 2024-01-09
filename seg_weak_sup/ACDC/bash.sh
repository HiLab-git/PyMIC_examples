
pymic_train config/unet2d_baseline.cfg
pymic_test  config/unet2d_baseline.cfg

pymic_train config/unet2d_dmpls.cfg
pymic_test  config/unet2d_dmpls.cfg

pymic_train config/unet2d_ustm.cfg
pymic_test  config/unet2d_ustm.cfg

pymic_eval_seg -cfg config/evaluation.cfg
