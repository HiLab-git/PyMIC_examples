export PYTHONPATH=$PYTHONPATH:/home/x/projects/PyMIC_project/PyMIC
nnFormer_path=/home/x/projects/pubcode/transformer/nnFormer
export PYTHONPATH=$PYTHONPATH:$PyMIC_path:$nnFormer_path
#export CUDA_VISIBLE_DEVICES=0


python ../../../PyMIC/pymic/net_run/predict.py config/coplenet.cfg
# python ../../../PyMIC/pymic/net/net2d/trans2d/transunet.py
#python ../../../PyMIC/pymic/net_run/predict.py config/unet.cfg
python ../../../PyMIC/pymic/util/evaluation_seg.py -cfg config/evaluation.cfg
#python ../../../PyMIC/pymic/util/evaluation_seg.py -metric dice -cls_index 255 -gt_dir ../../PyMIC_data/JSRT/label -seg_dir result/unet 

#pymic_train config/unet.cfg
#pymic_test  config/unet.cfg
#pymic_seg  config/evaluation.cfg
