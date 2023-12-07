# CUDA_VISIBLE_DEVICES=2 python main.py data/ --model t2t_vit_t_14 -b 100 --eval_checkpoint ckpts/81.7_T2T_ViTt_14.pth.tar
CUDA_VISIBLE_DEVICES=2,3 ./distributed_train.sh 2 data/ --model t2t_vit_7 -b 1024 --lr 1e-2 --weight-decay .03   --img-size 224
#MODIFIED: 10x lr, 8x batch size 
