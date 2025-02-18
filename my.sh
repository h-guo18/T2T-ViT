# CUDA_VISIBLE_DEVICES=2 python main.py data/ --model t2t_vit_t_14 -b 100 --eval_checkpoint ckpts/81.7_T2T_ViTt_14.pth.tar
# CUDA_VISIBLE_DEVICES=2,3 ./distributed_train.sh 2 data/ --model t2t_vit_7 -b 1024 --lr 1e-2 --weight-decay .03   --img-size 224
CUDA_VISIBLE_DEVICES=3 ./distributed_train.sh 1 data/ --model t2t_vit_7 -b 128 --lr 1e-3 --weight-decay .03   --img-size 224 
#MODIFIED: 10x lr, 8x batch size 
# CUDA_VISIBLE_DEVICES=2 speed_test.py
