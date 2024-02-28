

CUDA_VISIBLE_DEVICES=3

(

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python feature_stage1.py  && \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python feature_stage2.py  && \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python feature_stage3.py  && \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python feature_stage4.py  && \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python feature_stage5.py   \

) &


# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18_224 --dataset imagenet_lt --epochs ${epochs}   && \