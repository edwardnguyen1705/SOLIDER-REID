# Swin Base
# CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/msmt17/swin_base.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH 'path/to/SOLIDER/log/lup/swin_base/checkpoint_tea.pth' OUTPUT_DIR './log/msmt17/swin_base' SOLVER.BASE_LR 0.0002 SOLVER.OPTIMIZER_NAME 'SGD' MODEL.SEMANTIC_WEIGHT 0.2

DS_NAME="ccdmmpsssss"
# Swin Small
CUDA_VISIBLE_DEVICES=3 python3 train.py \
    --config_file configs/$DS_NAME/swin_small.yml \
    MODEL.PRETRAIN_CHOICE 'self' \
    MODEL.PRETRAIN_PATH checkpoints/swin_small_tea.pth\
    OUTPUT_DIR ./log/$DS_NAME/swin_small_30e \
    SOLVER.BASE_LR 0.0002 \
    SOLVER.OPTIMIZER_NAME 'SGD' \
    MODEL.SEMANTIC_WEIGHT 0.2

# Swin Tiny
#CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/msmt17/swin_tiny.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH 'path/to/SOLIDER/log/lup/swin_tiny/checkpoint_tea.pth' OUTPUT_DIR './log/msmt17/swin_tiny' SOLVER.BASE_LR 0.0008 SOLVER.OPTIMIZER_NAME 'SGD' MODEL.SEMANTIC_WEIGHT 0.2
