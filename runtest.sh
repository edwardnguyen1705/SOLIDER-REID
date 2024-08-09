TRAIN_DS_NAME="ccdmmpsssss"
TEST_DS_NAME="ccdmmpsssss"

EVAL_ON_TEST_SET=False CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --config_file configs/$TEST_DS_NAME/swin_small.yml \
    TEST.WEIGHT log/$TRAIN_DS_NAME/swin_small_30e/transformer_30.pth \
    TEST.RE_RANKING False \
    MODEL.SEMANTIC_WEIGHT 0.2
