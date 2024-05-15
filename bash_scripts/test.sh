IMG_DIR="data/ap_test"
# OUT_DIR="result/lsp/"
# CHECKPOINT="exp/lsp_data/model_145.pth"
# TRAIN_DATASET="LSP"
OUT_DIR="result/halpe"
CHECKPOINT="exp/5/model_final.pth"
TRAIN_DATASET="Halpe26"

python scripts/test.py \
    --img_dir $IMG_DIR \
    --output_path $OUT_DIR \
    --train_dataset $TRAIN_DATASET \
    --checkpoint $CHECKPOINT \
    # --save_joints