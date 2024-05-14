IMG_DIR="result/lsp/split/joint_vis"
OUT_DIR="result/lsp/split/angle_calc"
JOINT_DIR="result/lsp/split/joints"

python utils/calculate_angle.py \
    --img_dir $IMG_DIR \
    --output_path $OUT_DIR \
    --joint_dir $JOINT_DIR