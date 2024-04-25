DATA_DIR="/users/axing2/data/users/axing2/split_estimation/data/halpe"
BATCH_SIZE=48
NUM_EPOCHS=200
IMG_DIR="images/train2015"
IMG_WIDTH=192
IMG_HEIGHT=256

python scripts/train.py \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --img_dir $IMG_DIR \
    --img_width $IMG_WIDTH \
    --img_height $IMG_HEIGHT \