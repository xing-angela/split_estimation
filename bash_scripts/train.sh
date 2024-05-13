######### Halpe 26 Dataset #########
# DATA_DIR="/users/axing2/data/users/axing2/split_estimation/data/halpe"
# IMG_DIR="images/train2015"

############ LSP Dataset ############
DATA_DIR="/users/axing2/data/users/axing2/split_estimation/data/lsp"
IMG_DIR="images"

# BATCH_SIZE=8 # unipose
BATCH_SIZE=48
NUM_EPOCHS=50
IMG_WIDTH=192
IMG_HEIGHT=256
SAVE_EPOCH=5
EXP_NAME="lsp_data"
MODEL="AlphaPose"
DATASET="LSP"

python scripts/train.py \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --model $MODEL \
    --dataset $DATASET \
    --img_dir $IMG_DIR \
    --img_width $IMG_WIDTH \
    --img_height $IMG_HEIGHT \
    --save_epoch $SAVE_EPOCH \
    --exp_name $EXP_NAME \
    --plot_loss \
    --plot_acc \
    --from_epoch 145 \