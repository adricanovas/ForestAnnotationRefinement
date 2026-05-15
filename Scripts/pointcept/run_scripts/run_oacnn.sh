#!/bin/bash
# Automate OA-CNN setup with sed
DATASETS=("trees_0", "trees_25", "trees_50", "trees_100", "trees_ML")
TRAIN_CONF="configs/arboles_instance_train_oacnn.py"
TEST_CONF="configs/arboles_instance_test_oacnn.py"

for DS in "${DATASETS[@]}"
do
    echo "Configuring files for OA-CNN with dataset: $DS"
    
    # Modifies routes in TRAIN
    sed -i "s|data_root = '.*'|data_root = 'data/$DS'|g" $TRAIN_CONF
    sed -i "s|save_path = '.*'|save_path = 'exp/oacnn_$DS'|g" $TRAIN_CONF
    
    # Modifies routes in the TEST file
    sed -i "s|data_root = '.*'|data_root = 'data/$DS'|g" $TEST_CONF
    sed -i "s|save_path = '.*'|save_path = 'exp/oacnn_$DS/test_last'|g" $TEST_CONF
    sed -i "s|weight = '.*'|weight = 'exp/oacnn_$DS/model/model_last.pth'|g" $TEST_CONF

    echo "Starting TRAINING..."
    python tools/train.py --config-file $TRAIN_CONF --num-gpus 1
    
    echo "Starting TEST with model_last.pth..."
    python tools/test.py --config-file $TEST_CONF --num-gpus 1
done