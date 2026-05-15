#!/bin/bash
# Automatización SpUNet con sed
DATASETS=("tree_assets")
TRAIN_CONF="configs/arboles_instance_train_spunet.py"
TEST_CONF="configs/arboles_instance_test_spunet.py"

for DS in "${DATASETS[@]}"
do
    echo "Configurando archivos para SpUNet con dataset: $DS"
    
    # Modificar rutas en el archivo de TRAIN
    sed -i "s|data_root = '.*'|data_root = 'data/$DS'|g" $TRAIN_CONF
    sed -i "s|save_path = '.*'|save_path = 'exp/spunet_$DS'|g" $TRAIN_CONF
    
    # Modificar rutas en el archivo de TEST
    sed -i "s|data_root = '.*'|data_root = 'data/$DS'|g" $TEST_CONF
    sed -i "s|save_path = '.*'|save_path = 'exp/spunet_$DS/test_last'|g" $TEST_CONF
    sed -i "s|weight = '.*'|weight = 'exp/spunet_$DS/model/model_last.pth'|g" $TEST_CONF

    echo "Iniciando ENTRENAMIENTO..."
    python tools/train.py --config-file $TRAIN_CONF --num-gpus 1
    
    echo "Iniciando TEST con model_last.pth..."
    python tools/test.py --config-file $TEST_CONF --num-gpus 1
done