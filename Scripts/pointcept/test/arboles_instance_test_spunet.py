_base_ = ["_base_/default_runtime.py"]

# --- CONFIGURACIÓN GLOBAL ---
weight = 'exp/spunet_Original_Original/model/model_best.pth'
resume = False
evaluate = True
test_only = True
seed = 456
save_path = 'exp/arboles_instance_test_spunet' # Carpeta para SpUNet
num_worker = 4 
batch_size = 2 
batch_size_val = 1
batch_size_test = 1
epoch = 500
eval_epoch = 500 

sync_bn = False 
enable_amp = True 
empty_cache = True 
enable_wandb = False 
use_tensorboard = True
gradient_accumulation_steps = 2

# --- OPTIMIZADOR (Basado en el ejemplo de SpUNet) ---
# Usamos SGD con momentum, que es el estándar para SparseUNet en Pointcept
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="PolyLR")

# --- MÉTODOS ---
train = dict(type='DefaultTrainer')
test = dict(type='InsSegTester', verbose=True)

# --- MODELO (SpUNet-v1m1 + PointGroup) ---
model = dict(
    type='PG-v1m2', 
    backbone=dict(
        type='SpUNet-v1m1',
        in_channels=6,      # XYZ + Intensidad
        num_classes=0,      # En SpUNet, 0 indica que solo queremos extraer rasgos
        channels=(32, 64, 128, 256, 256, 128, 96, 96), # Configuración ScanNet
        layers=(2, 3, 4, 6, 2, 2, 2, 2),               # Profundidad por nivel
    ),
    backbone_out_channels=96, #DEBE coincidir con el último valor de 'channels'
    semantic_num_classes=2,
    semantic_ignore_index=-1,
    segment_ignore_index=(0,),  
    instance_ignore_index=-1,
    
    voxel_size=0.05,
    cluster_thresh=2.0,           # Radio de 10 cm (Muy estricto, ideal para evitar fusiones)
    cluster_min_points=120,       # Subimos un poco: obliga a que los grupos sean más densos
    
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1, weight=[1.0, 1.0]),
        dict(type='LovaszLoss', mode='multiclass', loss_weight=1.0, ignore_index=-1),
    ]
)

# --- DATOS ---
data_root = 'data/tree_assets'
data = dict(
    num_classes=2,
    ignore_index=-1,
    names=["Background", "Tree"],
    train=dict(
        type='DefaultDataset',
        split='train',
        data_root=data_root,
        transform=[
            
            dict(type='RandomRotate', angle=[-1, 1], axis='z', p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='GridSample', grid_size=0.05, hash_type='fnv', mode='train', return_grid_coord=True),
            dict(type='InstanceParser', segment_ignore_index=(0,), instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(type='Collect', 
                 keys=('coord', 'grid_coord', 'segment', 'instance', 'instance_centroid', 'name'), 
                 feat_keys=('coord', 'color',), offset_keys_dict=dict(offset="coord")),
        ],
        loop=10
    ),
    val=dict(
        type='DefaultDataset',
        split='val',
        data_root=data_root,
        transform=[
            dict(type='GridSample', grid_size=0.05, hash_type='fnv', mode='train', return_grid_coord=True),
            dict(type='InstanceParser', segment_ignore_index=(0,), instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(type='Collect', 
                 keys=('coord', 'grid_coord', 'segment', 'instance', 'instance_centroid', 'name'), 
                 feat_keys=('coord', 'color',), offset_keys_dict=dict(offset="coord")),
        ]
    ),
    test=dict(
        type='DefaultDataset',
        split='test',
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },),
            
            dict(type='GridSample', grid_size=0.05, hash_type='fnv', mode='train', return_grid_coord=True, return_inverse=True),
            dict(type="NormalizeColor"),
            
            dict(type='InstanceParser', segment_ignore_index=(0,), instance_ignore_index=-1),
            
            dict(type='ToTensor'),
            dict(type='Collect', 
                 keys=('coord', 'grid_coord', 'segment', 'instance', 'instance_centroid', 
                       'origin_coord', 'origin_segment', 'origin_instance','name', 'inverse'), 
                 feat_keys=('coord','color',), offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),),
            
        ]
    )
)

test = dict(
    type="InsSegTester",
    verbose=True,
    segment_ignore_index=[0],   # Usamos corchetes para que sea una lista iterable
    instance_ignore_index=-1
)

# --- HOOKS ---
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(
        type='InsSegTester', 
        verbose=True,
        # 0 es el índice del fondo/background tras el MapIndex
        segment_ignore_index=(0,), 
        instance_ignore_index=-1 
        ),
    dict(type='CheckpointSaver', save_freq=10)]
