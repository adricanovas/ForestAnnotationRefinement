_base_ = ["_base_/default_runtime.py"]

# --- CONFIGURACIÓN GLOBAL ---
weight = 'exp/train_ptv3_p1024_lr0.002/model/model_best.pth'
resume = False
evaluate = True
test_only = True
seed = 456
save_path = 'exp/arboles_instance_final_test2' 
num_worker = 4
batch_size = 2
batch_size_val = 1
batch_size_test = 1
epoch = 200
eval_epoch = 200

sync_bn = False 
enable_amp = True 
empty_cache = True 
find_unused_parameters = False
mix_prob = 0.0 
enable_wandb = False 
use_tensorboard = True 
gradient_accumulation_steps = 2
grad_max_norm = 5.0 

# --- OPTIMIZADOR ---
param_dicts = [dict(keyword='backbone', lr=0.0002)] 
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0
)

# --- MÉTODOS ---
train = dict(type='DefaultTrainer')
test = dict(type='InsSegTester', verbose=True)


# --- MODELO (Configuración Binaria: Fondo vs Árbol) ---
model = dict(
    type='PG-v1m2', 
    backbone=dict(
        type='PT-v3m1',
        in_channels=6,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),              # 4 reducciones
        enc_depths=(2, 2, 2, 6, 2),       # 5 niveles (Correcto: 4+1)
        enc_channels=(32, 64, 128, 256, 512), # 5 niveles
        enc_num_head=(2, 4, 8, 16, 32),   # 5 niveles
        enc_patch_size=(1024, 1024, 1024, 1024, 1024), # 5 niveles
        dec_depths=(1, 1, 1, 1),          # El decoder siempre tiene la longitud de stride
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.3,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
    ),
    backbone_out_channels=64,
    semantic_num_classes=2,
    semantic_ignore_index=-1,
    segment_ignore_index=(0,),
    instance_ignore_index=-1,
    

    # Post-proceso para Voxel 0.05
    voxel_size=0.05,
    cluster_thresh=3.0,           # Radio de 10 cm (Muy estricto, ideal para evitar fusiones)
    cluster_min_points=900,       # Subimos un poco: obliga a que los grupos sean más densos
    
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
