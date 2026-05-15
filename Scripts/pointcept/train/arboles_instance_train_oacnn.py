_base_ = ["_base_/default_runtime.py"]

# --- CONFIGURACIÓN GLOBAL ---
weight = None
resume = False
evaluate = True
test_only = False
seed = 456
save_path = 'exp/arboles_instance_oacnns' 
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
grad_max_norm = 5.0

# --- OPTIMIZADOR ---
# Usamos el LR del ejemplo (0.001) que es el estándar para OA-CNNs
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.02)
scheduler = dict(
    type='OneCycleLR',
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0
)

# --- MÉTODOS ---
train = dict(type='DefaultTrainer')
test = dict(type='InsSegTester', verbose=True)

# --- MODELO (OA-CNNs adaptado a PointGroup) ---
model = dict(
    type='PG-v1m2', 
    backbone=dict(
        type="OACNNs",
        in_channels=6,      # XYZ + Color/Intensidad
        num_classes=0,      # Se mantiene en 0 para que PointGroup maneje las cabezas
        embed_channels=64,  # Valor estándar en todas las versiones
        enc_channels=[64, 64, 128, 256], # 4 etapas de codificación 
        groups=[4, 4, 8, 16],
        # --- VERSIÓN (S) ---
        enc_depth=[2, 2, 2, 2],          # Solo 2 bloques por etapa para máxima velocidad 
        # -------------------------------------
        dec_channels=[256, 256, 256, 256], # Canales del decodificador alineados a las 4 etapas
        dec_depth=[1, 1, 1, 1],          # El decodificador de OA-CNN es extremadamente ligero
        point_grid_size=[[8, 12, 16, 16], [6, 9, 12, 12], [4, 6, 8, 8], [3, 4, 6, 6]],
        enc_num_ref=[16, 16, 16, 16],
    ),
    backbone_out_channels=256, # Coincide con el último dec_channel de la OA-CNN
    semantic_num_classes=2,      
    semantic_ignore_index=-1,
    segment_ignore_index=(0,),   
    instance_ignore_index=-1,
    

    voxel_size=0.05,
    cluster_thresh=2.0,           # Radio de 10 cm (Muy estricto, ideal para evitar fusiones)
    cluster_min_points=120,       # Subimos un poco: obliga a que los grupos sean más densos
    
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1, weight=[1.0, 2.0]),
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
            dict(type="CenterShift", apply_z=True),
            dict(type='RandomRotate', angle=[-1, 1], axis='z', p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(type='RandomFlip', p=0.5),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type='GridSample', grid_size=0.05, hash_type='fnv', mode='train', return_grid_coord=True),
            dict(type="NormalizeColor"),
            dict(type='InstanceParser', segment_ignore_index=(0,), instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(type='Collect', 
                 keys=('coord', 'grid_coord', 'segment', 'instance', 'instance_centroid', 'name'), 
                 feat_keys=('coord' ,'color',))
        ],
    ),
    val=dict(
        type='DefaultDataset',
        split='val',
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type='GridSample', grid_size=0.05, hash_type='fnv', mode='train', return_grid_coord=True),
            dict(type='InstanceParser', segment_ignore_index=(0,), instance_ignore_index=-1),
            dict(type="NormalizeColor"),
            dict(type='ToTensor'),
            dict(type='Collect', 
                 keys=('coord', 'grid_coord', 'segment', 'instance', 'instance_centroid', 'name'), 
                 feat_keys=('coord' ,'color',))
        ]
    ),
)

# --- HOOKS ---
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    # dict(
    #     type='SemSegEvaluator', 
    #     write_cls_iou=True
    # ),
    dict(
        type='InsSegEvaluator',
        segment_ignore_index=(0,), 
        instance_ignore_index=-1
    ),
    dict(type='CheckpointSaver', save_freq=10),
]