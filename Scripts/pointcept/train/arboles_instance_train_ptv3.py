_base_ = ["_base_/default_runtime.py"]

# --- GLOBAL CONFIGURATION ---
weight = None
resume = False
evaluate = False
test_only = False
seed = 456
save_path = 'exp/arboles_ptv3_prueba2' 
num_worker = 4
batch_size = 2 
batch_size_val = 1
batch_size_test = 1
epoch = 175
eval_epoch = 175

sync_bn = False 
enable_amp = True 
empty_cache = True 
find_unused_parameters = False
mix_prob = 0.0 
enable_wandb = False 
use_tensorboard = True 
gradient_accumulation_steps = 2
grad_max_norm = 5.0 

# --- OPTIMIZER (Adjusted for Outdoor Settings) ---
# The paper uses Weight Decay of 5e-3 for outdoor settings 
param_dicts = [dict(keyword='backbone', lr=0.0002)] 
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.005) # Changed from 0.05 to 0.005 
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0
)

# --- METHODS ---
train = dict(type='DefaultTrainer')
test = dict(type='InsSegTester', verbose=True)

# --- MODEL (PTv3 SOTA Architecture) ---
model = dict(
    type='PG-v1m2', 
    backbone=dict(
        type='PT-v3m1',
        in_channels=6,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),              # 4 downsampling steps
        enc_depths=(2, 2, 2, 6, 2),       # 5 levels (correct: 4+1)
        enc_channels=(32, 64, 128, 256, 512), # 5 levels
        enc_num_head=(2, 4, 8, 16, 32),   # 5 levels
        enc_patch_size=(1024, 1024, 1024, 1024, 1024), # 5 levels
        dec_depths=(1, 1, 1, 1),          # decoder length always matches stride length
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

    voxel_size=0.05,
    cluster_thresh=2.0,           # 10 cm radius (very strict, avoids over-merging)
    cluster_min_points=120,       # Raised slightly: forces denser clusters


    
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1, weight=[1.0, 1.0]),
        dict(type='LovaszLoss', mode='multiclass', loss_weight=1.0, ignore_index=-1),
    ]
)

# --- DATA ---
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
    # dict(
    #     type='InsSegEvaluator',
    #     segment_ignore_index=(0,), 
    #     instance_ignore_index=-1
    # ),
    dict(type='CheckpointSaver', save_freq=10),
]