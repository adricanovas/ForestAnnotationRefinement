_base_ = ["_base_/default_runtime.py"]

# --- GLOBAL CONFIGURATION ---
weight = 'exp/spunet_Original_Original/model/model_best.pth'
resume = False
evaluate = True
test_only = True
seed = 456
save_path = 'exp/arboles_instance_test_spunet' # SpUNet output folder
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

# --- OPTIMIZER (Based on the SpUNet example) ---
# SGD with momentum: the standard optimizer for SparseUNet in Pointcept
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="PolyLR")

# --- METHODS ---
train = dict(type='DefaultTrainer')
test = dict(type='InsSegTester', verbose=True)

# --- MODEL (SpUNet-v1m1 + PointGroup) ---
model = dict(
    type='PG-v1m2', 
    backbone=dict(
        type='SpUNet-v1m1',
        in_channels=6,      # XYZ + Intensity
        num_classes=0,      # 0 means feature extraction only (no classifier head) in SpUNet
        channels=(32, 64, 128, 256, 256, 128, 96, 96), # ScanNet configuration
        layers=(2, 3, 4, 6, 2, 2, 2, 2),               # Depth per level
    ),
    backbone_out_channels=96, # Must match the last value in 'channels'
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
    segment_ignore_index=[0],   # Square brackets make it a proper iterable
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
        # 0 is the background index after MapIndex
        segment_ignore_index=(0,), 
        instance_ignore_index=-1 
        ),
    dict(type='CheckpointSaver', save_freq=10)]
