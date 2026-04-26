import yaml

path_yaml = '/home/kerem/AOI/AOI-AI/implementations/ssd_models/custom_architecture/sweep_config.yaml'

sweep_config = {
    'method': 'bayes',  # You can choose 'grid', 'random', 'bayes', etc., depending on your needs.
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': {
        'filters1': {'values': [32, 64]},
        'filters2': {'values': [64, 128]},
        'filters3': {'values': [128, 256]},
        'filters4_5': {'values': [256, 512]},
        'filters6': {'values': [512, 1024]},
        'neg_pos_ratio': {'values': [2, 3, 4]},
        'alpha': {'values':[1.0, 2.0, 3.0, 4.0]},
        'kernels': {'values': [3, 5]},
        'detection_head_size_class': {'values': [1, 3, 5]},
        'detection_head_size_offset': {'values': [1, 3, 5]},
        'use_bias': {'values': [True, False]},
        'epochs': {'values': [75, 100, 125]},
        'l2_reg': {'values': [0.0, 0.1, 0.001, 0.01]},
        'initial_lr': {'values': [0.1, 0.01, 0.001]},
        'gaussian_noise': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3}
    },
    'early_terminate': {'type': 'hyperband', 'min_iter': 5}
}

with open(path_yaml, 'w') as yaml_file:
    yaml.dump(sweep_config, yaml_file)
