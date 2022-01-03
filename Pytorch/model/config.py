import ml_collections

def get_config():
    """
        Returns the transformer configuration for testing
    """

    config = ml_collections.ConfigDict()

    config.batch_size = 32
    config.num_epochs = 25
    config.image_height = 256
    config.image_width = 256
    config.image_channels = 3
    config.image_size = (config.image_height, config.image_width, config.image_channels)
    config.normalization_rate = 1e-4

    config.transformers = [
        {
            'dim': 32,
            'proj_kernel':3,
            'kv_proj_stride':2,
            'depth':1,
            'heads':1,
            'mlp_mult':4,
            'dropout':0.1
        },
        {
            'dim': 64,
            'proj_kernel':3,
            'kv_proj_stride':2,
            'depth':2,
            'heads':4,
            'mlp_mult':4,
            'dropout':0.1
        },
        {
            'dim': 128,
            'proj_kernel':3,
            'kv_proj_stride':2,
            'depth':6,
            'heads':4,
            'mlp_mult':4,
            'dropout':0.1
        },
    ]


    config.config_name = "testing"
    config.dataset_path = '../dataset_3D_37/'
    config.learning_rate = 0.00001
    config.optimizer = 'adam' #SGD, adam
    config.weight_decay = 1e-4
    config.momentum = 0.9
    config.dropout = 0.3


    config.num_classes = 2
    config.activation = 'softmax'

    return config
