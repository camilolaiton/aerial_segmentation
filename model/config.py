import ml_collections

def get_config_test():
    """
        Returns the transformer configuration for testing
    """

    config = ml_collections.ConfigDict()
    config.config_name = "testing"
    config.dataset_path = 'dataset_3D_p64/'
    config.unbatch = True
    config.learning_rate = 0.001
    config.weight_decay = 1e-4
    config.momentum = 0.9
    config.dropout = 0.2
    config.batch_size = 16
    config.num_epochs = 100
    config.image_height = 64
    config.image_width = 64
    config.image_depth = 64
    config.image_channels = 1
    config.image_size = (config.image_height, config.image_width, config.image_depth, config.image_channels)
    config.skip_connections = True
    config.data_augmentation = False
    config.loss_fnc = 'dice_focal_loss' #'dice_focal_loss'#'focal_tversky'#'weighted_crossentropy'#'dice_focal_loss'#'tversky' #crossentropy
    config.decoder_conv_localpath = False
    config.decoder_conv_globalpath = False
    config.act_func = 'relu'

    config.transformer = ml_collections.ConfigDict()
    config.transformer.patch_size = 16
    config.transformer.num_patches = (config.image_size[0] // config.transformer.patch_size)**3
    
    config.transformer.projection_dim = 64 #128
    config.transformer.units = [
        config.transformer.projection_dim * 2, # (3) 1536 --  (4) 2048
        # config.transformer.projection_dim * 2,
        config.transformer.projection_dim, # 512
        # config.transformer.projection_dim, # 64
    ]
    config.transformer.layers = 4
    config.transformer.num_heads = 4 #8
    config.transformer.dropout_rate = 0.1
    config.transformer.normalization_rate = 1e-6

    config.decoder_filters = [
      config.transformer.projection_dim,
      64,
      32,
      16,
    ]

    config.n_classes = 4
    config.activation = 'softmax'

    return config