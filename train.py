import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils import utils
from model.config import *
from model.model import *
from model.losses import *
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import glob
import pickle
import segmentation_models as sm
sm.set_framework('tf.keras')

def main():

    # Selecting cuda device

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    tf.keras.backend.clear_session()
    SEED = 12
    mb_limit = 9500
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 10GB of memory on the GPU
        try:
            # Setting visible devices
            tf.config.set_visible_devices(gpus, 'GPU')

            # Setting memory growth
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_memory_growth(gpus[1], True)

            # Setting max memory
            # tf.config.experimental.set_per_process_memory_fraction(0.80)
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mb_limit)])

            tf.config.experimental.set_virtual_device_configuration(gpus[1], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mb_limit)])

            # tf.config.experimental.set_per_process_memory_growth(True)

        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    retrain = False
    training_folder = 'trainings/version_30'
    model_path = f"{training_folder}/model_trained_architecture.hdf5"
    # model_path = f"{training_folder}/checkpoints_4/model_trained_09_0.68.hdf5"

    utils.create_folder(f"{training_folder}/checkpoints")

    # creating model
    # config = get_config_patchified()
    config = get_config_test()
    model = None

    # Mirrored strategy for parallel training
    mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Setting up weights 
    weights = utils.read_test_to_list(config.dataset_path + 'weights.txt')

    if (weights == False):
        end_path = '/train/masks'
        image_files = [file for file in glob.glob(config.dataset_path + end_path + '/*') if file.endswith('.npy')]
        weights, label_to_frequency_dict = utils.median_frequency_balancing(image_files=image_files, num_classes=4)
        if (weights == False):
            print("Please check the path")
            exit()
        utils.write_list_to_txt(weights, config.dataset_path + 'weights.txt')
        print("Weights calculated: ", weights)
    else:
        weights = [float(weight) for weight in weights]
        # weights = [0.0, 1, 2.7, 3]
        # weights = [0.0, 2.3499980585022096, 6.680915101433645, 7.439929426050408]
        print("Weights read! ", weights)

    # Setting up neural network loss
    #loss = tversky_loss()#
    
    with mirrored_strategy.scope():
    # model = build_model_patchified_patchsize16(config)
    # model = build_model_patchified_patchsize16(config)
        model = test_model_3(config)
    
    if (retrain):
        model.load_weights(model_path)

    loss = None
    if config.loss_fnc == 'tversky':
        loss = tversky_loss
        print('Using tversky loss...')
    elif config.loss_fnc == 'crossentropy':
        loss = 'categorical_crossentropy'
        print('Using categorical crossentropy loss...')
    elif config.loss_fnc == 'dice_focal_loss':
        loss = dice_focal_loss(weights)
        print('Using dice focal loss...')
    elif config.loss_fnc == 'weighted_crossentropy':
        loss = weighted_categorical_crossentropy(weights)
        print("Using weighted crossentropy")
    elif config.loss_fnc == 'gen_dice':
        loss = generalized_dice_loss(weights)
    elif config.loss_fnc == 'focal_tversky':
        loss = focal_tversky
    elif config.loss_fnc == 'dice_categorical':
        loss = dice_categorical(weights)
    else:
        print("No loss function")
        exit()
    
    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    #     return lr
    

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     config.learning_rate,
    #     decay_steps=5,
    #     decay_rate=0.01,
    #     staircase=True
    # )

    optimizer = tf.optimizers.SGD(
        learning_rate=config.learning_rate, 
        momentum=config.momentum,
        name='optimizer_SGD_0'
    )

    # optimizer = tf.optimizers.Adam(
    #     learning_rate=config.learning_rate, 
    #     name='optimizer_Adam'
    # )

    # lr_metric = get_lr_metric(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            # 'accuracy',
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
        ],
    )
    
    print(f"[+] Building model with config {config}")
    model.summary()
    
    tf.keras.utils.plot_model(
        model,
        to_file=f"{training_folder}/trained_architecture.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )
    
    # Setting up variables for data generators
    # TRAIN_IMGS_DIR = config.dataset_path + 'train/images/'
    # TRAIN_MSKS_DIR = config.dataset_path + 'train/masks/'

    # TEST_IMGS_DIR = config.dataset_path + 'test/images/'
    # TEST_MSKS_DIR = config.dataset_path + 'test/masks/'

    # train_imgs_lst = os.listdir(TRAIN_IMGS_DIR)
    # train_msks_lst = os.listdir(TRAIN_MSKS_DIR)

    # test_imgs_lst = os.listdir(TEST_IMGS_DIR)
    # test_msks_lst = os.listdir(TEST_MSKS_DIR)

    image_list_train = sorted(glob.glob(
        config.dataset_path + 'train/images/*'))
    mask_list_train = sorted(glob.glob(
        config.dataset_path + 'train/masks/*'))
    print(config.dataset_path, " ", len(image_list_train), " ", len(mask_list_train))
    image_list_test = sorted(glob.glob(
        config.dataset_path + 'test/images/*'))
    mask_list_test = sorted(glob.glob(
        config.dataset_path + 'test/masks/*'))

    # Getting image data generators
    # train_datagen = utils.mri_generator(
    #     TRAIN_IMGS_DIR,
    #     train_imgs_lst,
    #     TRAIN_MSKS_DIR,
    #     train_msks_lst,
    #     config.batch_size
    # )

    # reading for training
    # half = int(len(image_list_train)*0.5)
    # train_imgs = utils.read_files_from_directory(image_list_train, half)
    # train_msks = utils.read_files_from_directory(mask_list_train, half)

    # Reading for validation
    # test_imgs = utils.read_files_from_directory(image_list_test)
    # test_msks = utils.read_files_from_directory(mask_list_test)

    train_datagen = tf.data.Dataset.from_tensor_slices(
        (image_list_train, mask_list_train)
    )

    # val_datagen = utils.mri_generator(
    #     TEST_IMGS_DIR,
    #     test_imgs_lst,
    #     TEST_MSKS_DIR,
    #     test_msks_lst,
    #     config.batch_size
    # )
    weights = [0, 0.2, 0.4, 0.4]
    val_datagen = tf.data.Dataset.from_tensor_slices(
        # (test_imgs, test_msks)
        (image_list_test, mask_list_test)
    )

    dataset = {
        "train" : train_datagen,
        "val" : val_datagen
    }

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset['train'] = dataset['train'].map(load_files)#.map(augmentor, num_parallel_calls=AUTOTUNE)
    # dataset['train'] = dataset['train'].shuffle(buffer_size=config.batch_size, seed=SEED)
    if (config.unbatch):
        dataset['train'] = dataset['train'].unbatch()
    dataset['train'] = dataset['train'].repeat()
    # dataset['train'] = dataset['train'].shuffle(config.batch_size, reshuffle_each_iteration=True)
    dataset['train'] = dataset['train'].batch(config.batch_size)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)
    dataset['train'] = dataset['train'].with_options(options)

    dataset['val'] = dataset['val'].map(load_files)
    if (config.unbatch):
        dataset['val'] = dataset['val'].unbatch()
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(config.batch_size)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)
    dataset['val'] = dataset['val'].with_options(options)
    
    # Setting up callbacks
    monitor = 'val_iou_score'
    mode = 'max'

    # Early stopping
    early_stop = EarlyStopping(
        monitor=monitor, 
        mode=mode, 
        verbose=1,
        patience=20
    )

    # Model Checkpoing
    model_check = ModelCheckpoint(
        f"{training_folder}/model_trained_architecture.hdf5", 
        save_best_only=True,
        save_weights_only=True, 
        monitor=monitor, 
        mode=mode
    )

    model_check_2 = ModelCheckpoint(
        training_folder + "/checkpoints/model_trained_{epoch:02d}_{val_iou_score:.2f}_{val_f1-score:.2f}.hdf5", 
        save_best_only=False,
        save_weights_only=True, 
        monitor=monitor, 
        mode=mode,
        period=1
    )

    tb = TensorBoard(
        log_dir=f"{training_folder}/logs_tr", 
        write_graph=True, 
        update_freq='epoch'
    )

    pltau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=2, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
    )

    factor = 1

    if (config.unbatch):
        factor = 64

    steps_per_epoch = (len(image_list_train)*factor)//config.batch_size
    val_steps_per_epoch = (len(image_list_test)*factor)//config.batch_size

    utils.write_dict_to_txt(
        config, 
        f"{training_folder}/trained_architecture_config.txt"
    )
    # class_weights = {
    #     0: weights[0],
    #     1: weights[1],
    #     2: weights[2],
    #     3: weights[3],
    # }

    history = model.fit(dataset['train'],
        steps_per_epoch=steps_per_epoch,
        epochs=config.num_epochs,
        # batch_size=config.batch_size,
        verbose=1,
        validation_data=dataset['val'],
        validation_steps=val_steps_per_epoch,
        callbacks=[early_stop, model_check, model_check_2, tb, pltau],
        # class_weight=class_weights
    )

    with open(f"{training_folder}/history.obj", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == "__main__":
    main()