import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
from utils import utils
from torch.utils.data import DataLoader
from model.config import get_config, get_config_encoder
from model.network import CvTModified, CvT, CvT_Vgg11
from model.losses import *
from model.MassachusettsDataset import *
from tqdm import tqdm
from time import sleep
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
import numpy as np
import random
import os
import pandas as pd
import albumentations as album
import matplotlib.pyplot as plt

palette = np.array([
    [  0,   0,   0],   # black
    [255,   255,   255],
])

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

def load_checkpoint(checkpoint, model, optimizer):
    print("[+] Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("[+] Saving checkpoint")
    torch.save(state, filename)

def eval_step(model, test_loader, device, metric_collection, writer, epoch, dest_path, save_img=True):
    f1 = []
    # accuracy = []
    # precision = []
    # recall = []

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit='batch', position=0, leave=True) as tbatch:
            for idx, (imgs, msks) in enumerate(tbatch):
                imgs = imgs.to(device)
                msks = msks.to(device)

                pred = model(imgs)

                pred_argmax = torch.argmax(pred, dim=1)
                mask_argmax = torch.argmax(msks, dim=1)
                
                metrics = metric_collection(
                    pred_argmax, 
                    mask_argmax
                )

                if save_img and metrics['F1'].item() >= 0.80:
                    # imgs_np = (imgs.permute(0, 2,3,1)).cpu().numpy()
                    # imgs_np = (imgs_np * 255).astype(np.uint8)
                    
                    pred_np = pred_argmax.cpu().numpy()
                    msk_np = mask_argmax.cpu().numpy()
                    
                    # Squeezing in first dim
                    # imgs_np = np.squeeze(imgs_np, axis=0)
                    pred_np = np.squeeze(pred_np, axis=0)
                    msk_np = np.squeeze(msk_np, axis=0)

                    pred_img = palette[pred_np]
                    msk_img = palette[msk_np]
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.suptitle("Original mask VS Predicted")
                    ax1.imshow(msk_img)
                    ax1.set_title(f"mask image")
                    ax2.imshow(pred_img)
                    ax2.set_title(f"{round(metrics['F1'].item(), 2)} - epoch {epoch}")
                    # print(dest_path)
                    plt.savefig(f"{dest_path}/image_{idx}_F1_{round(metrics['F1'].item(), 2)}_epoch_{epoch}.png")

                # print("[INFO] Unique labels in this prediction: ", torch.unique(pred_argmax))

                # accuracy.append(metrics['Accuracy'].item())
                f1.append(metrics['F1'].item())
                # recall.append(metrics['Recall'].item())
                # precision.append(metrics['Precision'].item())

                tbatch.set_description("Testing")
                tbatch.set_postfix({
                    'Batch': f"{idx+1}",
                    # 'Accuracy': np.mean(accuracy),
                    'F1': np.mean(f1),
                    # 'Recall': np.mean(recall),
                    # 'Precision': np.mean(precision),
                })
                tbatch.update()
                sleep(0.01)

        writer.add_scalar('F1_Score/test', np.mean(f1), epoch)
    model.train()

def train(config:dict, load_model:bool, save_model:bool, training_folder:str, train_loader, test_loader):

    model_path = f"{training_folder}/model_trained_architecture.pt"

    # Creating folders for saving trainings
    utils.create_folder(f"{training_folder}/checkpoints")
    
    utils.write_dict_to_txt(
        config, 
        f"{training_folder}/trained_architecture_config.txt"
    )
    
    writer = SummaryWriter(training_folder)
    step = 0

    model = CvTModified(config=config) # CvT_Vgg11(config)
    print(model)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs!")
        # model = torch.nn.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
        model = torch.nn.DataParallel(model)

    model.to(device)
    trainable_params, total_params = count_params(model)
    print("[INFO] Trainable params: ", trainable_params, " total params: ", total_params)

    # Metrics
    average = 'micro'
    mdmc_avg = 'samplewise'
    metric_collection = MetricCollection([
        # Accuracy().to(device),
        F1(num_classes=config.num_classes, average=average, mdmc_average=mdmc_avg).to(device),
        # Precision(num_classes=config.num_classes, average=average, mdmc_average=mdmc_avg).to(device),
        # Recall(num_classes=config.num_classes, average=average, mdmc_average=mdmc_avg).to(device),
    ])

    loss_fn = None

    if 'focal_loss' == config.loss:
        loss_fn = FocalLoss(gamma=2.5, alpha=0.2, eps=1e-4)
    elif 'dice_focal' == config.loss:
        loss_fn = FocalDiceLoss()
    elif 'dice_cross' == config.loss:
        loss_fn = DiceBCELoss()
    elif 'dice' == config.loss:
        loss_fn = DiceLoss()
    else:
        exit("No loss function!")

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=config.weight_decay, 
        amsgrad=False 
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=1,
        T_mult=2,
        eta_min=5e-5, 
    )

    model.train()

    print("[INFO] Starting training!")
    best_f1 = -1

    if load_model:
        step = load_checkpoint(torch.load(model_path), model, optimizer)

    for epoch in range(config.num_epochs):
        
        running_loss = 0.0
        f1 = []
        last_idx = 0

        with tqdm(train_loader, unit='batch', position=0, leave=True) as tbatch:
            for idx, (imgs, msks) in enumerate(train_loader):
                imgs = imgs.to(device)
                msks = msks.to(device)

                preds = model(imgs)
                loss = loss_fn(preds, msks)

                step += 1

                optimizer.zero_grad()
                loss.backward(loss)
                optimizer.step()
                
                running_loss += loss.item()

                metrics = metric_collection(
                    torch.argmax(preds, dim=1), 
                    torch.argmax(msks, dim=1)
                )

                f1.append(metrics['F1'].item())
                last_idx = idx

                tbatch.set_description("Training")
                tbatch.set_postfix({
                    'Epoch batch': f"{epoch}-{idx+1}",
                    'Loss': running_loss/(idx+1),
                    # 'Accuracy': metrics['Accuracy'].item(),
                    'F1': np.mean(f1),
                    # 'Recall': metrics['Recall'].item(),
                    # 'Precision': metrics['Precision'].item(),
                })
                tbatch.update()
                sleep(0.01)

        actual_f1 = np.mean(f1)
        writer.add_scalar("Training Loss", running_loss/(last_idx+1), global_step=step)
        writer.add_scalar('F1_Score/train', actual_f1, epoch)
        save_imgs = False

        if (best_f1 < actual_f1):
            best_f1 = actual_f1
            best_epoch = epoch
            save_imgs = True
            if save_model:
                print(f"Saving best model in epoch {best_epoch} with loss {running_loss} and f1 {best_f1}")
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                save_checkpoint(checkpoint, filename=model_path)

        eval_step(model, test_loader, device, metric_collection, writer, epoch, f"{training_folder}/checkpoints", save_img=save_imgs)

    writer.close()

def get_training_augmentation(config):
    train_transform = [    
        album.RandomCrop(height=config.image_height, width=config.image_width, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
                album.Transpose(p=1),
                album.ShiftScaleRotate(p=1),
                # album.RandomSizedCrop(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation(config):   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [        
        album.CenterCrop (height=config.image_height, width=config.image_height, always_apply=True)        
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--retrain', metavar='retr', type=int,
                        help='Retrain architecture', default=0)

    parser.add_argument('--folder_name', metavar='folder', type=str,
                        help='Insert the folder for insights')

    parser.add_argument('--epochs', metavar='epochs', type=int,
                        help='epochs', default=0)
    args = vars(parser.parse_args())

    retrain = args['retrain']
    training_folder = 'trainings/' + args['folder_name']
    
    # getting training config
    config = get_config()#get_config()
    config.num_epochs = args['epochs'] if args['epochs'] else config.num_epochs

    print("[+] Epochs: ", config.num_epochs)

    # Defining dataset and data dirs

    # Getting info from csv
    class_dict = pd.read_csv(config.dataset_path + "/label_class_dict.csv")
    # Get class names
    class_names = class_dict['name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r','g','b']].values.tolist()

    print('All dataset classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)

    # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = ['background', 'building']

    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    DATA_DIR = config.dataset_path + 'tiff/'
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'train_labels')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'test_labels') 

    data_train_loader = MassachusettsBuildingsDataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(config),
        preprocessing=get_preprocessing(preprocessing_fn=None),
        class_rgb_values=select_class_rgb_values,
    )

    data_test_loader = MassachusettsBuildingsDataset(
        x_test_dir, y_test_dir, 
        augmentation=get_validation_augmentation(config), 
        preprocessing=get_preprocessing(preprocessing_fn=None),
        class_rgb_values=select_class_rgb_values,
    )

    num_workers = 2#os.cpu_count()
    train_loader = DataLoader(data_train_loader, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(data_test_loader, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    random_idx = random.randint(0, len(data_train_loader)-1)
    image, mask = data_train_loader[random_idx]

    print('image:',image.shape)
    print('mask:',mask.shape)
    print('reverse mask:', reverse_one_hot(mask).shape)
    # print('colour_code mask:', colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values).shape)
    print(data_train_loader.__len__())

    train(
        config=config, 
        load_model=retrain, 
        save_model=True, 
        training_folder=training_folder,
        train_loader=train_loader,
        test_loader=test_loader
    )

if __name__ == "__main__":
    main()
