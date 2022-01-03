import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
from utils import utils
from model.config import get_config
from model.network import CvTModified
from model.losses import *
from tqdm import tqdm
from time import sleep
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
import numpy as np

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

def train(load_model:bool, save_model:bool, training_folder:str):

    model_path = f"{training_folder}/model_trained_architecture.pt"

    # Creating folders for saving trainings
    utils.create_folder(f"{training_folder}/checkpoints")
    writer = SummaryWriter(training_folder)
    step = 0

    # getting training config
    config = get_config()

    model = CvTModified()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        Accuracy().to(device),
        F1(num_classes=config.num_classes, average=average, mdmc_average=mdmc_avg).to(device),
        Precision(num_classes=config.num_classes, average=average, mdmc_average=mdmc_avg).to(device),
        Recall(num_classes=config.num_classes, average=average, mdmc_average=mdmc_avg).to(device),
    ])

    loss = DiceBCELoss()
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
    if load_model:
        step = load_checkpoint(torch.load(model_path), model, optimizer)

    for epoch in range(config.num_epochs):
        
        running_loss = 0.0
        f1 = []

        with tqdm(train_loader, unit='batch', position=0, leave=True) as tbatch:
            for idx, (imgs, msks) in enumerate(train_loader):
                imgs = imgs.to(device)
                msks = msks.to(device)

                preds = model(imgs)
                loss = loss(preds, msks)

                writer.add_scalar("Training Loss", loss.item(), global_step=step)
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

                tbatch.set_description("Training")
                tbatch.set_postfix({
                    'Epoch batch': f"{epoch}-{idx+1}",
                    'Loss': running_loss/(idx+1),
                    'Accuracy': metrics['Accuracy'].item(),
                    'F1': metrics['F1'].item(),
                    'Recall': metrics['Recall'].item(),
                    'Precision': metrics['Precision'].item(),
                })
                tbatch.update()
                sleep(0.01)

        writer.add_scalar('F1_Score/train', np.mean(f1), epoch)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, filename=model_path)

        f1 = []
        accuracy = []
        precision = []
        recall = []

        model.eval()
        with torch.no_grad():
            with tqdm(test_dataloader, unit='batch', position=0, leave=True) as tbatch:
                for i, data in enumerate(tbatch):
                    image, mask = data['image'].to(device), data['mask'].to(device)
                    pred = model(image)

                    pred_argmax = torch.argmax(pred, dim=1)
                    mask_argmax = torch.argmax(mask, dim=1)
                    
                    metrics = metric_collection(
                        pred_argmax, 
                        mask_argmax
                    )
                    print("[INFO] Unique labels in this prediction: ", torch.unique(pred_argmax))

                    accuracy.append(metrics['Accuracy'].item())
                    f1.append(metrics['F1'].item())
                    recall.append(metrics['Recall'].item())
                    precision.append(metrics['Precision'].item())

                    tbatch.set_description("Training")
                    tbatch.set_postfix({
                        'Batch': f"{i+1}",
                        'Accuracy': np.mean(accuracy),
                        'F1': np.mean(f1),
                        'Recall': np.mean(recall),
                        'Precision': np.mean(precision),
                    })
                    tbatch.update()
                    sleep(0.01)

            writer.add_scalar('F1_Score/test', np.mean(f1), epoch)
        model.train()

    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--retrain', metavar='retr', type=int,
                        help='Retrain architecture', default=0)

    parser.add_argument('--folder_name', metavar='folder', type=str,
                        help='Insert the folder for insights')

    parser.add_argument('--lr_epoch_start', metavar='lr_decrease', type=int,
                        help='Start epoch lr decrease', default=10)
    args = vars(parser.parse_args())

    retrain = args['retrain']
    training_folder = 'trainings/' + args['folder_name']