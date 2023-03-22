"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
# import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import shutil
import glob
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

from model.model import AlexNet
from data_loader.data_loaders import CD_Dataset

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define model parameters
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 2  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0]  # GPUs to use
# modify this to point to your data directory
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in\\train'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '\\tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '\\models'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


if (not os.path.isdir(TRAIN_IMG_DIR + "\\dog")):
    os.makedirs(TRAIN_IMG_DIR + "\\cat")
    os.makedirs(TRAIN_IMG_DIR + "\\dog")
    
    for c in glob.glob(TRAIN_IMG_DIR + "\\cat*.jpg"):
        shutil.move(c,TRAIN_IMG_DIR+"\\cat")
    for d in glob.glob(TRAIN_IMG_DIR + "\\dog*.jpg"):
        shutil.move(c,TRAIN_IMG_DIR+"\\cat")

if __name__ == '__main__':
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))


    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')

    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)

    # train on multiple GPUs
    # alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    
    print(alexnet)
    print('AlexNet created')

    dataset = CD_Dataset(TRAIN_IMG_DIR,[transforms.CenterCrop(IMAGE_DIM),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    # dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
    #         # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    #         transforms.CenterCrop(IMAGE_DIM),
    #         # transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]))
    # # create dataset and data loader
    # dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose())
    print('Dataset created')
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')
    print(dataloader)


    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)

    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')
    print(f"Device >> {device}")

    # start training!!
    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)
            # sys.exit()

            # calculate the loss
            output = alexnet(imgs)
            loss = F.cross_entropy(output, classes)
        # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                        .format(epoch + 1, total_steps, loss.item(), accuracy.item()/BATCH_SIZE))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item()/BATCH_SIZE, total_steps)

            # print out gradient values and parameter average values
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                    parameter.grad.cpu().numpy(), total_steps)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                    parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)
            total_steps += 1
        lr_scheduler.step()

        # save checkpoints
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
        torch.save(state, checkpoint_path)
