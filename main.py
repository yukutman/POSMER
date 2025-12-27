import argparse
import datetime
import os
import time
import warnings
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.metrics import f1_score
from torchsampler import ImbalancedDatasetSampler
from tqdm import tqdm

from data_preprocessing.sam import SAM

from models.Posmer_7cls import *

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=r'archive/DATASET')
parser.add_argument('--data_type', default='RAF-DB', choices=['RAF-DB', 'AffectNet-7', 'CAER-S'],
                    type=str, help='dataset option')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lr', '--learning-rate', default=3.5e-5, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=24, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=None, type=str, help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()


# Helper to plot and save Confusion Matrix without blocking
def save_confusion_matrix(cm, classes, epoch, output_dir='./log'):
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    filename = os.path.join(output_dir, f'{time_str}cm_epoch_{epoch}.png')
    plt.savefig(filename)
    plt.close()
    return filename


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    # Create model
    model = pyramid_mamba_expr2(img_size=224, num_classes=7)

    model = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    recorder = RecorderMeter(args.epochs)
    recorder1 = RecorderMeter1(args.epochs)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            recorder1 = checkpoint['recorder1']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')

    if not os.path.exists('./log'):
        os.makedirs('./log')
    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')

    if args.evaluate is None:
        if args.data_type == 'RAF-DB':
            train_dataset = datasets.ImageFolder(traindir,
                                                 transforms.Compose([transforms.Resize((224, 224)),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225]),
                                                                     transforms.RandomErasing(scale=(0.02, 0.1))]))
        else:
            train_dataset = datasets.ImageFolder(traindir,
                                                 transforms.Compose([transforms.Resize((224, 224)),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225]),
                                                                     transforms.RandomErasing(p=1,
                                                                                              scale=(0.05, 0.05))]))

        if args.data_type == 'AffectNet-7':
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       sampler=ImbalancedDatasetSampler(train_dataset),
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=args.workers,
                                                       pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=args.workers,
                                                       pin_memory=True)

    test_dataset = datasets.ImageFolder(testdir,
                                        transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225]),
                                                            ]))

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate is not None:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            best_acc = checkpoint['best_acc']
            best_acc = best_acc.to()
            print(f'best_acc:{best_acc}')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.evaluate))
        validate(val_loader, model, criterion, args)
        return

    # Labels for Confusion Matrix
    class_names = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', 'NE']

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        print(f'\n{"=" * 40}')
        print(f'Epoch: {epoch + 1}/{args.epochs} | LR: {current_learning_rate:.2e}')
        print(f'{"=" * 40}')

        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(f'Epoch: {epoch + 1} | Learning rate: {current_learning_rate}\n')

        # Train
        train_acc, train_los, train_f1 = train(train_loader, model, criterion, optimizer, epoch, args)

        # Evaluate
        val_acc, val_los, val_f1, output, target, D = validate(val_loader, model, criterion, args)

        # Step Scheduler
        scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        # Update Recorder
        recorder.update(epoch, train_los, train_acc, train_f1, val_los, val_acc, val_f1, current_learning_rate,
                        epoch_duration)
        recorder1.update(output, target)

        # Save individual curves
        recorder.plot_curves('./log/')

        # Check Best Acc
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print(
            f'Epoch Time: {epoch_duration:.2f}s | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f} | Best Acc: {best_acc:.3f}')

        # Log Confusion Matrix EVERY Epoch
        save_confusion_matrix(D.astype(int), class_names, epoch + 1)
        print(f'Saved Confusion Matrix for Epoch {epoch + 1}')

        with open(txt_name, 'a') as f:
            f.write(f'Epoch Time: {epoch_duration:.2f}s | Val F1: {val_f1:.3f}\n')
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        if is_best:
            print('>>> New Best Model Saved! <<<')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder1': recorder1,
                         'recorder': recorder}, is_best, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')

    all_preds = []
    all_targets = []

    model.train()
    loop = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{args.epochs}', leave=True)

    for i, (images, target) in enumerate(loop):
        images = images.cuda()
        target = target.cuda()

        # SAM Step 1
        output = model(images)
        loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # SAM Step 2
        output = model(images)
        loss = criterion(output, target)

        # Metrics update
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # Collect predictions for F1 Score
        _, pred = output.topk(1, 1, True, True)
        all_preds.append(pred.t().cpu().numpy().flatten())
        all_targets.append(target.cpu().numpy().flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.second_step(zero_grad=True)

        loop.set_postfix(loss=losses.avg, acc=top1.avg.item())

    # Calculate Train F1
    final_preds = np.concatenate(all_preds)
    final_targets = np.concatenate(all_targets)
    train_f1 = f1_score(final_targets, final_preds, average='macro')

    return top1.avg, losses.avg, train_f1


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')

    all_preds = []
    all_targets = []

    model.eval()

    # Initialize matrix D (7x7)
    D = np.zeros((7, 7))

    loop = tqdm(val_loader, desc='Validating', leave=True)

    with torch.no_grad():
        for images, target in loop:
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            loss = criterion(output, target)

            acc, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()

            all_preds.append(pred.cpu().numpy().flatten())
            all_targets.append(target.cpu().numpy().flatten())

            # Update manual matrix D
            y_true_batch = target.cpu().numpy().flatten()
            y_pred_batch = pred.cpu().numpy().flatten()
            C = metrics.confusion_matrix(y_true_batch, y_pred_batch, labels=[0, 1, 2, 3, 4, 5, 6])
            D += C

            loop.set_postfix(val_loss=losses.avg, val_acc=top1.avg.item())

    final_output = np.concatenate(all_preds)
    final_target = np.concatenate(all_targets)

    # Calculate Val F1
    val_f1 = f1_score(final_target, final_output, average='macro')

    print(f' * Final Validation Accuracy: {top1.avg:.3f} | F1 Macro: {val_f1:.3f}')

    with open('../log/' + time_str + 'log.txt', 'a') as f:
        f.write(' * Accuracy {top1.avg:.3f} | F1 {val_f1:.3f}\n'.format(top1=top1, val_f1=val_f1))

    return top1.avg, losses.avg, val_f1, final_output, final_target, D


def save_checkpoint(state, is_best, args):
    # 1. Save the FULL state (with recorders) for resuming training later
    # We keep everything here so you can resume training if the server crashes.
    torch.save(state, args.checkpoint_path)

    # 2. Save the BEST model (Cleaned for Inference)
    if is_best:
        # Create a shallow copy to modify without affecting the original 'state' dict
        best_state = state.copy()

        # REMOVE keys that cause loading errors or waste space
        keys_to_remove = ['optimizer', 'recorder', 'recorder1']
        for key in keys_to_remove:
            if key in best_state:
                del best_state[key]

        # Save the clean, portable checkpoint
        torch.save(best_state, args.best_checkpoint_path)
        print(f"Saved portable best checkpoint to: {args.best_checkpoint_path}")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter1(object):
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, output, target):
        self.y_pred = output
        self.y_true = target


class RecorderMeter(object):
    """Computes and stores metrics and plots them."""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        # [0] = train, [1] = val
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)
        # F1 Storage
        self.epoch_f1 = np.zeros((self.total_epoch, 2), dtype=np.float32)

        self.epoch_lr = np.zeros((self.total_epoch, 1), dtype=np.float32)
        self.epoch_time = np.zeros((self.total_epoch, 1), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, lr, epoch_time):
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc

        self.epoch_f1[idx, 0] = train_f1
        self.epoch_f1[idx, 1] = val_f1

        self.epoch_lr[idx] = lr
        self.epoch_time[idx] = epoch_time
        self.current_epoch = idx + 1

    def plot_curves(self, log_dir):
        x_axis = np.arange(self.current_epoch)

        # Accuracy Plot
        plt.figure()
        # Changed colors to match your image: Train=Blue, Val=Orange
        plt.plot(x_axis, self.epoch_accuracy[:self.current_epoch, 0], color='tab:blue', label='Train')
        plt.plot(x_axis, self.epoch_accuracy[:self.current_epoch, 1], color='tab:orange', label='Val')
        plt.title('Accuracy')
        plt.ylabel('%')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(log_dir, time_str + 'accuracy_curve.png'))
        plt.close()

        # Loss Plot
        plt.figure()
        plt.plot(x_axis, self.epoch_losses[:self.current_epoch, 0], color='tab:blue', label='Train')
        plt.plot(x_axis, self.epoch_losses[:self.current_epoch, 1], color='tab:orange', label='Val')
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(log_dir, time_str + 'loss_curve.png'))
        plt.close()

        # F1 Score Plot
        plt.figure()
        plt.plot(x_axis, self.epoch_f1[:self.current_epoch, 0], color='tab:blue', label='Train')
        plt.plot(x_axis, self.epoch_f1[:self.current_epoch, 1], color='tab:orange', label='Val')
        plt.title('F1 Macro Score')
        plt.ylabel('Score')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(log_dir, time_str + 'f1_curve.png'))
        plt.close()

        # LR Plot (Kept as Blue)
        plt.figure()
        plt.plot(x_axis, self.epoch_lr[:self.current_epoch], color='tab:blue', label='LR')
        plt.title('Learning Rate Decay')
        plt.ylabel('LR')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, time_str + 'lr_curve.png'))
        plt.close()

        # Time Plot (Kept as Red)
        plt.figure()
        plt.plot(x_axis, self.epoch_time[:self.current_epoch], 'r-', label='Time')
        plt.title('Time per Epoch')
        plt.ylabel('Seconds')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, time_str + 'time_curve.png'))
        plt.close()


if __name__ == '__main__':
    main()
