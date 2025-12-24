import shutil
import warnings
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torch.utils.data as data
import os
import argparse
from sklearn.metrics import f1_score, confusion_matrix
from data_preprocessing.sam import SAM
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
import time
from torchsampler import ImbalancedDatasetSampler
from tqdm import tqdm  # [NEW] Import tqdm for progress bars

# --- Import from the new Mamba-based PosterV2 file ---
from models.PosterV2_7cls import *

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
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lr', '--learning-rate', default=3.5e-5, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=24, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=None, type=str, help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    # Create model (Uses the Mamba-based PosterV2)
    # Ensure ir50_path and facenet_path point to the correct files in 'models/pretrain/'
    model = pyramid_trans_expr2(img_size=224, num_classes=7)

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

    # [CHANGED] Cosine Scheduler is better for Mamba/Transformers than Exponential
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
    valdir = os.path.join(args.data, 'valid')

    # Ensure log directories exist
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

    test_dataset = datasets.ImageFolder(valdir,
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

    matrix = None

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        # [PRETTIER] Use a separator line
        print(f'\n{"=" * 40}')
        print(f'Epoch: {epoch + 1}/{args.epochs} | LR: {current_learning_rate:.2e}')
        print(f'{"=" * 40}')

        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(f'Epoch: {epoch + 1} | Learning rate: {current_learning_rate}\n')

        # Train
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)

        # Evaluate
        val_acc, val_los, output, target, D = validate(val_loader, model, criterion, args)

        # Step Scheduler
        scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        recorder.update(epoch, train_los, train_acc, val_los, val_acc, current_learning_rate, epoch_duration)
        recorder1.update(output, target)

        curve_name = time_str + 'cnn_dashboard.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        print(f'Epoch Time: {epoch_duration:.2f}s | Val Acc: {val_acc:.3f} | Best Acc: {best_acc:.3f}')
        with open(txt_name, 'a') as f:
            f.write(f'Epoch Time: {epoch_duration:.2f}s\n')

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if is_best:
            matrix = D
            print('>>> New Best Model Saved! <<<')

        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder1': recorder1,
                         'recorder': recorder}, is_best, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')

    # [PRETTIER] We define the ProgressMeter but only for logging to file, not printing
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()

    # [PRETTIER] TQDM Progress Bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch + 1}", unit="batch",
                leave=True)

    for i, (images, target) in pbar:
        images = images.cuda()
        target = target.cuda()

        # --- Step 1 ---
        output = model(images)
        loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()

        # [NEW] Clip Gradients for Stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.first_step(zero_grad=True)

        # --- Step 2 (SAM) ---
        output = model(images)
        loss = criterion(output, target)

        loss.backward()

        # [NEW] Clip Gradients again
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.second_step(zero_grad=True)

        # [PRETTIER] Update the progress bar with metrics
        pbar.set_postfix({'Loss': f'{losses.avg:.4f}', 'Acc': f'{top1.avg.item():.2f}%'})

        if i % args.print_freq == 0:
            # We only write to file to avoid messing up the bar
            progress.write_to_log(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')

    # [PRETTIER] Log file only
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    model.eval()

    # [FIXED] Initialize with numpy zeros
    D = np.zeros((7, 7))

    # [PRETTIER] TQDM for Validation
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", unit="batch", leave=True)

    with torch.no_grad():
        for i, (images, target) in pbar:
            images = images.cuda()
            target = target.cuda()
            output = model(images)
            loss = criterion(output, target)

            acc, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

            topk = (1,)
            with torch.no_grad():
                maxk = max(topk)
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()

            output = pred
            target = target.squeeze().cpu().numpy()
            output = output.squeeze().cpu().numpy()

            im_re_label = np.array(target)
            im_pre_label = np.array(output)
            y_ture = im_re_label.flatten()
            y_pred = im_pre_label.flatten()

            # [FIXED] Confusion Matrix Accumulation
            C = metrics.confusion_matrix(y_ture, y_pred, labels=[0, 1, 2, 3, 4, 5, 6])
            D += C

            # [PRETTIER] Update bar
            pbar.set_postfix({'Loss': f'{losses.avg:.4f}', 'Acc': f'{top1.avg.item():.2f}%'})

            if i % args.print_freq == 0:
                progress.write_to_log(i)

        # Print final results nicely
        print(f' * Final Val Accuracy: {top1.avg:.3f}%')
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')

    return top1.avg, losses.avg, output, target, D


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        best_state = state.pop('optimizer')
        torch.save(best_state, args.best_checkpoint_path)


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """Standard display that prints AND writes to file"""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def write_to_log(self, batch):
        """ [PRETTIER] ONLY writes to file, does not print to console"""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        # NO PRINT HERE
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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


labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']


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
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_lr = np.zeros((self.total_epoch, 1), dtype=np.float32)
        self.epoch_time = np.zeros((self.total_epoch, 1), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.epoch_lr[idx] = lr
        self.epoch_time[idx] = epoch_time
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        x_axis = np.arange(self.current_epoch)
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics (Epoch {self.current_epoch})', fontsize=16)

        axs[0, 0].plot(x_axis, self.epoch_accuracy[:self.current_epoch, 0], 'g-', label='Train')
        axs[0, 0].plot(x_axis, self.epoch_accuracy[:self.current_epoch, 1], 'y-', label='Valid')
        axs[0, 0].set_title('Accuracy')
        axs[0, 0].set_ylabel('%')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[0, 1].plot(x_axis, self.epoch_losses[:self.current_epoch, 0], 'g-', label='Train')
        axs[0, 1].plot(x_axis, self.epoch_losses[:self.current_epoch, 1], 'y-', label='Valid')
        axs[0, 1].set_title('Loss')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        axs[1, 0].plot(x_axis, self.epoch_lr[:self.current_epoch], 'b-', label='LR')
        axs[1, 0].set_title('Learning Rate Decay')
        axs[1, 0].set_ylabel('LR')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].grid(True)

        axs[1, 1].plot(x_axis, self.epoch_time[:self.current_epoch], 'r-', label='Time')
        axs[1, 1].set_title('Time per Epoch')
        axs[1, 1].set_ylabel('Seconds')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].grid(True)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
            print('Saved dashboard figure')
        plt.close(fig)


if __name__ == '__main__':
    main()