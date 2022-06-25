"""Train AGXNet."""
import warnings
import argparse
import os
import random
import shutil
import time
import builtins
from pathlib import Path
import json

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

from tensorboard_logger import configure, log_value
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset import MIMICCXRDataset

from AGXNet import AGXNet
from pu_learning import top_bin_estimator_count
from utils import Bunch, keep_dicom_idx_in_batch


parser = argparse.ArgumentParser(description='AGXNet Training on MIMIC-CXR dataset.')

# Experiment
parser.add_argument('--exp-dir', metavar='DIR', default='/ocean/projects/asc170022p/yuke/PythonProject/AGXNet/exp/test_001', help='experiment path')

# Dataset
parser.add_argument('--img-chexpert-file', metavar='PATH', default='/ocean/projects/asc170022p/yuke/PythonProject/AGXNet/data/mimic-cxr-chexpert.csv',
                    help='master table including the image path and chexpert labels.')
parser.add_argument('--radgraph-sids-npy-file', metavar='PATH', default='/ocean/projects/asc170022p/yuke/PythonProject/AGXNet/preprocessing/landmark_observation_sids.npy',
                    help='radgraph study ids.')
parser.add_argument('--radgraph-adj-mtx-npy-file', metavar='PATH', default='/ocean/projects/asc170022p/yuke/PythonProject/AGXNet/preprocessing/landmark_observation_adj_mtx.npy',
                    help='radgraph adjacent matrix landmark - observation.')
parser.add_argument('--nvidia-bounding-box-file', metavar='PATH', default='/ocean/projects/asc170022p/yuke/PythonProject/AGXNet/data/mimic-cxr-annotation.csv',
                    help='bounding boxes annotated for pneumonia and pneumothorax.')
parser.add_argument('--chexpert-names', nargs='+', default=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'])
parser.add_argument('--full-anatomy-names', nargs='+', default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'])
parser.add_argument('--landmark-names-spec', nargs='+', default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
'cardiophrenic_sulcus', 'mediastinal', 'spine', 'rib', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
'interstitium', 'parenchymal', 'cavoatrial_junction', 'stomach', 'clavicle'])
parser.add_argument('--landmark-names-unspec', nargs='+', default=['cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'])
parser.add_argument('--full-obs', nargs='+', default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
'tail_abnorm_obs', 'excluded_obs'])
parser.add_argument('--norm-obs', nargs='+', default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free', 'expand', 'hyperinflate'])
parser.add_argument('--abnorm-obs', nargs='+', default=['effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
'device', 'engorgement', 'picc', 'clip', 'elevation', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration'])
parser.add_argument('--tail-abnorm-obs', nargs='+', default=['tail_abnorm_obs'])
parser.add_argument('--excluded-obs', nargs='+', default=['excluded_obs'])
parser.add_argument('--selected-obs', nargs='+', default=['pneumonia', 'pneumothorax'])

# PNU labels
parser.add_argument('--warm-up', default=2, type=int,
                    help='number of epochs warm up before PU learning.')

# Training
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                    help='PyTorch image models')
parser.add_argument('--pool1', metavar='ARCH', default='average',
                    help='type of pooling layer for net1. the options are: average, max, log-sum-exp')
parser.add_argument('--pool2', metavar='ARCH', default='average',
                    help='type of pooling layer for net2. the options are: average, max, log-sum-exp')
parser.add_argument('--gamma', default=5.0, type=float,
                    help='hyper-parameter for log-sum-exp pooling layer')
parser.add_argument('--nets-dep', metavar='ARCH', default='dep',
                    help='whether pass CAM1 to net2, dep or indep')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--training-schedule', default='end-to-end',
                    help='training schedule. the options are: '
                         '(1) end-to-end: concurrently train two networks end-to-end; '
                         '(2) interleave: train two networks independently')
parser.add_argument('--loss1', default='BCE_W',
                    help='anatomical landmark loss type.')
parser.add_argument('--loss2', default='CE',
                    help='observation loss type.')
parser.add_argument('--beta', default=0.1, type=float,
                    help='scaling weight of CAM1')
parser.add_argument('--dropout-method', default='random',
                    help='dropout method. 1, random, 2 proportional')
parser.add_argument('--dropout-rate', default=0.1, type=float,
                    help='randomly drop out x% of channels in the last conv. layer of net1')
parser.add_argument('--cam1-norm', default='norm',
                    help='cam1 normalization method. default: norm [0, 1]')
parser.add_argument('--cam2-norm', default='norm',
                    help='cam2 normalization method. default: norm [0, 1]')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--anneal-function', default='constant', type=str,
                    help='possible anneal functions: (1) logistic, (2) linear, (3) constant.')
parser.add_argument('--k',  default=0.5, type=float,
                    help='rate of annealing function.')
parser.add_argument('--x0', default=15, type=int,
                    help='offset of annealing function.')
parser.add_argument('--C', default=1.0, type=float,
                    metavar='W', help='weight of observation classification loss.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--ngpus-per-node', default=2, type=int,
                    help='Number of GPUs per node.')
# Image Augmentation
parser.add_argument('--resize', default=512, type=int,
                    help='input image resize')
parser.add_argument('--crop', default=448, type=int,
                    help='resize image crop')
parser.add_argument('--mini-data', default=None, type=int, help='small dataset for debugging')

# initialize global vars.
best_auprc = 0

def main():
    args = parser.parse_args()
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)

    args.N_landmarks_spec = len(args.landmark_names_spec)
    args.N_selected_obs = len(args.selected_obs)

    # save configurations to a dictionary
    with open(os.path.join(args.exp_dir, 'configs.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    f.close()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = args.ngpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # create tensorboard summary
    configure(args.exp_dir)

    global best_auprc
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> Creating AGXNet model '{}'".format(args.arch))
    model = AGXNet(args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define optimizer
    # custom learning rate, weigh decay at different layers/modules
    optimizer = torch.optim.SGD([{'params': list(model.net1.parameters()), 'lr': args.lr, 'weight_decay': args.weight_decay, 'momentum': args.momentum},
                                 {'params': list(model.fc1.parameters()), 'lr': args.lr, 'weight_decay': args.weight_decay , 'momentum': args.momentum},
                                 {'params': list(model.net2.parameters()), 'lr': args.lr, 'weight_decay': args.weight_decay, 'momentum': args.momentum},
                                 {'params': list(model.fc2.parameters()), 'lr': args.lr, 'weight_decay': args.weight_decay, 'momentum': args.momentum},
                                ])

    # optionally resume from a checkpoint
    if args.resume:
        ckpt_path = os.path.join(args.resume, 'model_epoch_2.pth.tar')
        if os.path.isfile(ckpt_path):
            f = open(os.path.join(args.resume, 'configs.json'))
            args = Bunch(json.load(f))
            args.distributed = False
            model = AGXNet(args)
            checkpoint = torch.load(ckpt_path)
            args.start_epoch = checkpoint['epoch']
            best_auprc = checkpoint['best_auprc']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.cuda()
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    arr_radgraph_sids = np.load(args.radgraph_sids_npy_file)  # N_sids
    arr_radgraph_adj = np.load(args.radgraph_adj_mtx_npy_file)  # N_sids * 51 * 75

    train_dataset = MIMICCXRDataset(args=args,
                                    radgraph_sids=arr_radgraph_sids,
                                    radgraph_adj_mtx=arr_radgraph_adj,
                                    mode='train',
                                    transform=transforms.Compose([
                                        transforms.Resize(args.resize),
                                        transforms.CenterCrop(args.resize),
                                        transforms.ToTensor(),
                                        normalize
                                    ])
                                    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_dataset = MIMICCXRDataset(args=args,
                              radgraph_sids = arr_radgraph_sids,
                              radgraph_adj_mtx = arr_radgraph_adj,
                              mode='valid',
                              transform=transforms.Compose([
                                  transforms.Resize(args.resize),
                                  transforms.CenterCrop(args.resize),
                                  transforms.ToTensor(),
                                  normalize
                                 ])
                             )

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # initialize alpha dicom id in U dataset
    u_alpha_obs_did = []
    obs_u_alpha = np.zeros(len(args.selected_obs))
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        train_did_arr, train_obs_pnu_arr, train_obs_prob_arr = train(train_loader, model, optimizer, epoch, args, u_alpha_obs_did, obs_u_alpha)

        # evaluate on validation set
        auroc1, auprc1, auroc2, auprc2, loss_val, u_alpha_obs_did, obs_u_alpha = validate(val_loader, model, args, epoch, train_did_arr, train_obs_pnu_arr, train_obs_prob_arr)

        # remember best acc@1 and save checkpoint
        is_best = auroc2 > best_auprc
        best_auprc = max(auprc2, best_auprc)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            # save the best model
            if epoch > args.warm_up:
                ckpt_name = 'model_epoch_' + str(epoch) + '_pnu.pth.tar'
            else:
                ckpt_name = 'model_epoch_' + str(epoch) + '.pth.tar'
            file_name = os.path.join(args.exp_dir, ckpt_name)
            save_checkpoint(args, epoch, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_auprc': best_auprc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, file_name)


def train(train_loader, model, optimizer, epoch, args, u_alpha_obs_did, obs_u_alpha):
    # update p label weights in dataloader
    if epoch > args.warm_up:
        train_loader.dataset.selected_obs_p_weights = train_loader.dataset.selected_obs_p_weights / (1-obs_u_alpha)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses1 = AverageMeter('Loss1', ':.4e')
    losses2 = AverageMeter('Loss2', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses1, losses2, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # set require gradients and optimizer
    if args.training_schedule == 'interleave':
        # only update network 1
        if epoch % 2 == 0:
            for q in model.net1.parameters():
                q.requires_grad = True
            for q in model.fc1.parameters():
                q.requires_grad = True
            for q in model.net2.parameters():
                q.requires_grad = False
            for q in model.fc2.parameters():
                q.requires_grad = False
            params = list(model.net1.parameters()) + list(model.fc1.parameters())
            optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        # only update network 2
        else:
            for q in model.net1.parameters():
                q.requires_grad = False
            for q in model.fc1.parameters():
                q.requires_grad = False
            for q in model.net2.parameters():
                q.requires_grad = True
            for q in model.fc2.parameters():
                q.requires_grad = True
            params = list(model.net2.parameters()) + list(model.fc2.parameters())
            optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)

    end = time.time()
    train_u_did_lst = []
    train_obs_pnu_lst = []
    train_obs_prob_lst = []
    for i, data in enumerate(train_loader):
        dicom_id, image, adj_mtx, _, _, landmark_spec_label, landmarks_spec_inverse_weight, landmark_spec_label_pnu, selected_obs_label, selected_obs_inverse_weight, selected_obs_label_pnu, _, _, _, _, _ = data
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)

        if torch.cuda.is_available():
            adj_mtx = adj_mtx.cuda(args.gpu, non_blocking=True)
            landmark_spec_label = landmark_spec_label.cuda(args.gpu, non_blocking=True)
            landmarks_spec_inverse_weight = landmarks_spec_inverse_weight.cuda(args.gpu, non_blocking=True)
            selected_obs_label = selected_obs_label.cuda(args.gpu, non_blocking=True)
            selected_obs_inverse_weight = selected_obs_inverse_weight.cuda(args.gpu, non_blocking=True)

        # compute output logit
        annealing = anneal_function(args, epoch)
        logit1, logit2, cam1_norm, cam1_agg_norm, cam2_norm = model(image, adj_mtx, annealing, mode='train')

        p2 = torch.sigmoid(logit2)
        train_u_did_lst.append(dicom_id)
        train_obs_pnu_lst.append(selected_obs_label_pnu.data.cpu())
        train_obs_prob_lst.append(p2.data.cpu())

        # weighted binary cross entropy loss, default reduction = 'mean' across samples & classes
        # top obs. alpha U samples
        if epoch > args.warm_up and len(u_alpha_obs_did) > 0:
            loss1 = compute_loss(args, logit1, landmark_spec_label, landmarks_spec_inverse_weight)
            loss2 = 0
            cnt_did = 0
            for io in range(len(args.selected_obs)):
                idx_did = keep_dicom_idx_in_batch(dicom_id, list(u_alpha_obs_did[io]))
                if len(idx_did) > 0:
                    cnt_did += len(idx_did)
                    z2 = logit2[:, io][idx_did]
                    y2 = selected_obs_label[:,io][idx_did]
                    w2 = selected_obs_inverse_weight[:, io][idx_did]
                    loss2 += F.binary_cross_entropy(torch.sigmoid(z2), y2, weight=w2, reduction='sum')
            loss2 = loss2 / (cnt_did + 1e-12)
        else:
            loss1 = compute_loss(args, logit1, landmark_spec_label, landmarks_spec_inverse_weight)
            loss2 = compute_loss(args, logit2, selected_obs_label, selected_obs_inverse_weight)

        loss = compute_total_loss(args, loss1, loss2, epoch)

        # update average meter
        losses1.update(loss1.item(), image.size(0))
        losses2.update(loss2.item(), image.size(0))
        losses.update(loss.item(), image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            step = i + len(train_loader) * epoch
            progress.display(i)
            log_value('train/epoch', epoch, step)
            log_value('train/loss1', progress.meters[2].avg, step)
            log_value('train/loss2', progress.meters[3].avg, step)
            log_value('train/loss', progress.meters[4].avg, step)

    train_did_arr = np.concatenate(train_u_did_lst, axis=0)
    train_obs_pnu_arr = np.concatenate(train_obs_pnu_lst, axis=0)
    train_obs_prob_arr = np.concatenate(train_obs_prob_lst, axis=0)
    return train_did_arr, train_obs_pnu_arr, train_obs_prob_arr


def validate(val_loader, model, args, epoch, train_did_arr, train_obs_pnu_arr, train_obs_prob_arr):
    batch_time = AverageMeter('Time', ':6.3f')
    losses1 = AverageMeter('Loss1', ':.4e')
    losses2 = AverageMeter('Loss2', ':.4e')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses1, losses2, losses],
        prefix='Validation: ')

    # switch to evaluate mode
    model.eval()

    loss_val = 0
    steps = 0
    # landmark
    target1_lst = []
    output1_lst = []
    land_pnu_lst = []

    # observation
    target2_lst = []
    output2_lst = []
    obs_pnu_lst = []

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            dicom_id, image, adj_mtx, _, _, landmark_spec_label, landmarks_spec_inverse_weight, landmark_spec_label_pnu, selected_obs_label, selected_obs_inverse_weight, selected_obs_label_pnu, _, _, _, _, _= data

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)

            if torch.cuda.is_available():
                adj_mtx = adj_mtx.cuda(args.gpu, non_blocking=True)
                landmark_spec_label = landmark_spec_label.cuda(args.gpu, non_blocking=True)
                landmarks_spec_inverse_weight = landmarks_spec_inverse_weight.cuda(args.gpu, non_blocking=True)
                selected_obs_label = selected_obs_label.cuda(args.gpu, non_blocking=True)
                selected_obs_inverse_weight = selected_obs_inverse_weight.cuda(args.gpu, non_blocking=True)

            # compute output logit
            annealing = anneal_function(args, epoch)
            logit1, _, cam1_norm, _, _ = model(image, adj_mtx, annealing, mode='valid')

            f2 = model.net2(image)[-1]
            f2_p = model.pool2(f2)
            logit2 = model.fc2(f2_p.squeeze())  # b * o

            # convert to probability
            p1 = torch.sigmoid(logit1)
            p2 = torch.sigmoid(logit2)

            # weighted binary cross entropy loss, default reduction = 'mean' across samples & classes
            loss1 = compute_loss(args, logit1, landmark_spec_label, landmarks_spec_inverse_weight)

            # weighted binary cross entropy loss, default reduction = 'mean' across samples & classes
            loss2 = compute_loss(args, logit2, selected_obs_label, selected_obs_inverse_weight)

            loss = compute_total_loss(args, loss1, loss2, epoch)

            # update average meter
            losses1.update(loss1.item(), image.size(0))
            losses2.update(loss2.item(), image.size(0))
            losses.update(loss.item(), image.size(0))

            # save output
            output1_lst.append(p1.data)
            target1_lst.append(landmark_spec_label.data)
            land_pnu_lst.append(landmark_spec_label_pnu.data)

            output2_lst.append(p2.data)
            target2_lst.append(selected_obs_label.data)
            obs_pnu_lst.append(selected_obs_label_pnu.data)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            loss_val += loss.item()
            steps += 1

            if i % args.print_freq == 0:
                progress.display(i)

    target1_tensor = torch.cat(target1_lst)
    output1_tensor = torch.cat(output1_lst)

    target2_tensor = torch.cat(target2_lst)
    output2_tensor = torch.cat(output2_lst)
    obs_pnu_tensor = torch.cat(obs_pnu_lst)

    # measure landmark labels AUC scores and record loss (PN only)
    AUROCs1, AUPRCs1 = compute_AUCs(target1_tensor, output1_tensor, args.N_landmarks_spec)
    AUROC1_avg = np.array(AUROCs1).mean()
    AUPRC1_avg = np.array(AUPRCs1).mean()
    print('The average anatomy AUROC and AUPRC is {AUROC_avg:.4f} and {AUPRC_avg:.4f}'.format(AUROC_avg=AUROC1_avg, AUPRC_avg=AUPRC1_avg))

    for i in range(len(args.landmark_names_spec)):
        print('The AUROC and AUPRC of {} is {:.4f} and {:.4f}'.format(args.landmark_names_spec[i], AUROCs1[i], AUPRCs1[i]))

    # measure observation labels AUC scores and record loss
    AUROCs2, AUPRCs2 = compute_AUCs_PN(target2_tensor, output2_tensor, obs_pnu_tensor, args.N_selected_obs)
    AUROC2_avg = np.array(AUROCs2).mean()
    AUPRC2_avg = np.array(AUPRCs2).mean()
    print('The average observation AUROC and AUPRC is {AUROC_avg:.4f} and {AUPRC_avg:.4f}'.format(AUROC_avg=AUROC2_avg, AUPRC_avg=AUPRC2_avg))

    for i in range(len(args.selected_obs)):
        print('The AUROC and AUPRC of {} is {:.4f} and {:.4f}'.format(args.selected_obs[i], AUROCs2[i], AUPRCs2[i]))

    log_value('valid/loss1', progress.meters[1].avg, epoch)
    log_value('valid/loss2', progress.meters[2].avg, epoch)
    log_value('valid/loss',  progress.meters[3].avg, epoch)
    log_value('valid/landmark_auroc', AUROC1_avg, epoch)
    log_value('valid/landmark_auprc', AUPRC1_avg, epoch)
    log_value('valid/observation_auroc', AUROC2_avg, epoch)
    log_value('valid/observation_auprc', AUPRC2_avg, epoch)

    # Start PU Learning
    # Best Bin Estimation for observation labels
    obs_prob_arr = output2_tensor.detach().cpu().numpy()
    obs_pnu_arr = obs_pnu_tensor.detach().cpu().numpy()
    # estimate mixture ration in the unlabeled data
    u_alpha_obs_did = []
    obs_u_alpha = np.ones(obs_prob_arr.shape[1])
    for io in range(obs_prob_arr.shape[1]):

        alpha_estimate = top_bin_estimator_count(obs_prob_arr[:, io], obs_pnu_arr[:, io])
        obs_u_alpha[io] = alpha_estimate
        print(args.selected_obs[io] + ' estimated alpha = ' + str(round(alpha_estimate, 2)))
        log_value('valid/' + args.selected_obs[io] +'_alpha', alpha_estimate, epoch)

        u_idx = train_obs_pnu_arr[:, io] == -1 # indices of U samples for ith obs.
        nu = u_idx.sum() # number of U samples for ith obs. in the train dataset
        train_u_obs_did = train_did_arr[u_idx] # their dicom id
        train_u_obs_prob = train_obs_prob_arr[:, io][u_idx] # predicted probs in the last iteration of training
        sorted_idx = np.argsort(train_u_obs_prob)[::-1] # sorted from high to low
        sorted_idx_alpha = sorted_idx[:int(nu * alpha_estimate)] # indices of top alpha U samples
        u_alpha_obs_did.append(train_u_obs_did[sorted_idx_alpha])

    return AUROC1_avg, AUPRC1_avg, AUROC2_avg, AUPRC2_avg, loss_val/steps, u_alpha_obs_did, obs_u_alpha


def save_checkpoint(args, epoch, state, is_best, filename):
    torch.save(state, os.path.join(args.exp_dir, filename))
    if is_best:
        if epoch > args.warm_up:
            ckpt_name = 'model_best_pnu.pth.tar'
        else:
            ckpt_name = 'model_best.pth.tar'
        shutil.copyfile(filename, os.path.join(args.exp_dir, ckpt_name))


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
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 6 epochs"""
    lr = args.lr * (0.33 ** (epoch // 12))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_AUCs(gt, pred, n):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    AUROCs = []
    AUPRCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
            AUPRCs.append(average_precision_score(gt_np[:, i], pred_np[:, i]))
        except:
            AUROCs.append(0.5)
            AUPRCs.append(0.5)
    return AUROCs, AUPRCs


def compute_AUCs_PN(gt, pred, pnu, n):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    AUROCs = []
    AUPRCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    pnu_np = pnu.cpu().numpy()
    for i in range(n):
        try:
            idx = pnu_np[:, i] != -1
            AUROCs.append(roc_auc_score(gt_np[idx, i], pred_np[idx, i]))
            AUPRCs.append(average_precision_score(gt_np[idx, i], pred_np[idx, i]))
        except:
            AUROCs.append(0.5)
            AUPRCs.append(0.5)
    return AUROCs, AUPRCs


def compute_loss(args, logit1, y1, w1):
    # logit b * N_landmarks
    if args.loss1 == 'BCE':
        loss1 = F.binary_cross_entropy(torch.sigmoid(logit1), y1, reduction='mean')
    elif args.loss1 == 'BCE_W':
        loss1 = F.binary_cross_entropy(torch.sigmoid(logit1), y1, weight=w1, reduction='mean')
    else:
        raise Exception('Invalid loss 1 type.')
    return loss1


def compute_total_loss(args, loss1, loss2, epoch):
    #anneal_weight = anneal_function(args, epoch)
    if args.training_schedule == 'end-to-end':
        loss = loss1 + loss2
    elif args.training_schedule == 'interleave':
        if epoch % 2 == 0:
            loss = loss1
        else:
            loss = loss2
    else:
        raise Exception('Invalid training schedule.')
    return loss

def anneal_function(args, epoch):
    """
    annealing function
    :param args: experiment configurations
    :param epoch: current epoch
    :return: annealing weight
    """
    k = args.k  # logistic function rate
    x0 = args.x0
    C = args.C # constant

    if args.anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (epoch - x0))))
    elif args.anneal_function == 'linear':
        return min(1, epoch / x0)
    elif args.anneal_function == 'constant':
        return C


if __name__ == '__main__':
    main()