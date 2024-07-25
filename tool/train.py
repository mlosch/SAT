# System modules
import os
import sys
import time
import logging
import random
from collections import OrderedDict
 
# 
import argparse
import numpy as np

# PyTorch modules
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
### >>>
# Following workaround is necessary, if tensorflow is also installed
# according to: https://github.com/pytorch/pytorch/issues/30966#issuecomment-582747929
try:
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    from torch.utils.tensorboard import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter
### <<<

try:
    import apex.amp as amp
except:
    # print(' --- No mixed precision via apex ---')
    pass

# Package modules
from model import model_builder as builder
from model import BaseModel
from model.metrics import Histogram, ConfusionMatrix, Plottable
from model import AttackerModel
from lib import config
from lib import transforms
from lib import datasets
from lib.util import Meter, AverageMeter
from lib.scheduler import ScheduledModule
from lib.dataparallel_helpers import GraciousDataParallel


# torch.autograd.set_detect_anomaly(True)

class VoidSummaryWriter(object):
    def __init__(self):
        print('Initializing VoidSummaryWriter.')
        
    def add_histogram(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', required=True, type=str, help='config file')
    parser.add_argument('opts', help='see config/*.yaml for options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.model_name = os.path.splitext(os.path.basename(args.config))[0]

    adjust_save_paths(cfg, cfg.model_name)

    return cfg


def adjust_save_paths(config, model_name):
    for name in config.keys():
        value = config[name]

        if isinstance(value, dict):
            adjust_save_paths(value, model_name)
        elif type(value) is str and '$MODELNAME' in value:
            # update save_path
            config[name] = value.replace('$MODELNAME', model_name)


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main():
    args = get_parser()
    # check(args)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.training['train_gpu'])
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.training['train_gpu'])
    if len(args.training['train_gpu']) == 1:
        args.training['sync_bn'] = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        # port = find_free_port()
        # args.dist_url = "tcp://127.0.0.1:{}".format(port)
        print(args.dist_url)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.training['train_gpu'], args.ngpus_per_node, args)


def load_model_checkpoint(model, checkpoint, apply=True):
    if 'state_dict' in checkpoint:
        named_modules = dict(model.named_modules())
        for key in list(checkpoint['state_dict'].keys()):
            if 'auic' in key:
                del checkpoint['state_dict'][key]
            if not key.startswith('module.'):
                # if checkpoint was stored without DataParallel application
                key_ = 'module.'+key
                checkpoint['state_dict'][key_] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]
                key = key_
            if key.endswith('power_iterate') or key.endswith('input_shape') or key.endswith('lip_estimate'):
                model_dict = model.state_dict()
                if model_dict[key].shape != checkpoint['state_dict'][key].shape:
                    module = named_modules[key[:key.rfind('.')]]
                    param_name = key[key.rfind('.')+1:]
                    module.__getattr__(param_name).data = checkpoint['state_dict'][key]

        model_dict = model.state_dict()
        for key in list(checkpoint['state_dict'].keys()):
            if key not in model_dict:
                print('Layer {} not in checkpoint. Deleting key in checkpoint'.format(key))
            elif model_dict[key].shape != checkpoint['state_dict'][key].shape:
                print('Shape mismatch for layer {}. Deleting key in checkpoint'.format(key))
                del checkpoint['state_dict'][key]
        if apply:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        return checkpoint['state_dict']

    elif 'model' in checkpoint:
        for key in list(checkpoint['model'].keys()):
            if not 'module.model' in key:
                del checkpoint['model'][key]
            else:
                if 'conv' in key or 'pool' in key or 'fc' in key:
                    new_key = key[:key.rfind('.')] + '.parent' + key[key.rfind('.'):]
                    new_key = new_key.replace('.model', '')
                    # print(key, new_key)
                    checkpoint['model'][new_key] = checkpoint['model'][key]
                    del checkpoint['model'][key]

                    # shape_key = new_key[:new_key.rfind('.')] + '.input_shape'
                    # print(shape_key)
                    # checkpoint['model'][shape_key] = torch.Tensor()
        if apply:
            model.load_state_dict(checkpoint['model'], strict=False)
        return checkpoint['model']
    else:
        raise RuntimeError


def setup_optimizer(model, args):
    include_layers = args['training']['optimizer'].pop('layers', None)
    # exclude_layers = args['training']['optimizer'].pop('exclude_layers', None)

    # if 'l1_decay' in args['training']['optimizer']:
    #   l1_decay = args['training']['optimizer'].pop('l1_decay')


    if include_layers is None:
        optimizer = optim.__dict__[args['training']['optimizer'].pop('type')](model.parameters(), **args['training']['optimizer'])
    else:
        params = []
        for layer in include_layers:
            params += model._modules[layer].parameters()
        optimizer = optim.__dict__[args['training']['optimizer'].pop('type')](params, **args['training']['optimizer'])

    return optimizer


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if type(gpu) is list:
        args.gpu = None
    else:
        args.gpu = gpu

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.training['train_gpu'])
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    if main_process():
        global logger, writer
        logger = get_logger()
        if args.training.get('evaluate_only', False):
            writer = VoidSummaryWriter()
        else:
            writer = SummaryWriter(args.training['save_path'])
        logger.info(args)

    model = builder.build(args)
    if main_process():
    # model.eval()
        logger.info(model)

        if args.training['sync_bn']:
            # if args.distributed:
                # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # else:
            logger.warning('sync_bn flag not implemented. Use BatchNorm with caution')
        # else:
        #     raise NotImplementedError('sync bn without distribution not implemented at this point')

    # set up optimizer
    optimizer = setup_optimizer(model, args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        model = model.cuda()

    if 'mixed_precision' in args['training']:
        opt_level = args['training']['mixed_precision'].get('opt_level', 'O2')
        loss_scale = args['training']['mixed_precision'].get('loss_scale', 1.0) #'dynamic')
        keep_fp32_weights = args['training']['mixed_precision'].get('keep_fp32_weights', False)
        amp_args = dict(opt_level=opt_level, loss_scale=loss_scale, verbosity=False)
        if opt_level == 'O2':
            amp_args['master_weights'] = keep_fp32_weights
        model, optimizer = amp.initialize(model, optimizer, **amp_args)
        # model._forward_pass, optimizer = amp.initialize(model._forward_pass, optimizer, **amp_args)
        # def amp_forward_pass(old_fwd):
        #     def new_fwd(*args, **kwargs):
        #         output = old_fwd(*amp.applier(args, input_caster),
        #                          **amp.applier(kwargs, input_caster))
        #         return applier(output, output_caster)
        #     return new_fwd
        # model._forward_pass = amp_forward_pass(model._forward_pass)
        # amp.register_half_function(model, '_forward_pass')
        model.optimizer = optimizer
        if main_process():
            logger.info('Using mixed precision with opt_level {}'.format(opt_level))

    if args.training.get('freeze_bn', False):
        if main_process():
            logger.info('Freezing BatchNorm layers during training')
        model.freeze_bn = True

    if main_process():
        nb_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
        logger.info(f' ## Number of parameters to train: {nb_parameters} ##')

    if args.distributed:
        args.batch_size = int(args.training['batch_size'] / ngpus_per_node)
        args.batch_size_val = int(args.validate['batch_size'] / ngpus_per_node)
        args.workers = int((args.training['workers'] + ngpus_per_node - 1) / ngpus_per_node)
        # if 'mixed_precision' in args['training']:
        #     from apex.parallel import DistributedDataParallel
        #     model = DistributedDataParallel(model)
        # else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu]) #, find_unused_parameters=True)
    else:
        args.batch_size = args.training['batch_size']
        model = GraciousDataParallel(model)
        # model = torch.nn.DataParallel(model)

    # set up lr scheduler (if applicable)
    scheduler = None
    if 'lrscheduler' in args['training']:
        last_iter = args['training']['lrscheduler'].pop('last_iter', 0)
        scheduler = lr_scheduler.__dict__[args['training']['lrscheduler'].pop('type')](optimizer, **args['training']['lrscheduler'])
        for it in range(last_iter):
            scheduler.step()

    args.start_epoch = 0
    if args.training.get('resume', None) is not None:
        if os.path.isfile(args.training['resume']):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.training['resume']))
            # checkpoint = torch.load(args.resume)
            if args.gpu:
                loc = 'cuda:{}'.format(gpu)
            else:
                loc = 'cuda'
            checkpoint = torch.load(args.training['resume'], map_location=loc) #lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']

            load_model_checkpoint(model, checkpoint)

            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None and 'lrscheduler' in checkpoint and checkpoint['lrscheduler'] is not None:
                scheduler.load_state_dict(checkpoint['lrscheduler'])

            if 'mixed_precision' in args['training']:
                amp.load_state_dict(checkpoint['amp'])

            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.training['resume'], checkpoint['epoch']))

            if 'Lipschitz' in args.architecture['type']:
                if main_process():
                    if 'model' in checkpoint:
                        logger.info("=> running lipschitz estimation once to update internal states")
                        with torch.no_grad():
                            K = model.module.lipschitz_estimate(num_iter=-1)
                        logger.info('Global Lipschitz estimate: {:.2f}'.format(K.item()))
                        logger.info("=> Done")
                    # else:
                    #     with torch.no_grad():
                    #         model.module.lipschitz_estimate(num_iter=-1)
                        
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.training['resume']))
                raise RuntimeError
    elif args.training.get('pretrained_weights', None) is not None:
        if os.path.isfile(args.training['pretrained_weights']):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.training['pretrained_weights']))
            checkpoint = torch.load(args.training['pretrained_weights'], map_location=lambda storage, loc: storage.cuda())
            load_model_checkpoint(model, checkpoint)
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.training['pretrained_weights']))
                raise RuntimeError
    elif args.training.get('average_pretrained_weights', None) is not None:
        assert isinstance(args.training['average_pretrained_weights'], list)
        if main_process():
            logger.info('Loading and averaging {} checkpoints'.format(len(args.training['average_pretrained_weights'])))
            # logger.info("=> loading checkpoint '{}'".format(args.training['average_pretrained_weights']))
        state_avg = None
        for checkpoint_fp in args.training['average_pretrained_weights']:
            assert os.path.isfile(checkpoint_fp)

            checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage.cuda())
            state_dict = load_model_checkpoint(model, checkpoint, apply=False)
            if state_avg is None:
                state_avg = state_dict
            else:
                with torch.no_grad():
                    for key, v in state_dict.items():
                        state_avg[key] += v

        with torch.no_grad():
            if main_process():
                logger.info('Averaging ...')
            Navgs = float(len(args.training['average_pretrained_weights']))
            for key, v in state_dict.items():
                if main_process():
                    logger.info('{} of type {}'.format(key, type(v)))
                state_avg[key] = state_avg[key] / Navgs


    train_data = compose_data(args.training_data)
    val_data = compose_data(args.validation_data)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.training['workers'], pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.training['workers'], pin_memory=True, sampler=val_sampler)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.training['workers'], pin_memory=True, sampler=val_sampler)

    best_rob_acc = 0
    if args.training.get('resume', None) is not None:
        filename = args.training['save_path'] + '/best_rob_acc.pth'
        if os.path.isfile(filename) and main_process():
            best_ckpt = torch.load(filename, map_location='cpu')
            best_rob_acc = best_ckpt.get('robust_acc', 0)
            if best_rob_acc != 0:
                logger.info('=> found previous best value for robust accuracy: {:.2f}'.format(best_rob_acc))

    for epoch in range(args.start_epoch, args.training['epochs']):
        epoch_log = epoch + 1

        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.training.get('evaluate_only', False) is False:
            if main_process():
                logger.info('Starting training ...')
            meters = train(epoch, args, model, train_loader, optimizer, scheduler)

            if main_process():
                for category, meter in meters.items():
                    for name, mtr in meter.items():
                        if isinstance(mtr, dict):
                            for subname, submtr in mtr.items():
                                if isinstance(submtr, Histogram):
                                    if submtr.already_binned:
                                        writer.add_histogram('train-'+category+'_'+name+'/'+subname, submtr.values, epoch_log, bins=len(submtr.values))
                                    else:
                                        writer.add_histogram('train-'+category+'_'+name+'/'+subname, submtr.values, epoch_log)
                                else:
                                    writer.add_scalar('train-'+category+'_'+name+'/'+subname, submtr.avg, epoch_log)
                        else:
                            if isinstance(mtr, Plottable):
                                writer.add_figure('train-'+category+'_'+name, mtr.plot(epoch_log), epoch_log)
                            else:
                                writer.add_scalar('train-'+category+'/'+name, mtr.avg, epoch_log)

            if (epoch_log % args.training['save_freq'] == 0) and main_process():
                filename = args.training['save_path'] + '/train_epoch_' + str(epoch_log) + '.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({
                    'epoch': epoch_log, 
                    'state_dict': model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'lrscheduler': scheduler.state_dict() if scheduler is not None else None,
                    'amp': amp.state_dict() if 'mixed_precision' in args['training'] else None,
                    }, filename)
                # if epoch_log / args.training['save_freq'] > 2:
                #     deletename = args.training['save_path'] + '/train_epoch_' + str(epoch_log - args.training['save_freq'] * 2) + '.pth'
                #     if os.path.exists(deletename):
                #         os.remove(deletename)
                if epoch_log / args.training['save_freq'] > 1:
                    deletename = args.training['save_path'] + '/train_epoch_' + str(epoch_log - args.training['save_freq']) + '.pth'
                    if os.path.exists(deletename):
                        os.remove(deletename)

            val_freq = args.training.get('val_freq', 1)
            if (type(val_freq) is list and epoch_log in val_freq) or \
                (type(val_freq) is not list and (epoch_log % val_freq) == 0):
                with torch.no_grad():
                    meters = validate(epoch, args, model, val_loader)
                if main_process():
                    for category, meter in meters.items():
                        for name, mtr in meter.items():
                            if isinstance(mtr, dict):
                                for subname, submtr in mtr.items():
                                    writer.add_scalar('val-'+category+'_'+name+'/'+subname, submtr.avg, epoch_log)
                            else:
                                if isinstance(mtr, Plottable):
                                    writer.add_figure('val-'+category+'_'+name, mtr.plot(epoch_log), epoch_log)
                                else:
                                    writer.add_scalar('val-'+category+'/'+name, mtr.avg, epoch_log)


        if args.training.get('evaluate_only', False):
            epoch_log -= 1
            epoch -= 1
            with torch.no_grad():
                meters = validate(epoch, args, model, val_loader)
            model_ = model.module if isinstance(model, torch.nn.DataParallel) else model
            if hasattr(model_, 'lipschitz_estimate'):
                with torch.no_grad():
                    lc = model_.lipschitz_estimate
                    # update lipschitz estimates
                    # run until convergence
                    Kt = lc(num_iter=1)
                    Kt1 = lc(num_iter=1)
                    eps = 1.e-6
                    num_iters = 2
                    while (Kt1-Kt).abs() > eps:
                        Kt = Kt1
                        Kt1 = lc(num_iter=1)
                        num_iters += 1

                    if main_process():
                        logger.info('Number of lipschitz estimate iterations: {}'.format(num_iters))
                        logger.info('\tepsilon: {}'.format((Kt1-Kt).abs().item()))
                        logger.info('\tLipschitz estimate: {}'.format(Kt1.item()))
            else:
                if main_process():
                    logger.info('No Lipschitz estimate. Model is not of instance LipschitzModel')
        
        if main_process():
            eval_kwargs = dict(
                train_loader=train_loader,
                val_loader=val_loader,
                train_dataset=train_data,
                val_dataset=val_data,
                epoch=epoch_log)
            eval_kwargs['logger'] = logger
            logger.info('Running evaluation modules ...')

            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                base = model.module
            else:
                base = model

            if isinstance(base, BaseModel):
                evaluation_results = base.evaluate(**eval_kwargs)

                for category, meter in evaluation_results.items():
                    for name, mtr in meter.items():
                        if isinstance(mtr, dict):
                            for subname, submtr in mtr.items():
                                if isinstance(submtr, Histogram):
                                    writer.add_histogram('eval-'+category+'_'+name+'/'+subname, submtr.values, epoch_log)
                                else:
                                    writer.add_scalar('eval-'+category+'_'+name+'/'+subname, submtr.item(), epoch_log)
                        else:
                            if isinstance(mtr, Histogram):
                                writer.add_histogram('eval-'+category+'/'+name, mtr.values, epoch_log)
                            else:
                                writer.add_scalar('eval-'+category+'/'+name, mtr.item(), epoch_log)

                if isinstance(model.module, AttackerModel):
                    # if 'save_best_eval_key'
                    # save if robust accuracy has improved
                    robust_norm_key = 'auto_attack_linf'
                    if model.module.norm_p == 1:
                        robust_norm_key = 'auto_attack_l1'
                    elif model.module.norm_p == 2:
                        robust_norm_key = 'auto_attack_l2'

                    if (args.training.get('evaluate_only', False) is False) and robust_norm_key in evaluation_results:
                        eps_target = model.module.eps
                        robust_acc_key = 'robust_acc_{:.3f}'.format(eps_target)
                        if not robust_acc_key in evaluation_results[robust_norm_key]:
                            robust_acc_key = 'robust_acc_all_{:.3f}'.format(eps_target)
                        cur_rob_acc = evaluation_results[robust_norm_key][robust_acc_key]

                        # if robust_norm_key.endswith('linf'):
                        #     if 'robust_acc_all_0.031' in evaluation_results[robust_norm_key]:
                        #         cur_rob_acc = evaluation_results[robust_norm_key]['robust_acc_all_0.031']
                        #     else:
                        #         cur_rob_acc = evaluation_results[robust_norm_key]['robust_acc_0.031']
                        # else:
                        #     if 'robust_acc_all_0.500' in evaluation_results[robust_norm_key]:
                        #         cur_rob_acc = evaluation_results[robust_norm_key]['robust_acc_all_0.500']
                        #     else:
                        #         cur_rob_acc = evaluation_results[robust_norm_key]['robust_acc_0.500']

                        if cur_rob_acc > best_rob_acc:
                            filename = args.training['save_path'] + '/best_rob_acc.pth'

                            logger.info('Improved robust accuracy (p={}) {} -> {}.'.format(model.module.norm_p, best_rob_acc, cur_rob_acc))
                            logger.info('Saving checkpoint to: ' + filename)
                            best_rob_acc = cur_rob_acc
                            torch.save({
                                'epoch': epoch_log, 
                                'robust_acc': cur_rob_acc,
                                'state_dict': model.state_dict(), 
                                'optimizer': optimizer.state_dict(), 
                                'lrscheduler': scheduler.state_dict() if scheduler is not None else None,
                                'amp': amp.state_dict() if 'mixed_precision' in args['training'] else None,
                                }, filename)
                            # torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'lrscheduler': scheduler.state_dict() if scheduler is not None else None}, filename)

        if args.distributed:
            # when evaluating, make sure we're synchronized
            dist.barrier()

        if args.training.get('evaluate_only', False) is True:
            break


def compose_data(args):
    dataset_type = args['type']
    transform_list = []
    
    for tr_name, tr_config in args['transformation'].items():
        tr_type = tr_config.pop('type')
        if main_process():
            logger.info('Composing Dataset :: Initializing {} with arguments: {} {}'.format(
                tr_type, 
                str(tr_config.get('args', [])), 
                str(tr_config.get('kwargs', ''))
                )
            )
        if tr_type.startswith('torchvision.transforms.'):
            transformation_class = torchvision.transforms.__dict__[tr_type.replace('torchvision.transforms.','')]
        else:
            transformation_class = transforms.__dict__[tr_type]

        transform_list.append(transformation_class(*tr_config.get('args', []), **tr_config.get('kwargs', {})))

    if dataset_type.startswith('torchvision.datasets.'):
        dataset_class = torchvision.datasets.__dict__[dataset_type.replace('torchvision.datasets.', '')]
    else:
        dataset_class = datasets.__dict__[dataset_type]

    if args.get('two_crop_transform', False):
        data = dataset_class(
            *args.get('args', []),
            transform=transforms.TwoCropTransform(torchvision.transforms.Compose(transform_list)), 
            **args.get('kwargs', {}))
    else:
        data = dataset_class(
            *args.get('args', []),
            transform=torchvision.transforms.Compose(transform_list), 
            **args.get('kwargs', {}))

    return data


def train(epoch, args, model, train_loader, optimizer, lr_scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    meters = OrderedDict()
    meters['loss'] = OrderedDict()
    meters['metric'] = OrderedDict()
    meters['scheduler'] = OrderedDict()

    schedulers = []
    # Check modules for schedulers
    for mod in model.modules():
        if isinstance(mod, ScheduledModule):
            schedulers.append(mod.schedulers())

    model.train()

    base = model
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        base = model.module
    if isinstance(base, BaseModel):
        base.pre_training_hook()

    if args.gpu:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cuda')

    end = time.time()
    max_iter = args.training['epochs'] * len(train_loader)
    for i, train_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        current_iter = epoch * len(train_loader) + i + 1

        data_kwargs = {}
        if len(train_data) == 2:
            input, target = train_data
        elif len(train_data) == 3:
            input, target, data_kwargs = train_data
        data_kwargs['epoch'] = epoch+1

        if type(input) is list:
            # merge inputs for SupConLoss
            n = input[0].size(0)
            input = torch.cat(input, dim=0)
        else:
            n = input.size(0)

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        

        if len(input.shape) == 5:
            target = target.unsqueeze(1).repeat(1,input.shape[1])
            target = target.view(-1)
        
        outputs = model(input, target, **data_kwargs)

        losses = outputs['loss']
        optimizer.zero_grad()
        for _, loss in outputs['loss'].items():
            if 'mixed_precision' in args['training']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        # loss = 0
        # for _, l in outputs['loss'].items():
        #     loss = l + loss

        # optimizer.zero_grad()
        # if 'mixed_precision' in args['training']:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()
        optimizer.step()

        metrics = outputs.get('metric', {})
        # print(metrics)

        total_loss = 0

        
        if args.multiprocessing_distributed:
            # count = target.new_tensor([n], dtype=torch.long)
            # for name in losses.keys():
            #     losses[name] = losses[name] * n
            #     dist.all_reduce(losses[name])
            # for name in metrics.keys():
            #     metrics[name] = metrics[name] * n
            #     dist.all_reduce(metrics[name])
            # dist.all_reduce(count)
            # n = count.item()
            # for name in losses.keys():
            #     losses[name] = losses[name] / n
            # for name in metrics.keys():
            #     metrics[name] = metrics[name] / n

            count = torch.tensor([n], dtype=torch.long, device=device)
            dist.all_reduce(count, dist.ReduceOp.SUM, async_op=False)
            n = count.item()

            for name, loss_ in losses.items():
                loss_ /= args.world_size
                dist.all_reduce(loss_, dist.ReduceOp.SUM, async_op=False)
                losses[name] = loss_
                total_loss = total_loss + loss_
            for name, measure in metrics.items():
                if isinstance(measure, dict):
                    for subname, submeasure in measure.items():
                        submeasure /= args.world_size
                        dist.all_reduce(submeasure, dist.ReduceOp.SUM, async_op=False)
                        metrics[name][subname] = submeasure / args.world_size
                elif isinstance(measure, ConfusionMatrix) or isinstance(measure, Histogram):
                    # TODO:
                    #  If parallel, make use of this code
                    # metrics[name] = measure[0]
                    # for measure_i in range(1,len(measure)):
                    #     metrics[name].update(measure[measure_i])
                    metrics[name] = measure
                else:
                    if measure is None:
                        metrics[name] = None
                    else:
                        measure /= args.world_size
                        dist.all_reduce(measure, dist.ReduceOp.SUM, async_op=False)
                        metrics[name] = measure

        else:
            for name, loss_ in losses.items():
                losses[name] = torch.sum(loss_) / args.ngpus_per_node
                total_loss = losses[name] + total_loss
            for name, measure in metrics.items():
                if isinstance(measure, dict):
                    for subname, submeasure in measure.items():
                        metrics[name][subname] = torch.sum(submeasure) / args.ngpus_per_node
                elif isinstance(measure, ConfusionMatrix) or isinstance(measure, Histogram):
                    # TODO:
                    #  If parallel, make use of this code
                    # metrics[name] = measure[0]
                    # for measure_i in range(1,len(measure)):
                    #     metrics[name].update(measure[measure_i])
                    metrics[name] = measure
                else:
                    if measure is None:
                        metrics[name] = None
                    else:
                        metrics[name] = torch.sum(measure) / args.ngpus_per_node

        for name, loss_ in losses.items():
            if name not in meters['loss']:
                meters['loss'][name] = AverageMeter()
            meters['loss'][name].update(loss_.item(), n)
        for name, measure in metrics.items():
            if isinstance(measure, dict):
                if name not in meters['metric']:
                    meters['metric'][name] = OrderedDict()
                for subname, submeasure in measure.items():
                    if isinstance(submeasure, Histogram):
                        meters['metric'][name][subname] = submeasure
                    else:
                        if subname not in meters['metric'][name]:
                            meters['metric'][name][subname] = AverageMeter()
                        meters['metric'][name][subname].update(submeasure.item(), n)
            else:
                if isinstance(measure, Histogram):
                    if name not in meters['metric'] or not hasattr(measure, 'update'):
                        meters['metric'][name] = measure
                    else:
                        meters['metric'][name].update(measure)
                elif isinstance(measure, ConfusionMatrix):
                    if name not in meters['metric']:
                        meters['metric'][name] = measure
                    else:
                        meters['metric'][name].update(measure)
                else:
                    if name not in meters['metric']:
                        meters['metric'][name] = AverageMeter()
                    if measure is not None:
                        meters['metric'][name].update(measure.item(), n)

        post_kwargs = dict(
            save_path=args.training['save_path'],
            epoch=epoch,
            iteration=current_iter)
        if isinstance(model, BaseModel) or \
        (isinstance(model, torch.nn.DataParallel) and isinstance(model.module, BaseModel)):
            if isinstance(model, torch.nn.DataParallel):
                post_metrics = model.module.post_training_iteration_hook(**post_kwargs)
            else:
                post_metrics = model.post_training_iteration_hook(**post_kwargs)

        batch_time.update(time.time() - end)
        end = time.time()

        if lr_scheduler is not None:
            lr_scheduler.step()

        for schedulerlist in schedulers:
            sched_kwargs = dict(curr_iter=current_iter, max_iter=max_iter)
            for scheduler in schedulerlist:
                var_val = scheduler.update(**sched_kwargs)
                name = scheduler.name
                if name not in meters['scheduler']:
                    meters['scheduler'][name] = Meter()
                meters['scheduler'][name].update(var_val)

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.training['print_freq'] == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'ETA {remain_time} '
                        '{losses} '
                        '[{metrics}]'.format(epoch+1, args.training['epochs'], i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          losses=' '.join(['{} {:.4f}'.format(name.title(), mtr.avg) for name, mtr in meters['loss'].items()]),
                                                          metrics=' '.join(['{} {:.1f}'.format(name.title(), mtr.avg) for name, mtr in meters['metric'].items() if not isinstance(mtr, dict) and not isinstance(mtr, Plottable)])))


    post_kwargs = dict(
        save_path=args.training['save_path'],
        epoch=epoch)
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        base = model.module
    if isinstance(base, BaseModel):
        post_metrics = base.post_training_hook(**post_kwargs)

        if main_process():
            for name, measure in post_metrics.items():
                if isinstance(measure, dict):
                    if name not in meters['metric']:
                        meters['metric'][name] = OrderedDict()
                    for subname, submeasure in measure.items():
                        if isinstance(submeasure, Histogram):
                            if subname not in meters['metric'][name] or not hasattr(submeasure, 'update'):
                                meters['metric'][name][subname] = submeasure
                            else:
                                meters['metric'][name][subname].update(submeasure)
                        else:
                            if subname not in meters['metric'][name]:
                                meters['metric'][name][subname] = AverageMeter()
                            meters['metric'][name][subname].update(submeasure.item(), n)
                else:
                    if name not in meters['metric']:
                        meters['metric'][name] = AverageMeter()
                    meters['metric'][name].update(measure.item(), n)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: {losses} [{metrics}].'.format(epoch+1, args.training['epochs'], 
            losses=' '.join(['{} {:.4f}'.format(name.title(), mtr.avg) for name, mtr in meters['loss'].items()]),
            metrics=' '.join(['{} {:.1f}'.format(name.title(), mtr.avg) for name, mtr in meters['metric'].items() if not isinstance(mtr, dict) and not isinstance(mtr, Plottable)])))
        for name, measure in meters['metric'].items():
            if isinstance(measure, dict):
                logger.info('------------------------------------------------')
                logger.info('{}:'.format(name.title()))
                for subname, submeasure in measure.items():
                    if isinstance(submeasure, Histogram):
                        hist = submeasure.values
                        logger.info('\t{} = min: {:.3f}, mean: {:.3f}, max:{:.3f}'.format(subname, hist.min().item(), hist.mean().item(), hist.max().item()))
                    else:
                        logger.info('\t{} = {}'.format(subname, submeasure.avg))
        logger.info('\n')
    return meters


def validate(epoch, args, model, val_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    meters = OrderedDict()
    meters['loss'] = OrderedDict()
    meters['metric'] = OrderedDict()

    # schedulers = []
    # # Check modules for schedulers
    # for mod in model.modules():
    #     if isinstance(mod, ScheduledModule):
    #         schedulers.append(mod.schedulers())

    model.eval()

    # if isinstance(model, BaseModel):
    #     model.pre_training_hook()

    if args.gpu:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cuda')

    end = time.time()
    for i, val_data in enumerate(val_loader):
        data_time.update(time.time() - end)

        data_kwargs = {}
        if len(val_data) == 2:
            input, target = val_data
        elif len(val_data) == 3:
            input, target, data_kwargs = val_data
        data_kwargs['epoch'] = epoch+1

        current_iter = epoch * len(val_loader) + i + 1

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        if len(input.shape) == 5:
            target = target.unsqueeze(1).repeat(1,input.shape[1])
            target = target.view(-1)

        if len(val_data) == 3:
            outputs = model(input, target.squeeze(), **data_kwargs)
        else:
            outputs = model(input, target.squeeze())

        losses = outputs['loss']
        metrics = outputs.get('metric', {})

        total_loss = 0

        n = input.size(0)
        if args.multiprocessing_distributed:
            # count = target.new_tensor([n], dtype=torch.long)
            # for name in losses.keys():
            #     losses[name] = losses[name] * n
            #     dist.all_reduce(losses[name])
            # for name in metrics.keys():
            #     metrics[name] = metrics[name] * n
            #     dist.all_reduce(metrics[name])
            # dist.all_reduce(count)
            # n = count.item()
            # for name in losses.keys():
            #     losses[name] = losses[name] / n
            # for name in metrics.keys():
            #     metrics[name] = metrics[name] / n

            count = torch.tensor([n], dtype=torch.long, device=device)
            dist.all_reduce(count, dist.ReduceOp.SUM, async_op=False)
            n = count.item()

            for name, loss_ in losses.items():
                loss_ /= args.world_size
                dist.all_reduce(loss_, dist.ReduceOp.SUM, async_op=False)
                losses[name] = loss_
                total_loss = loss_ + total_loss
            for name, measure in metrics.items():
                if isinstance(measure, dict):
                    for subname, submeasure in measure.items():
                        submeasure /= args.world_size
                        dist.all_reduce(submeasure, dist.ReduceOp.SUM, async_op=False)
                        metrics[name][subname] = submeasure
                elif isinstance(measure, ConfusionMatrix) or isinstance(measure, Histogram):
                    # TODO:
                    #  If parallel, make use of this code
                    # metrics[name] = measure[0]
                    # for measure_i in range(1,len(measure)):
                    #     metrics[name].update(measure[measure_i])
                    metrics[name] = measure
                else:
                    if measure is None:
                        metrics[name] = None
                    else:
                        measure /= args.world_size
                        dist.all_reduce(measure, dist.ReduceOp.SUM, async_op=False)
                        metrics[name] = measure
        else:
            for name, loss_ in losses.items():
                losses[name] = torch.mean(loss_)
                total_loss = losses[name] + total_loss
            for name, measure in metrics.items():
                if isinstance(measure, dict):
                    for subname, submeasure in measure.items():
                        metrics[name][subname] = torch.sum(submeasure) / args.ngpus_per_node
                elif isinstance(measure, Histogram):
                    if name not in meters['metric'] or not hasattr(measure, 'update'):
                        meters['metric'][name] = measure
                    else:
                        meters['metric'][name].update(measure)
                elif isinstance(measure, ConfusionMatrix):
                    # TODO:
                    #  If parallel, make use of this code
                    # metrics[name] = measure[0]
                    # for measure_i in range(1,len(measure)):
                    #     metrics[name].update(measure[measure_i])
                    metrics[name] = measure
                else:
                    if measure is None:
                        metrics[name] = None
                    else:
                        metrics[name] = torch.sum(measure) / args.ngpus_per_node

        for name, loss_ in losses.items():
            if name not in meters['loss']:
                meters['loss'][name] = AverageMeter()
            meters['loss'][name].update(loss_.item(), n)
        for name, measure in metrics.items():
            if isinstance(measure, dict):
                if name not in meters['metric']:
                    meters['metric'][name] = OrderedDict()
                for subname, submeasure in measure.items():
                    if isinstance(submeasure, Histogram):
                        if subname not in meters['metric'][name] or not hasattr(submeasure, 'update'):
                            meters['metric'][name][subname] = submeasure
                        else:
                            meters['metric'][name][subname].update(submeasure)
                    else:
                        if subname not in meters['metric'][name]:
                            meters['metric'][name][subname] = AverageMeter()
                        meters['metric'][name][subname].update(submeasure.item(), n)
            else:
                if isinstance(measure, Histogram):
                    if name not in meters['metric'] or not hasattr(measure, 'update'):
                        meters['metric'][name] = measure
                    else:
                        meters['metric'][name].update(measure)
                elif isinstance(measure, ConfusionMatrix):
                    if name not in meters['metric']:
                        meters['metric'][name] = measure
                    else:
                        meters['metric'][name].update(measure)
                else:
                    if name not in meters['metric']:
                        meters['metric'][name] = AverageMeter()
                    if measure is not None:
                        before=meters['metric'][name].avg
                        meters['metric'][name].update(measure.item(), n)
                        # print('metric', name, 'before: {}, after: {} (value: {}, n={})'.format(before, meters['metric'][name].avg, measure.item(), n))

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.validate['print_freq'] == 0 and main_process():
            logger.info('Validation: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        '{losses} '
                        '[{metrics}]'.format(epoch+1, args.training['epochs'], i + 1, len(val_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          losses=' '.join(['{} {:.4f}'.format(name.title(), mtr.avg) for name, mtr in meters['loss'].items()]),
                                                          metrics=' '.join(['{} {:.1f}'.format(name.title(), mtr.avg) for name, mtr in meters['metric'].items() if not isinstance(mtr, dict) and not isinstance(mtr, Plottable)])))


    # if isinstance(model, BaseModel):
    #     model.post_training_hook()

    if main_process():
        logger.info('Validation result at epoch [{}/{}]: {losses} [{metrics}].'.format(epoch+1, args.training['epochs'], 
            losses=' '.join(['{} {:.4f}'.format(name.title(), mtr.avg) for name, mtr in meters['loss'].items()]),
            metrics=' '.join(['{} {:.1f}'.format(name.title(), mtr.avg) for name, mtr in meters['metric'].items() if not isinstance(mtr, dict) and not isinstance(mtr, Plottable)])))
        for name, measure in meters['metric'].items():
            if isinstance(measure, dict):
                logger.info('------------------------------------------------')
                logger.info('{}:'.format(name.title()))
                for subname, submeasure in measure.items():
                    logger.info('\t{} = {}'.format(subname, submeasure.avg))
        logger.info('\n')
    return meters


if __name__ == '__main__':
    main()
