#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import csv
import argparse
import csv
import glob
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial
from math import sqrt
import torchvision.utils
from timm import utils
import torch
import torch.nn as nn
import torch.nn.parallel


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from timm.loss import MSELoss
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.layers import apply_test_time_pool, set_fast_norm
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser, \
    decay_batch_step, check_batch_size_retry, ParseKwargs, reparameterize_model

try:
    from apex import amp

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')

"""添加用于回归训练的标记， 默认为回归训练"""
parser.add_argument('--is-regression', type=bool, default=True,
                    help='Determine whether it is a regression task')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in dataset split, for IterableDatasets.')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--input-key', default=None, type=str,
                    help='Dataset key for input images.')
parser.add_argument('--input-img-mode', default=None, type=str,
                    help='Dataset image conversion mode for input images.')
parser.add_argument('--target-key', default=None, type=str,
                    help='Dataset key for target labels.')

parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=0.888, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--crop-border-pixels', type=int, default=None,
                    help='Crop pixels from image border.')
parser.add_argument('--mean', type=float, nargs='+', default=(0., 0., 0.), metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=(1., 1., 1.), metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=20, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=True,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use, "native" or "apex" (default: native)')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--results-format', default='csv', type=str,
                    help='Format for results file one of (csv, json) (default: csv).')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')

def validate(args):


    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    # 初始化计时器和滑动平均计来衡量批量时间、损失、Top-1和Top-5准确率
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    """用于计算R^2"""
    output_list=[]
    target_list=[]

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)



    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_autocast = suppress
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            assert args.amp_dtype == 'float16'
            use_amp = 'apex'
            _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            assert args.amp_dtype in ('float16', 'bfloat16')
            use_amp = 'native'
            amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
            _logger.info('Validating in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    if args.fast_norm:
        set_fast_norm()

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=in_chans,
        global_pool=args.gp,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    if args.reparam:
        model = reparameterize_model(model)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if use_amp == 'apex':
        model = amp.initialize(model, opt_level='O1')

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    if not args.is_regression:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = MSELoss().to(device)

    root_dir = args.data or args.data_dir
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        download=args.dataset_download,
        load_bytes=args.tf_preprocessing,
        class_map=args.class_map,
        num_samples=args.num_samples,
        input_key=args.input_key,
        input_img_mode=input_img_mode,
        target_key=args.target_key,
        is_regression=True,
    )

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = [int(line.rstrip()) for line in f]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']  # crop_pct裁剪比例，新图占旧图的比例
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        crop_mode=data_config['crop_mode'],
        crop_border_pixels=args.crop_border_pixels,
        pin_memory=args.pin_mem,
        device=device,
        tf_preprocessing=args.tf_preprocessing,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    rmses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    laiRealPre=[]
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(device)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.to(device)
                input = input.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)

                if valid_labels is not None:
                    output = output[:, valid_labels]
                loss = criterion(output, target)
            """将推理结果记录到laiRealPre用于后续画图"""



            """计算当前batch的r2和累计的r2"""
            output_list.append(output.detach().cpu().numpy())
            target_list.append(target.cpu().numpy())

            batch_mse = mean_squared_error(target.cpu().numpy(), output.detach().cpu().numpy())
            sum_mse = mean_squared_error(np.concatenate(target_list), np.concatenate(output_list))

            batch_r2 = r2_score(target.cpu().numpy(), np.concatenate(output.detach().cpu().numpy()))
            sum_r2 = r2_score(np.concatenate(target_list), np.concatenate(output_list))

            batch_rmse = np.sqrt(mean_squared_error(target.cpu().numpy(), output.detach().cpu().numpy()))
            sum_rmse = np.sqrt(mean_squared_error(np.concatenate(target_list), np.concatenate(output_list)))

            batch_rrmse = batch_rmse / np.mean(target.cpu().numpy())
            sum_rrmse = sum_rmse / np.mean(np.concatenate(output_list))

            batch_mae = mean_absolute_error(target.cpu().numpy(), output.detach().cpu().numpy())
            sum_mae = mean_absolute_error(np.concatenate(target_list), np.concatenate(output_list))

            batch_mpe = np.abs((target.cpu().numpy() - output.detach().cpu().numpy()) / target.cpu().numpy())
            batch_mape = np.mean(batch_mpe) * 100
            sum_mpe = np.abs(
                (np.concatenate(target_list) - np.concatenate(output_list)) / np.concatenate(target_list))
            sum_mape = np.mean(sum_mpe) * 100

            batch_nrmse = batch_rmse / np.mean(target.cpu().numpy())
            sum_nrmse = sum_rmse / np.mean(np.concatenate(target_list))

            if real_labels is not None:
                real_labels.add_result(output)

            if not args.is_regression:
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))
            else:
                losses.update(loss.item(), input.size(0))
                rmses.update(sqrt(losses.avg))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                if not args.is_regression:
                    _logger.info(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                        'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                            batch_idx,
                            len(loader),
                            batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg,
                            loss=losses,
                            top1=top1,
                            top5=top5
                        )
                    )
                else:
                    # _logger.info(
                    #     'Test: [{0:>4d}/{1}]  '
                    #     'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    #     'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    #     'Rmse:{rmse:>7.4f} ({rmse_:>6.4f})'.format(
                    #         batch_idx,
                    #         len(loader),
                    #         batch_time=batch_time,
                    #         rate_avg=input.size(0) / batch_time.avg,
                    #         loss=losses,
                    #         rmse=sqrt(losses.val),
                    #         rmse_=sqrt(losses.avg)
                    #     )
                    # )
                    _logger.info(
                        f'Test: [{batch_idx:>4d}/{len(loader)}] '
                        f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f}) '
                        f'Loss: {losses.val:>7.3f} ({losses.avg:>6.3f}) '
                        f'Rmse: {batch_rmse:>6.3f} ({sum_rmse:>6.3f}) '
                        f'R2: {batch_r2:>6.3f} ({sum_r2:>6.3f}) '
                        f'rRmse: {batch_rrmse:>6.3f} ({sum_rrmse:>6.3f}) '
                        f'Mae: {batch_mae:>6.3f} ({sum_mae:>6.3f}) '
                        f'Mape: {batch_mape:>6.3f} ({sum_mape:>6.3f}) '
                        f'Nmse: {batch_nrmse:>6.3f} ({sum_nrmse:>6.3f}) '
                        f'Val_label: {set(target.cpu().numpy().tolist())}'
                    )

        # y_true = np.concatenate(target_list).tolist()
        # y_pred = np.concatenate(output_list).tolist()
        # complete = [[t,o[0]] for t,o in zip(y_true,y_pred)]
        # print("MAPE： ", mean_absolute_percentage_error(np.concatenate(target_list), np.concatenate(output_list)))



    if not args.is_regression:
        if real_labels is not None:
            # real labels mode replaces topk values at the end
            top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
        else:
            top1a, top5a = top1.avg, top5.avg
        results = OrderedDict(
            model=args.model,
            top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
            top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
            param_count=round(param_count / 1e6, 2),
            img_size=data_config['input_size'][-1],
            crop_pct=crop_pct,
            interpolation=data_config['interpolation'],
        )

        _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
            results['top1'], results['top1_err'], results['top5'], results['top5_err']))
    else:
        results = OrderedDict(
            model=args.model,
            loss=round(losses.avg,4),
            rmse=round(float(sum_rmse), 4),
            r2 = round(float(sum_r2), 6),
            rrmse = round(float(sum_rrmse), 6),
            mae = round(float(sum_mae), 6),
            mape=round(float(sum_mape), 6),
            nrmse=round(float(sum_nrmse), 6),
            param_count=round(param_count / 1e6, 2),
            img_size=data_config['input_size'][-1],
            crop_pct=crop_pct,
            interpolation=data_config['interpolation'],
        )

        _logger.info(' * loss {:.3f} rmse {:.3f}'.format(
            results['loss'], results['rmse'], results['r2'], results['rrmse'], results['mae'],results['mape'],results['nrmse']))

    """解包output_lst"""
    tmpOutputlst = []
    for item in output_list:    # item->[[], [], [],...]
        for item_ in item:
            tmpOutputlst.append(item_[0])
    tmpTarlst = []
    for item in target_list:    # item->[, , ,...]
        for item_ in item:
            tmpTarlst.append(item_)
    laiRealPre = [[targ, outp] for targ, outp in zip(tmpTarlst, tmpOutputlst)]


    return results, laiRealPre


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = 'Unknown'
    while batch_size:
        args.batch_size = batch_size * args.num_gpu  # multiply by num-gpu for DataParallel case
        try:
            if torch.cuda.is_available() and 'cuda' in args.device:
                torch.cuda.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            _logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        _logger.warning(f'Reducing batch size to {batch_size} for retry.')
    results['error'] = error_str
    _logger.error(f'{args.model} failed to validate ({error_str}).')
    return results


_NON_IN1K_FILTERS = ['*_in21k', '*_in22k', '*in12k', '*_dino', '*fcmae', '*seer']


def mean_absolute_percentage_error(y_true, y_pred):
    """
    计算 Mean Absolute Percentage Error (MAPE)

    参数:
    y_true (array-like): 真实值
    y_pred (array-like): 预测值

    返回:
    float: MAPE 值
    """
    # 确保输入是 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 避免除以零的情况
    if np.any(y_true == 0):
        raise ValueError("真实值中包含零，无法计算 MAPE")

    # 计算绝对百分比误差
    absolute_percentage_errors = np.abs((y_true - y_pred) / y_true)

    # 计算 MAPE
    mape = np.mean(absolute_percentage_errors) * 100

    return mape

def validate_norm():
    setup_default_logging()
    args = parser.parse_args()

    # args.model = model_name
    # args.is_regression = True
    # args.data_dir = data_dir
    args.batch_size = 32
    args.num_classes = 1
    args.img_size = 224
    args.device = "cuda"
    # args.checkpoint = model_ckpt
    result_csv = os.path.join("/".join(args.checkpoint.split("/")[:-1]),
                              f"{args.model}_{args.data_dir.split('/')[-1]}.csv")
    print(f"result_csv path: {result_csv}")

    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(
                pretrained=True,
                exclude_filters=_NON_IN1K_FILTERS,
            )
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(
                args.model,
                pretrained=True,
            )
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            initial_batch_size = args.batch_size
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                r = _try_run(args, initial_batch_size)
                if 'error' in r:
                    continue
                if args.checkpoint:
                    r['checkpoint'] = args.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
    else:
        if args.retry:
            results = _try_run(args, args.batch_size)
        else:
            start = time.time()
            results,laiRealPre = validate(args)
            end = time.time()
            print(end-start,' sec')

    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)

    if result_csv:
        write_to_csv(laiRealPre, result_csv)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')
    return results.get('r2')





def write_to_csv(data, filename='output.csv'):
    # 打开文件
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入标题行
        writer.writerow(['real', 'pre'])

        # 写入数据行
        for row in data:
            writer.writerow(row)

def write_results(results_file, results, format='csv'):
    with open(results_file, mode='w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()

model_dic={
    1:['ctflnet','Please fill in your model weight path']
}

if __name__ == '__main__':

    validate_norm()
