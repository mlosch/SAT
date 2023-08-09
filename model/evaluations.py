import os
import csv

from collections import OrderedDict
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import autoattack

import model.losses as losses
from model.metrics import Histogram
from lib import concept_alignment, util


class EvaluationBase(nn.Module):
    def __init__(self, eval_freq=1, robust_logit_index=None):
        super(EvaluationBase, self).__init__()
        self.eval_freq = eval_freq
        self.robust_logit_index = robust_logit_index

    def _construct_loss(self, loss_type, default_reduction='sum'):
        return losses.construct_loss(loss_type, default_reduction=default_reduction)

    def _init_optimizer(self, params, optim_cfg):
        cfg = dict(optim_cfg)
        optim = torch.optim.__dict__[cfg.pop('type')](params, **cfg)
        return optim

    def _forward_pass(self, model, inputs):
        outputs = model(inputs)
        predictions = outputs['prediction']
        if self.robust_logit_index is not None:
            if self.robust_logit_index == -1:
                return predictions[:, :-1], predictions[:, -1]
            elif self.robust_logit_index == 0:
                return predictions[:, 1:], predictions[:, 0]
            else:
                return torch.cat([predictions[:, :self.robust_logit_index], predictions[:, self.robust_logit_index+1:]], dim=1), predictions[:, self.robust_logit_index]
        return predictions, None

    def forward(self, **kwargs):
        raise NotImplementedError


class InputSpaceEvaluationBase(EvaluationBase):
    def __init__(self, data_mean, data_std, **kwargs):
        super(InputSpaceEvaluationBase, self).__init__(**kwargs)
        self.data_mean = torch.Tensor(data_mean)
        self.data_std = torch.Tensor(data_std)

    def _remove_normalization(self, x):
        # remove image normalization
        if x.device != self.data_mean.device:
            self.data_mean = self.data_mean.to(x.device)
            self.data_std = self.data_std.to(x.device)
        std = self.data_std[None, :, None, None]
        mean = self.data_mean[None, :, None, None]
        x = x * std + mean
        return x

    def _normalize(self, x):
        if x.device != self.data_mean.device:
            self.data_mean = self.data_mean.to(x.device)
            self.data_std = self.data_std.to(x.device)
        std = self.data_std[None, :, None, None]
        mean = self.data_mean[None, :, None, None]
        x = (x - mean) / std
        return x


class AUiCMetric(EvaluationBase):
    def __init__(self, layer, num_thresholds, broden_cfg, batchsize=32, debug_path=None, **kwargs):
        super(AUiCMetric, self).__init__(**kwargs)
        self.layer = dict(module=layer)
        self.batchsize = batchsize
        self.debug_path = debug_path
        self.broden_cfg = broden_cfg
        self.num_thresholds = num_thresholds

        assert 'ignore_index' in broden_cfg
        assert 'domain' in broden_cfg
        assert 'split' in broden_cfg
        assert 'data_root' in broden_cfg

    # def make_broden_loader(args, domain, shorten=False):
    #     test_transform = transform.Compose([transform.ToTensor()])
    #     dinfo = args['broden']
    #     test_data = dataset.BrodenData(split=dinfo['split'], concept_class=domain, data_root=dinfo['data_root'], transform=test_transform, shorten=True)
    #     print('BrodenData loaded {} samples for concept class {}'.format(len(test_data), domain))
    #     dinfo['classes'] = test_data.num_classes
    #     test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    #     return test_loader, dinfo

    def make_broden_loader(self):
        from torchvision import transforms
        from lib.broden_data import BrodenData

        test_transform = lambda img,seg: (torch.from_numpy(img).permute(2,0,1), torch.from_numpy(seg).long())
        dinfo = self.broden_cfg
        test_data = BrodenData(split=dinfo['split'], concept_class=dinfo['domain'], data_root=dinfo['data_root'], transform=test_transform, num_samples=dinfo.get('num_samples', None))
        if self.logger is not None:
            self.logger.info('AUiC :: BrodenData loaded {} samples for concept class {}'.format(len(test_data), dinfo['domain']))
        dinfo['classes'] = test_data.num_classes
        dinfo['num_samples'] = len(test_data)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        return test_loader, dinfo

    @staticmethod
    def miou(iou_stats):
        # mIoU:
        pos = (iou_stats[3] / iou_stats[1])
        neg = (iou_stats[2] / iou_stats[0])
        miou = 0.5*(pos + neg)
        return miou

    @staticmethod
    def auic(iou_stats):
        miou_ = AUiCMetric.miou(iou_stats)
        miou_[np.isnan(miou_)]=0
        
        x = np.linspace(0,1,miou_.shape[0])
        
        # theta_best
        auc = miou_.max(0)
        max_auc_per_unit = np.max(auc, 1)

        thresholds = np.linspace(0.5, 1.0, 101)
        unit_associable = (max_auc_per_unit > thresholds[..., np.newaxis])
        y = unit_associable.sum(-1)/float(unit_associable.shape[1])

        auc_x = np.linspace(0,1,len(thresholds))
        AUiC = np.trapz(x=auc_x, y=y)
        return AUiC


    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        logger = kwargs.get('logger', None)
        self.logger = logger
        loader = kwargs['train_loader']
        epoch = kwargs['epoch']

        broden_loader, dinfo = self.make_broden_loader()

        with torch.no_grad():
            calibrator = concept_alignment.LayerOutputLimits()
            calibrator.register_hooks(self.layer['module'])
            for i, (inputs, targets) in enumerate(loader):
                self._forward_pass(model, inputs.to(model_device))
                if i % (len(loader)//10) == 0:
                    if logger is not None:
                        logger.info('AUiC Output limit calibration [{}/{}]'.format(i, len(loader)))

            mins, maxs = calibrator.mins, calibrator.maxs
            calibrator.unregister_hooks()

            thresholds = torch.zeros((len(mins), self.num_thresholds))
            for ui in range(len(mins)):
                thresholds[ui] = torch.linspace(mins[ui], maxs[ui], self.num_thresholds)

            evaluater = concept_alignment.AUIC(thresholds, 
                {self.broden_cfg['domain']: dinfo['classes']}, 
                {self.broden_cfg['domain']: self.broden_cfg['ignore_index']}, 
                batchsize=self.batchsize)
            evaluater.register_hooks(self.layer['module'])

            for i, (inputs, targets) in enumerate(broden_loader):
                inputs = inputs.to(model_device)
                evaluater.register_input_sample(inputs, [targets], [self.broden_cfg['domain']])
                self._forward_pass(model, inputs)

                if i % (len(broden_loader)//10) == 0:
                    if logger is not None:
                        iou_stats = evaluater.results_by_category[dinfo['domain']]
                        auic = self.auic(iou_stats)
                        logger.info('AUiC[{}] [{}/{}]: {}'.format(self.broden_cfg['domain'], i, len(broden_loader), auic))

            iou_stats = evaluater.results_by_category[dinfo['domain']]

        return {self.broden_cfg['domain'] : self.auic(iou_stats)}


class LocalizationException(Exception):
    pass


class LocalizationMetric(EvaluationBase):
    def __init__(self, n_samples, num_classes, multi_image_size, attribution_config, debug_path=None, **kwargs):
        super(LocalizationMetric, self).__init__(**kwargs)
        assert multi_image_size in [4, 9]
        self.num_classes = num_classes
        self.n_samples = n_samples
        self.multi_image_size = multi_image_size
        self.debug_path = debug_path

        self.attributions = OrderedDict()
        self.attr_kwargs = OrderedDict()
        self.properties = dict(model=None)

        for name, cfg in attribution_config.items():
            attr_type = cfg.pop('type')

            if attr_type.startswith('captum.attr.'):
                import captum.attr as captum
                attr_type = attr_type.replace('captum.attr.', '')
                self.attr_kwargs[name] = cfg.pop('kwargs', {})
                attr_instance = captum.__dict__[attr_type](self._captum_forward, **cfg)
            else:
                raise NotImplementedError

            self.attributions[name] = attr_instance


    def _captum_forward(self, inputs):
        # for captum only return network output
        return self._forward_pass(self.properties['model'], inputs)[0]

    def compute_sorted_confs(self, dataloader, model):
        """
        Sort image indices by the confidence of the classifier and store in sorted_confs.
        Returns: None
        """
        # save_path = join(self.trainer.save_path, "localisation_analysis", "epoch_{}".format(self.trainer.epoch))
        # fp = join(save_path, self.conf_fn)

        confidences = {i: [] for i in range(self.num_classes)}
        model_device = torch.ones(1).cuda().device

        loader = dataloader
        img_idx = -1
        with torch.no_grad():
            tgt9 = 0
            for img, tgt in loader:
                img, tgt = img.to(model_device), tgt.to(model_device)
                output, _ = self._forward_pass(model, img)
                output = F.softmax(output, dim=1)
                logits, classes = output.max(1)
                # logits, classes = trainer.predict(img, to_probabilities=False).max(1)
                for logit, pd_class, gt_class in zip(logits, classes, tgt): #tgt.argmax(1)):
                    img_idx += 1
                    if pd_class != gt_class:
                        continue
                    confidences[int(gt_class.item())].append((img_idx, logit.item()))

        for k, vlist in confidences.items():
            confidences[k] = sorted(vlist, key=lambda x: x[1], reverse=True)

        return confidences

    def get_sorted_indices(self, confidences):
        """
        This method generates a list of indices to be used for sampling from the dataset and evaluating the
            multi images.
        In particular, the images per class are sorted by their confidence.
        Then, a random set of n classes (for the multi image) is sampled and for each class the next
            most confident image index that was not used yet is added to the list.
        Thus, when using this list for creating multi images, the list contains blocks of size n with
        image indices such that (1) each class occurs at most once per block and (2) the class confidences
            decrease per block for each class individually.
        Returns: list of indices
        """
        idcs = []
        classes = np.array([k for k in confidences.keys()])
        class_indexer = {k: 0 for k in classes}

        # Only use images with a minimum confidence of 50%
        # This is, of course, the same for every attribution method
        def get_conf_mask_v(_c_idx):
            return torch.tensor(confidences[_c_idx][class_indexer[_c_idx]][1]).item() > 0.5
        # Only use classes that are still confidently classified
        
        for k in range(len(confidences)):
            if len(confidences[k]) == 0:
                raise LocalizationException('Not enough examples for at least one class: {}\n Accidentally set robust_logit_index?'.format(str([(clazz, len(conf)) for clazz, conf in confidences.items()])))
        
        mask = np.array([get_conf_mask_v(k) for k in classes])
        n_imgs = self.multi_image_size

        if mask.sum() <= n_imgs:
            raise LocalizationException('Not enough confident examples')

        # Always use the same set of classes for a particular model
        np.random.seed(42)
        # print(mask.sum(), n_imgs)
        # print(mask)
        
        while mask.sum() > n_imgs:
            # Of the still available classes, sample a set of classes randomly
            sample = np.random.choice(classes[mask], size=n_imgs, replace=False)

            for c_idx in sample:
                # Store the corresponding index of the next class image for each of the randomly sampled classes
                img_idx, conf = confidences[c_idx][class_indexer[c_idx]]
                class_indexer[c_idx] += 1
                mask[c_idx] = get_conf_mask_v(c_idx) if class_indexer[c_idx] < len(confidences[c_idx]) else False
                idcs.append(img_idx)

                # print(c_idx, mask.sum(), n_imgs)
        return idcs

    @staticmethod
    def make_multi_image(n_imgs, dataset, offset=0, fixed_indices=None):
        """
        From the offset position takes the next n_imgs that are of different classes according to the order in the
        dataset or fixed_indices .
        Args:
            n_imgs: how many images should be combined for a multi images
            dataset: dataset
            offset: current offset
            fixed_indices: whether or not to use pre-defined indices (e.g., first ordering images by confidence).
        Returns: the multi_image, the targets in the multi_image and the new offset
        """
        assert n_imgs in [4, 9]
        tgts = []
        imgs = []
        count = 0
        i = 0
        if fixed_indices is not None:
            mapper = fixed_indices
        else:
            mapper = list(range(len(dataset)))

        # Going through the dataset to sample images
        while count < n_imgs:
            idx = mapper[(i + offset) % len(mapper)]
            img, tgt_idx = dataset[idx]
            i += 1
            # if the class of the new image is already added to the list of images for the multi-image, skip this image
            # This should actually not happen since the indices are sorted in blocks of 9 unique labels
            if tgt_idx in tgts:
                continue
            imgs.append(img[None])
            tgts.append(tgt_idx)
            count += 1
        img = torch.cat(imgs, dim=0)
        img = img.view(-1, int(np.sqrt(n_imgs)), int(np.sqrt(n_imgs)), *img.shape[-3:]).permute(0, 3, 2, 4, 1, 5).reshape(
            -1, img.shape[1], img.shape[2] * int(np.sqrt(n_imgs)), img.shape[3] * int(np.sqrt(n_imgs)))
        return img.cuda(), tgts, i + offset + 1


    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.properties['model'] = model
        self.logger = kwargs.get('logger', None)
        loader = kwargs['val_loader']
        epoch = kwargs['epoch']

        with torch.no_grad():

            sample_size, n_imgs = self.n_samples, self.multi_image_size

            try:
                confidences = self.compute_sorted_confs(loader, model)
                fixed_indices = self.get_sorted_indices(confidences)
            except LocalizationException as e:
                if self.logger is not None:
                    self.logger.warn(e)
                return {}
            
            offset = 0
            single_shape = int(loader.dataset[0][0].shape[-1])

            results = OrderedDict()
            for attr_name in self.attributions.keys():
                results[attr_name] = []

            sample_size = min(len(fixed_indices), sample_size)
            assert sample_size > 0

            for count in range(sample_size):
                # if offset >= sample_size:
                #     if self.logger is not None:
                #         self.logger.info('LocalizationMetric: Not enough correctly predicted images to continue evaluation.')
                #     break
                multi_img, tgts, offset = self.make_multi_image(n_imgs, loader.dataset, offset=offset,
                                                                fixed_indices=fixed_indices)
                # calculate the attributions for all classes that are participating and only save positive contribs
                for attr_name, attributer in self.attributions.items():
                    
                    attributions = torch.zeros((len(tgts),)+multi_img.shape[1:]).float().to(model_device)
                    # print(count, sample_size, attr_name, len(tgts))
                    for tgt_idx in range(len(tgts)):
                        attribution = attributer.attribute(multi_img, target=tgts[tgt_idx], **self.attr_kwargs[attr_name])[0]
                        if attribution.shape[-2:] != multi_img.shape[-2:]:
                            # if the attribution returns a downsampled output (e.g. GradCam), upsample to input image size
                            attribution = F.interpolate(attribution.unsqueeze(0), multi_img.shape[-2:], mode='nearest').squeeze(0)
                        attributions[tgt_idx] = attribution

                    if count == 0 and self.debug_path is not None:
                        # prepare file system
                        if not os.path.exists(self.debug_path):
                            os.makedirs(self.debug_path)

                        save_fp = os.path.join(self.debug_path, 'epoch_{:03d}_{}_count_{:03d}.png'.format(epoch, attr_name, count))

                        grid_images = []
                        ncol = int(np.sqrt(n_imgs))
                        for grid_i in range(n_imgs):
                            xi = int(grid_i % ncol)
                            yi = int(grid_i // ncol)
                            #grid_images.append(multi_img[0,:,yi*single_shape:(yi+1)*single_shape,xi*single_shape:(xi+1)*single_shape])
                            grid_images.append(multi_img[0])
                            grid_images.append(attributions[grid_i].permute(0,2,1)/attributions[grid_i].max())
                        grid = make_grid(grid_images, pad_value=0.5, normalize=False, nrow=ncol*2)
                        save_image(grid, fp=save_fp)

                    attributions = attributions.sum(1, keepdim=True).clamp(0)
                    # Calculate the relative amount of attributions per region. Use avg_pool for simplicity.
                    contribs = F.avg_pool2d(attributions, single_shape, stride=single_shape).permute(0, 1, 3, 2).reshape(
                        attributions.shape[0], -1)
                    total = contribs.sum(1, keepdim=True)
                    contribs = torch.where(total * contribs > 0, contribs/total, torch.zeros_like(contribs)).cpu().numpy()
                    results[attr_name].append([contrib[idx] for idx, contrib in enumerate(contribs)])

                if count % (max(1,sample_size//10)) == 0:
                    if self.logger is not None:
                        self.logger.info("LocalizationMetric: {:>6.2f}% of processing complete".format(100*(count+1.)/sample_size))
            
            for attr_name in list(results.keys()):
                if self.logger is not None:
                    values = np.array(results[attr_name]).flatten()
                    # self.logger.info("LocalizationMetric: [{}] Raw values: {}".format(attr_name, values))
                    self.logger.info("LocalizationMetric: [{}] Percentiles of localisation accuracy (25, 50, 75, 100): {}".format(attr_name, str(np.percentile(values, [25, 50, 75, 100]))))
                results[attr_name+'_mean'] = torch.Tensor(results[attr_name]).mean()
                results[attr_name] = Histogram(torch.Tensor(results[attr_name]).view(-1))

        # release model property
        self.properties['model'] = None
        return results


class LocalLowerLipschitzBoundEstimationViaGradientNorm(InputSpaceEvaluationBase):
    def __init__(self, num_classes, n_samples, loss_type, norm_p=None, dataset='val', **kwargs):
        super(LocalLowerLipschitzBoundEstimationViaGradientNorm, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.n_samples = n_samples
        self.logger = None
        self.dataset_key = dict(val='val_loader', train='train_loader')[dataset]
        self.norm_p = None

        self.loss = self._construct_loss(loss_type)

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        loader = kwargs[self.dataset_key]
        epoch = kwargs['epoch']

        if self.norm_p is None:
            norm_p = 2 #model.norm_p
        else:
            norm_p = self.norm_p

        max_grad_norm = None
        all_grad_norms = []

        batch_size = 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            inputs.requires_grad = True

            batch_size = max(inputs.shape[0], batch_size)

            for class_idx in range(self.num_classes):
                targets.fill_(class_idx)

                output, _ = self._forward_pass(model, inputs)
                loss = self.loss(output, targets)
                loss.backward()

                with torch.no_grad():
                    grads = inputs.grad
                    grads = self._remove_normalization(grads)
                    grad_norms = torch.norm(grads, p=norm_p, dim=[1,2,3])
                    if max_grad_norm is None:
                        max_grad_norm = torch.max(grad_norms)
                    else:
                        max_grad_norm = torch.max(max_grad_norm, torch.max(grad_norms))

                    all_grad_norms.append(grad_norms)
                    inputs.grad.zero_()

            num_batches = np.ceil(self.n_samples / batch_size)

            if batch_idx % (num_batches//10) == 0:
                if self.logger is not None:
                    self.logger.info('LocalLowerLipschitzBoundEstimationViaGradientNorm: [{}/{}] Lower bound estimate: {:.4f}'.format(batch_idx, num_batches, max_grad_norm.item()))

            if batch_idx >= num_batches:
                break

        if self.logger is not None:
            self.logger.info('LocalLowerLipschitzBoundEstimationViaGradientNorm: [Done] Lower bound estimate: {:.4f}'.format(max_grad_norm.item()))

        all_grad_norms = torch.cat(all_grad_norms)
        results = dict(lower_bound=max_grad_norm, gradient_norms=Histogram(all_grad_norms))
        return results


class LowerLipschitzBoundEstimation(InputSpaceEvaluationBase):
    def __init__(self, n_samples, batch_size, optimizer_cfg, max_iter, dataset='val', input_norm_correction=1.0, **kwargs):
        super(LowerLipschitzBoundEstimation, self).__init__(**kwargs)
        assert batch_size % 2 == 0, 'batch_size must be multiple of 2'
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.logger = None
        self.dataset_key = dict(val='val_dataset', train='train_dataset')[dataset]
        self.input_norm_correction = input_norm_correction

        assert 'type' in optimizer_cfg
        self.optimizer_cfg = optimizer_cfg

    def lbo_loss(self, model, inputs):        
        outputs, _ = self._forward_pass(model, self._normalize(inputs))
        prediction_values, predictions = outputs.max(dim=1)

        N = inputs.shape[0] // 2
        X1, X2 = inputs[:N], inputs[N:]
        y1, y2 = outputs[:N], outputs[N:]

        j = predictions[:N]

        y1_j = prediction_values[:N]
        y2_j = torch.gather(y2, dim=1, index=j.unsqueeze(1))

        margin1 = y1_j.unsqueeze(1) - y1
        margin2 = y2_j - y2

        L = torch.abs(margin1 - margin2) / torch.norm(X1-X2, p=2, dim=[1,2,3]).unsqueeze(1)
        L = L * self.input_norm_correction

        loss = -L.max(dim=1)[0]

        return loss

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs[self.dataset_key]
        epoch = kwargs['epoch']

        all_inputs, all_labels = [], []

        im_inds = torch.randperm(len(dataset))
        
        for i in range(self.n_samples*2):
            image, label = dataset[im_inds[i]]
            all_inputs.append(image.unsqueeze(0))
            all_labels.append(label)

        max_loss = None

        for j, bs_idx in enumerate(range(0, self.n_samples*2, self.batch_size)):
            inputs = torch.cat(all_inputs[bs_idx:bs_idx+self.batch_size], dim=0).to(model_device)
            labels = torch.Tensor(all_labels[bs_idx:bs_idx+self.batch_size]).to(model_device)

            inputs = self._remove_normalization(inputs)

            inputs.requires_grad = True

            optimizer = self._init_optimizer([inputs], self.optimizer_cfg)

            for i in range(self.max_iter):
                loss = self.lbo_loss(model, inputs)
                loss.sum().backward()

                L = -loss

                if max_loss is None:
                    max_loss = L.max().detach()
                else:
                    with torch.no_grad():
                        max_loss = torch.max(max_loss, L.max().detach())

                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    # mean = self.data_mean
                    # std = self.data_std
                    inputs.clamp_(0,1)
                    # inputs[:,0].clamp_(-mean[0]/std[0],(1-mean[0])/std[0])
                    # inputs[:,1].clamp_(-mean[1]/std[1],(1-mean[1])/std[1])
                    # inputs[:,2].clamp_(-mean[2]/std[2],(1-mean[2])/std[2])

                if i % 100 == 0:
                    if self.logger is not None:
                        sum_max_iter = self.max_iter * ((self.n_samples*2)/self.batch_size)
                        cur_iter = j*self.max_iter + i
                        self.logger.info('LowerLipschitzBoundEstimation: [{}/{}] Lower bound estimate: {:.4f}'.format(cur_iter, sum_max_iter, max_loss.item()))

        results = dict(lower_bound=max_loss)
        return results

class PGDAttackEvaluationBase(InputSpaceEvaluationBase):
    def __init__(self, **kwargs):
        super(PGDAttackEvaluationBase, self).__init__(**kwargs)

    def _step_and_project(self, eps, original_inputs, inputs, grad, lrs):
        # mean = self.data_mean
        # std = self.data_std

        original_inputs = self._remove_normalization(original_inputs)

        l = len(inputs.shape) - 1
        g = grad #inputs.grad
        if self.norm_p == 2:
            # g = self._adam_update(g)
            # scaled_g = g
            
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
            scaled_g = g / (g_norm + 1.e-10)

            step = scaled_g * lrs[:, None, None, None]
            diff = self._remove_normalization(inputs - step) - original_inputs
            diff = diff.renorm(p=2, dim=0, maxnorm=eps)
        elif np.isinf(self.norm_p):
            step = torch.sign(g) * lrs[:, None, None, None]
            diff = self._remove_normalization(inputs - step) - original_inputs
            diff = torch.clamp(diff, -eps, eps)
        else:
            raise NotImplementedError

        inputs = original_inputs + diff
        inputs.clamp_(0,1)
        inputs = self._normalize(inputs)

        return inputs


class GDAccuracy(EvaluationBase):
    def __init__(self, num_classes, max_iter,
        n_samples, batch_size, loss_type,
        data_mean, data_std, epsilons, optimizer_cfg, norm_p=None, **kwargs):
        super(GDAccuracy, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.save_path = None #save_path
        self.max_iter = max_iter
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.data_mean = data_mean
        self.data_std = data_std
        self.logger = None
        self.epsilons = epsilons
        self.norm_p = norm_p

        if norm_p is None:
            raise RuntimeError('norm_p must be defined')

        self.loss = self._construct_loss(loss_type)

        assert 'type' in optimizer_cfg
        assert 'lr' in optimizer_cfg
        self.optimizer_cfg = optimizer_cfg

    def _remove_normalization(self, inputs):
        # remove image normalization
        std = torch.Tensor(self.data_std).to(inputs.device)[None, :, None, None]
        mean = torch.Tensor(self.data_mean).to(inputs.device)[None, :, None, None]
        inputs = inputs * std + mean
        return inputs

    def _init_optimizer(self, inputs, optim_cfg):
        cfg = dict(optim_cfg)
        inputs.requires_grad = True
        params = [inputs]
        optim = torch.optim.__dict__[cfg.pop('type')](params, **cfg)
        return optim

    def _attack(self, model, inputs, labels):
        # initial forward pass to acquire correct predicted samples
        with torch.no_grad():
            output, robust = self._forward_pass(model, inputs)
            _, predictions = output.max(dim=1)
            correct_preds = predictions.eq(labels)

            inputs = inputs[correct_preds]
            labels = labels[correct_preds]
            original_predictions = predictions[correct_preds].detach()
            original_inputs = inputs.detach().clone()

        success = torch.full((inputs.shape[0],), 0, dtype=torch.bool).to(inputs.device)
        epsilons = torch.norm(self._remove_normalization(original_inputs) - \
                        self._remove_normalization(inputs), p=2, dim=[1,2,3])

        optimizer = self._init_optimizer(inputs, self.optimizer_cfg)

        mean = self.data_mean
        std = self.data_std

        iteration = 0
        while success.sum() < inputs.shape[0] and iteration < self.max_iter:
            output, robust = self._forward_pass(model, inputs)
            pred_values, predictions = output.max(dim=1)

            with torch.no_grad():
                not_success = predictions.eq(original_predictions)
                # we only update epsilons of samples that have not yet been attacked successfully
                new_epsilons = torch.norm(self._remove_normalization(original_inputs) - \
                        self._remove_normalization(inputs), p=2, dim=[1,2,3])
                epsilons[not_success] = new_epsilons[not_success]
                success = ~not_success

            # update
            loss = -self.loss(output, labels)
            loss.backward(retain_graph=False)

            optimizer.step()

            with torch.no_grad():
                # inputs -= lrs[:, None, None, None]*inputs.grad
                inputs[:,0].clamp_(-mean[0]/std[0],(1-mean[0])/std[0])
                inputs[:,1].clamp_(-mean[1]/std[1],(1-mean[1])/std[1])
                inputs[:,2].clamp_(-mean[2]/std[2],(1-mean[2])/std[2])
                # inputs.grad.fill_(0)

            optimizer.zero_grad()

            iteration += 1

            if iteration % 1000 == 0:
                if self.logger is not None:
                    self.logger.info('GDAccuracy: [{}/{}] Success rate: {:.2f}. Avg epsilon: {:.4f}'.format(iteration, self.max_iter, success.sum()/float(inputs.shape[0]), epsilons[success].mean().item()))
        
        return success, epsilons, correct_preds

    def _sample_loader(self, dataset):
        im_inds = torch.randperm(len(dataset))
        
        for i in range(self.n_samples):
            image, label = dataset[im_inds[i]]
            yield image, label

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs['val_dataset']
        epoch = kwargs['epoch']

        all_labels = []
        all_success = []
        all_epsilons = []
        all_correct = []
        

        inputs = []
        labels = []
        for image, label in self._sample_loader(dataset):
            inputs.append(image.unsqueeze(0))
            labels.append(label)

            if len(inputs) == self.batch_size:
                inputs = torch.cat(inputs, dim=0).to(model_device)
                labels = torch.Tensor(labels).long().to(model_device)

                success, epsilons, correct = self._attack(model, inputs, labels)

                all_success.append(success.detach().cpu())
                all_epsilons.append(epsilons.detach().cpu())
                all_correct.append(correct.detach().cpu())

                inputs = []
                labels = []

        if len(inputs) > 0:
            success, epsilons, correct = self._attack(model, inputs, labels)

            all_success.append(success.detach().cpu())
            all_epsilons.append(epsilons.detach().cpu())
            all_correct.append(correct.detach().cpu())

        all_success = torch.cat(all_success, dim=0)
        all_epsilons = torch.cat(all_epsilons, dim=0)
        all_correct = torch.cat(all_correct, dim=0)

        if self.logger is not None:
            self.logger.info('GDAccuracy: [Done] Success rate: {:.2f}. Avg epsilon: {:.4f}'.format(all_success.sum()/float(all_success.shape[0]), all_epsilons[all_success].mean().item()))

        # self.save_results(all_inputs, all_targets, all_original_preds, all_success, all_epsilons, epoch)

        eps_robust = all_epsilons.unsqueeze(1) > torch.Tensor(self.epsilons)

        correct_and_robust = eps_robust
        robust_acc = correct_and_robust.float().mean(0)
        # clean_acc = all_correct.mean()

        results = dict(fraction_success=all_success.float().mean(), 
            avg_epsilon=all_epsilons[all_success].mean())

        for eps, rob_acc in zip(self.epsilons, robust_acc):
            results['robust_acc_{:.3f}'.format(eps)] = rob_acc

        return results


class PGDAccuracy(PGDAttackEvaluationBase):
    def __init__(self, num_classes, max_iter,
        n_samples, batch_size, loss_type,
        epsilons, lr, lr_scheduler, norm_p=None, **kwargs):
        super(PGDAccuracy, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.save_path = None #save_path
        self.max_iter = max_iter
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.logger = None
        self.epsilons = epsilons
        self.norm_p = norm_p
        self.lr = lr
        self.lr_scheduler_cfg = lr_scheduler
        self.adam_state = {}
        self.adam_betas = (0.9, 0.999)

        if norm_p is None:
            raise RuntimeError('norm_p must be defined')

        self.loss = self._construct_loss(loss_type)

    def _construct_lrscheduler(self, cfg):
        sched_type = cfg.pop('type')
        scheduler = StepLRScheduler(base_lr=self.lr, **cfg)
        cfg['type'] = sched_type
        return scheduler

    def _adam_update(self, grad):
        exp_avgs = []
        exp_avg_sqs = []

        beta1, beta2 = self.adam_betas
        state = self.adam_state

        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(grad, memory_format=torch.preserve_format).to(grad.device)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(grad, memory_format=torch.preserve_format).to(grad.device)

        state['step'] += 1

        state['exp_avg'] = beta1 * state['exp_avg'] + (1.0 - beta1) * grad
        state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1.0 - beta2) * (grad**2)

        # include exp moving average bias corrections
        m_t = state['exp_avg'] / (1.0 - beta1**state['step'])
        v_t = state['exp_avg_sq'] / (1.0 - beta2**state['step'])

        return m_t / (torch.sqrt(v_t).clamp_(1.e-8))

    def _random_perturb(self, x, eps):
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return torch.clamp(x + eps * rp / (rp_norm + 1e-10), 0, 1)

    def _attack(self, eps, model, inputs, labels):
        lrs = torch.full((inputs.shape[0],), 1.0).to(inputs.device)
        
        # reset adam state
        self.adam_state = {}

        original_inputs = inputs.detach().clone()

        success = torch.full((inputs.shape[0],), 0, dtype=torch.bool).to(inputs.device)
        epsilons = torch.norm(self._remove_normalization(original_inputs) - \
                        self._remove_normalization(inputs), p=2, dim=[1,2,3])

        # inputs = self._random_perturb(inputs, eps)

        lr_scheduler = None
        if self.lr_scheduler_cfg is not None:
            lr_scheduler = self._construct_lrscheduler(self.lr_scheduler_cfg)

        mean = self.data_mean
        std = self.data_std

        original_predictions = None

        iteration = 0
        while success.sum() < inputs.shape[0] and iteration < self.max_iter:
            inputs = inputs.clone().detach().requires_grad_(True)

            output, _ = self._forward_pass(model, inputs)
            pred_values, predictions = output.max(dim=1)

            with torch.no_grad():
                if original_predictions is None:
                    unsuccessful_attack = torch.ones_like(predictions).bool()
                    original_predictions = predictions.detach().clone()
                else:
                    unsuccessful_attack = predictions.eq(original_predictions)

                # we only update epsilons of samples that have not yet been attacked successfully
                epsilons = torch.norm(self._remove_normalization(original_inputs) - \
                    self._remove_normalization(inputs), p=2, dim=[1,2,3])
                # epsilons[unsuccessful_attack] = new_epsilons[unsuccessful_attack]
                success = ~unsuccessful_attack
                lrs[success] = 0

            # update
            loss = -self.loss(output, labels)
            # loss.backward(retain_graph=False)
            grad = torch.autograd.grad(loss, [inputs], grad_outputs=torch.ones_like(loss))[0]
            with torch.no_grad():
                inputs = self._step_and_project(eps, original_inputs, inputs, grad, lrs*lr_scheduler.get_last_lr())

            lr_scheduler.step()

            iteration += 1

            if iteration % (self.max_iter//10) == 0:
            # if self.logger is not None:
                self.logger.info('PGDAccuracy: [{}/{}]{{eps={:.3f}}} Robust acc: {:.2f}. Avg eps: (suc/unsuc) ({:.4f}, {:.4f}), lr: {:.4f}'.format(iteration, self.max_iter, eps, 1.0-success.sum()/float(inputs.shape[0]), epsilons[success].mean().item(), epsilons[~success].mean().item(), lr_scheduler.get_last_lr()))
        
        return success, epsilons

    def _get_correct_preds(self, model, inputs, labels):
        # initial forward pass to acquire correct predicted samples
        with torch.no_grad():
            output, _ = self._forward_pass(model, inputs)
            _, predictions = output.max(dim=1)
            correct_preds = predictions.eq(labels)

        return correct_preds

    def _sample_loader(self, dataset):
        im_inds = torch.randperm(len(dataset))
        
        for i in range(self.n_samples):
            image, label = dataset[im_inds[i]]
            yield image, label

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs['val_dataset']
        epoch = kwargs['epoch']

        all_norms_by_eps = {eps: [] for eps in self.epsilons}
        all_success_by_eps = {eps: [] for eps in self.epsilons}
        all_initial_correct = []
        
        inputs = []
        labels = []
        for image, label in self._sample_loader(dataset):
            inputs.append(image.unsqueeze(0))
            labels.append(label)

            if len(inputs) == self.batch_size:
                inputs = torch.cat(inputs, dim=0).to(model_device)
                labels = torch.Tensor(labels).long().to(model_device)

                correct_preds = self._get_correct_preds(model, inputs, labels)
                # correct_preds = torch.ones(inputs.shape[0], device=inputs.device).bool()
                all_initial_correct.append(correct_preds.cpu())

                for eps in self.epsilons:
                    success, diff_norms = self._attack(eps, model, inputs[correct_preds], labels[correct_preds])

                    all_norms_by_eps[eps].append(diff_norms.detach().cpu())
                    all_success_by_eps[eps].append(success.detach().cpu())

                inputs = []
                labels = []

        if len(inputs) > 0:
            inputs = torch.cat(inputs, dim=0).to(model_device)
            labels = torch.Tensor(labels).long().to(model_device)

            correct_preds = self._get_correct_preds(model, inputs, labels)
            all_initial_correct.append(correct_preds.cpu())

            for eps in self.epsilons:
                success, diff_norms = self._attack(eps, model, inputs, labels)

                all_norms_by_eps[eps].append(diff_norms.detach().cpu())
                all_success_by_eps[eps].append(success.detach().cpu())


        # Tally end results
        for key, values in list(all_success_by_eps.items()):
            all_success_by_eps[key] = torch.cat(values, dim=0)
        for key, values in list(all_norms_by_eps.items()):
            all_norms_by_eps[key] = torch.cat(values, dim=0)

        all_initial_correct = torch.cat(all_initial_correct, dim=0)

        # if self.logger is not None:
            # self.logger.info('PGDAttack: [Done] Success rate: {:.2f}. Avg epsilon: {:.4f}'.format(all_correct_by_eps.sum()/float(all_success.shape[0]), all_epsilons[all_success].mean().item()))

        if self.logger is not None:
            self.logger.info('PGDAccuracy: [Done] Tallied results:')
        results = {}
        for eps, rob_acc in all_success_by_eps.items():
            key = 'robust_acc_{:.3f}'.format(eps)
            results[key] = 1.0-rob_acc.float().mean()
            if self.logger is not None:
                self.logger.info(' {} = {:.2f}'.format(key, results[key]))

        return results


class AutoAttackAccuracy(InputSpaceEvaluationBase):
    def __init__(self, n_samples, batch_size,
        epsilons, norm_p=2, dataset='val', 
        attacks_to_run=None, seed=90622, 
        eval_mode=False, save_path=None, **kwargs):
        super(AutoAttackAccuracy, self).__init__(**kwargs)
        self.save_path = None #save_path
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.logger = None
        self.epsilons = epsilons
        self.norm_p = norm_p
        self.dataset_key = dict(val='val_dataset', train='train_dataset')[dataset]
        self.attacks_to_run = attacks_to_run
        self.seed = seed
        self.eval_mode = eval_mode
        self.save_path = save_path

    def normalizing_forward_pass(self, model, x):
        x = self._normalize(x)
        preds, _ = self._forward_pass(model, x)
        return preds

    def _get_correct_preds(self, model, inputs, labels):
        # initial forward pass to acquire correct predicted samples
        with torch.no_grad():
            output, _ = self._forward_pass(model, inputs)
            _, predictions = output.max(dim=1)
            correct_preds = predictions.eq(labels)

        return correct_preds

    def _sample_loader(self, dataset):
        torch.manual_seed(self.seed)
        im_inds = torch.randperm(len(dataset))
        
        if self.eval_mode:
            # e.g. from 1000 to 10.000
            gen = range(len(dataset)-self.n_samples, self.n_samples)
        else:
            # e.g. from 0 to 1000
            gen = range(min(len(dataset), self.n_samples))

        for i in gen:
            tuple_ = dataset[im_inds[i]]
            image, label = tuple_[0], tuple_[1]
            yield image, label

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs[self.dataset_key]
        loader = kwargs['val_loader']
        epoch = kwargs['epoch']

        norm_p = {1: 'L1', 2: 'L2', np.inf: 'Linf'}[self.norm_p]

        adversaries = {}
        for epsilon in self.epsilons:
            if self.attacks_to_run is not None:
                adversaries[epsilon] = autoattack.AutoAttack(
                    partial(self.normalizing_forward_pass, model), 
                    norm=norm_p, eps=epsilon, version='custom', attacks_to_run=self.attacks_to_run, verbose=True)
            else:
                adversaries[epsilon] = autoattack.AutoAttack(
                    partial(self.normalizing_forward_pass, model), 
                    norm=norm_p, eps=epsilon, version='standard', verbose=True)

        all_norms_by_eps = {eps: [] for eps in self.epsilons}
        all_success_by_eps = {eps: [] for eps in self.epsilons}
        all_initial_correct = []


        all_initial_correct = []
        inputs = []
        labels = []
        for image, label in self._sample_loader(dataset):
            inputs.append(image.unsqueeze(0))
            labels.append(label)

            if len(inputs) == self.batch_size:
                inputs = torch.cat(inputs, dim=0).to(model_device)
                labels = torch.Tensor(labels).long().to(model_device)

                correct_preds = self._get_correct_preds(model, inputs, labels)
                all_initial_correct.append(correct_preds.cpu())

                # remove normalization
                inputs = self._remove_normalization(inputs)

                for epsilon in self.epsilons:
                    adversary = adversaries[epsilon]
                    with torch.no_grad():
                        x_adv, y_adv = adversary.run_standard_evaluation(inputs[correct_preds], labels[correct_preds], bs=min(1000,self.batch_size), return_labels=True)
                        diff_norms = ((x_adv - inputs[correct_preds]) ** 2).reshape(x_adv.shape[0], -1).sum(-1).sqrt()
                        success = (~labels[correct_preds].eq(y_adv)).float()

                    all_norms_by_eps[epsilon].append(diff_norms.detach().cpu())
                    all_success_by_eps[epsilon].append(success.detach().cpu())

                inputs = []
                labels = []

        if len(inputs) > 0:
            inputs = torch.cat(inputs, dim=0).to(model_device)
            labels = torch.Tensor(labels).long().to(model_device)

            correct_preds = self._get_correct_preds(model, inputs, labels)
            all_initial_correct.append(correct_preds.cpu())

            # remove normalization
            inputs = self._remove_normalization(inputs)

            for epsilon in self.epsilons:
                adversary = adversaries[epsilon]
                with torch.no_grad():
                    x_adv, y_adv = adversary.run_standard_evaluation(inputs[correct_preds], labels[correct_preds], bs=self.batch_size, return_labels=True)
                    diff_norms = ((x_adv - inputs[correct_preds]) ** 2).reshape(x_adv.shape[0], -1).sum(-1).sqrt()
                    success = (~labels[correct_preds].eq(y_adv)).float()

                all_norms_by_eps[epsilon].append(diff_norms.detach().cpu())
                all_success_by_eps[epsilon].append(success.detach().cpu())

        # Tally end results
        for key, values in list(all_success_by_eps.items()):
            all_success_by_eps[key] = torch.cat(values, dim=0)
        for key, values in list(all_norms_by_eps.items()):
            all_norms_by_eps[key] = torch.cat(values, dim=0)

        all_initial_correct = torch.cat(all_initial_correct, dim=0)

        # if self.logger is not None:
            # self.logger.info('PGDAttack: [Done] Success rate: {:.2f}. Avg epsilon: {:.4f}'.format(all_correct_by_eps.sum()/float(all_success.shape[0]), all_epsilons[all_success].mean().item()))

        if self.logger is not None:
            self.logger.info('AutoAttack: [Done] Tallied results:')

        acc = all_initial_correct.float().mean()
        results = {}
        for eps, rob_acc in all_success_by_eps.items():
            success_rate = 1.0-rob_acc.float().mean()
            robust_acc = acc * (1.0-rob_acc.float().mean())
            results['success_rate_{:.3f}'.format(eps)] = success_rate
            results['robust_acc_{:.3f}'.format(eps)] = robust_acc*100.0
            if self.logger is not None:
                self.logger.info(' {} = {:.2f}% ({:.4f})'.format(eps, robust_acc*100.0, success_rate))

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(results, os.path.join(self.save_path, 'epoch_{}.pth'.format(epoch)))

        return results


class AutoAttackDomainAccuracy(InputSpaceEvaluationBase):
    def __init__(self, n_samples, batch_size, 
        epsilons, domains, norm_p=2, dataset='val', 
        attacks_to_run=None, seed=90622, 
        eval_mode=False, save_path=None, **kwargs):
        super(AutoAttackDomainAccuracy, self).__init__(**kwargs)
        self.save_path = None #save_path
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.logger = None
        self.epsilons = epsilons
        self.norm_p = norm_p
        self.dataset_key = dict(val='val_dataset', train='train_dataset')[dataset]
        self.attacks_to_run = attacks_to_run
        self.domains = dict()
        self.seed = seed
        self.eval_mode = eval_mode
        for key, values in domains.items():
            self.domains[key] = torch.Tensor(values)
        self.save_path = save_path

    def normalizing_forward_pass(self, model, x):
        x = self._normalize(x)
        preds, _ = self._forward_pass(model, x)
        return preds

    def _get_correct_preds(self, model, inputs, labels):
        # initial forward pass to acquire correct predicted samples
        with torch.no_grad():
            output, _ = self._forward_pass(model, inputs)
            _, predictions = output.max(dim=1)
            correct_preds = predictions.eq(labels)

        return correct_preds, output

    def _sample_loader(self, dataset):
        torch.manual_seed(self.seed)
        im_inds = torch.randperm(len(dataset))
        
        if self.eval_mode:
            # e.g. from 1000 to 10.000
            gen = range(len(dataset)-self.n_samples, self.n_samples)
            # gen = range(len(dataset)-self.n_samples, len(dataset))
        else:
            # e.g. from 0 to 1000
            gen = range(self.n_samples)

        for i in gen:
            tuple_ = dataset[im_inds[i]]
            image, label = tuple_[0], tuple_[1]
            yield image, label

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs[self.dataset_key]
        loader = kwargs['val_loader']
        epoch = kwargs['epoch']

        norm_p = {1: 'L1', 2: 'L2', np.inf: 'Linf'}[self.norm_p]

        adversaries = {}
        for epsilon in self.epsilons:
            if self.attacks_to_run is not None:
                adversaries[epsilon] = autoattack.AutoAttack(
                    partial(self.normalizing_forward_pass, model), 
                    norm=norm_p, eps=epsilon, version='custom', attacks_to_run=self.attacks_to_run, verbose=True)
            else:
                adversaries[epsilon] = autoattack.AutoAttack(
                    partial(self.normalizing_forward_pass, model), 
                    norm=norm_p, eps=epsilon, version='standard', verbose=True)

        all_norms_by_eps = {eps: [] for eps in self.epsilons}
        all_success_by_eps = {eps: [] for eps in self.epsilons}
        all_initial_correct = []


        all_initial_correct = []
        all_logits = []
        all_targets = []
        inputs = []
        labels = []
        for image, label in self._sample_loader(dataset):
            inputs.append(image.unsqueeze(0))
            labels.append(label)

            if len(inputs) == self.batch_size:
                inputs = torch.cat(inputs, dim=0).to(model_device)
                labels = torch.Tensor(labels).long().to(model_device)

                correct_preds, logits = self._get_correct_preds(model, inputs, labels)

                all_initial_correct.append(correct_preds.cpu())
                all_logits.append(logits.cpu())
                all_targets.append(labels.cpu())

                # remove normalization
                inputs = self._remove_normalization(inputs)

                for epsilon in self.epsilons:
                    adversary = adversaries[epsilon]
                    with torch.no_grad():
                        x_adv, y_adv = adversary.run_standard_evaluation(inputs[correct_preds], labels[correct_preds], bs=min(1000,self.batch_size), return_labels=True)
                        diff_norms = ((x_adv - inputs[correct_preds]) ** 2).reshape(x_adv.shape[0], -1).sum(-1).sqrt()
                        success = (~labels[correct_preds].eq(y_adv)).float()

                    all_norms_by_eps[epsilon].append(diff_norms.detach().cpu())
                    all_success_by_eps[epsilon].append(success.detach().cpu())

                inputs = []
                labels = []

        if len(inputs) > 0:
            inputs = torch.cat(inputs, dim=0).to(model_device)
            labels = torch.Tensor(labels).long().to(model_device)

            correct_preds, logits = self._get_correct_preds(model, inputs, labels)
            all_initial_correct.append(correct_preds.cpu())
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

            # remove normalization
            inputs = self._remove_normalization(inputs)

            for epsilon in self.epsilons:
                adversary = adversaries[epsilon]
                with torch.no_grad():
                    x_adv, y_adv = adversary.run_standard_evaluation(inputs[correct_preds], labels[correct_preds], bs=self.batch_size, return_labels=True)
                    diff_norms = ((x_adv - inputs[correct_preds]) ** 2).reshape(x_adv.shape[0], -1).sum(-1).sqrt()
                    success = (~labels[correct_preds].eq(y_adv)).float()

                all_norms_by_eps[epsilon].append(diff_norms.detach().cpu())
                all_success_by_eps[epsilon].append(success.detach().cpu())

        # Tally end results
        for key, values in list(all_success_by_eps.items()):
            all_success_by_eps[key] = torch.cat(values, dim=0)
        for key, values in list(all_norms_by_eps.items()):
            all_norms_by_eps[key] = torch.cat(values, dim=0)

        all_initial_correct = torch.cat(all_initial_correct, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_logits = torch.cat(all_logits, dim=0)

        if self.logger is not None:
            self.logger.info('AutoAttack: [Done] Tallied results:')

        results = {}
        for eps, rob_acc in all_success_by_eps.items():
            for name, domain in self.domains.items():
                if len(domain) == 0:
                    continue
                else:
                    mask = util.domain_mask(domain, all_targets[all_initial_correct])
                    all_mask = util.domain_mask(domain, all_targets)
                    success_rate = 1.0-rob_acc[mask].float().mean()
                    all_logits_ = all_logits
                    all_logits_ = all_logits[all_mask] #.clone()
                    # all_logits_[:, domain] += 100000.  #hack to weight domain classes higher
                    domain_acc = (all_logits_.max(dim=1)[1]).eq(all_targets[all_mask])
                    robust_acc = domain_acc.float().mean() * (1.0-rob_acc[mask].float().mean())
                    # results['success_rate_{}_{:.3f}'.format(name, eps)] = success_rate
                    results['robust_acc_{}_{:.3f}'.format(name, eps)] = robust_acc*100.0
                    if self.logger is not None:
                        self.logger.info(' {} - {} = {:.2f}% ({:.4f})'.format(eps, name, robust_acc*100.0, success_rate))

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(results, os.path.join(self.save_path, 'epoch_{}.pth'.format(epoch)))

        return results


class DomainNoiseAccuracy(InputSpaceEvaluationBase):
    def __init__(self, domains, sigmas, dataset='val', **kwargs):
        super(DomainNoiseAccuracy, self).__init__(**kwargs)
        self.domains = domains
        self.sigmas = sigmas
        self.dataset_key = dict(val='val_loader', train='train_loader')[dataset]

    def normalizing_forward_pass(self, model, x):
        x = self._normalize(x)
        preds, _ = self._forward_pass(model, x)
        return preds

    def perturb(self, x, sigma):
        noise = torch.randn_like(x) * sigma
        img = torch.clamp(x + noise, 0, 1)
        return img

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        loader = kwargs[self.dataset_key]
        epoch = kwargs['epoch']

        all_correct = dict()
        all_targets = dict()

        with torch.no_grad():
            for sigma in self.sigmas:
                all_correct[sigma] = []
                all_targets[sigma] = []
                for batch, targets in loader:
                    batch = self.perturb(batch, sigma)
                    logits, _ = self._forward_pass(model, batch.cuda())
                    predictions = logits.max(dim=1)[1]
                    correct = predictions.eq(targets.cuda())

                    all_correct[sigma].append(correct.cpu())
                    all_targets[sigma].append(targets.cpu())

                all_correct[sigma] = torch.cat(all_correct[sigma], dim=0)
                all_targets[sigma] = torch.cat(all_targets[sigma], dim=0)


        results = {}
        accuracies = {}
        for name, domain in self.domains.items():
            accuracies[name] = []

        for sigma in self.sigmas:
            for name, domain in self.domains.items():
                mask = util.domain_mask(domain, all_targets[sigma])
                domain_acc = all_correct[sigma][mask].float().mean()
                
                results['acc_{}_{:.3f}'.format(name, sigma)] = domain_acc*100.0
                accuracies[name].append(domain_acc.item())

                if self.logger is not None:
                    self.logger.info(' {} - {} = {:.2f}'.format(sigma, name, domain_acc*100.0))

        for name, domain in self.domains.items():
            avg_acc = torch.mean(torch.Tensor(accuracies[name]))
            results['acc_{}_avg'.format(name)] = avg_acc*100.0

            if self.logger is not None:
                self.logger.info(' Averaged - {} = {:.2f}'.format(name, avg_acc*100.0))

        return results
            



class StepLRScheduler(object):
    def __init__(self, base_lr, gamma, step_size):
        self.base_lr = base_lr
        self.gamma = gamma
        self.step_size = step_size
        self.lr = self.base_lr
        self.it = 0

    def step(self, iteration=None):
        if iteration is not None:
            self.it = iteration
        else:
            self.it += 1

        power = self.it // self.step_size

        self.lr = self.base_lr * (self.gamma**power)

    def get_last_lr(self):
        return self.lr


class RepresentationInversion(PGDAttackEvaluationBase):
    def __init__(self, match_layer, image_selection, 
        num_samples, sample_from, deltas,
        save_path, norm_p, lr, max_iter,
        lr_scheduler=None, **kwargs):
        super(RepresentationInversion, self).__init__(**kwargs)
        self.match_layer = match_layer
        self.image_selection = image_selection
        self.save_path = save_path
        self.logger = None
        self.max_iter = max_iter

        # self.loss = self._construct_loss(loss_type)
        self.norm_p = norm_p
        self.lr = lr
        self.deltas = deltas
        self.lr_scheduler_cfg = lr_scheduler

        self.num_samples = num_samples

        if type(sample_from) is str:
            self.sample_from = sample_from
            assert sample_from in ['gaussian_noise']
        elif isinstance(sample_from, dict):
            self.sample_from = 'dataset'
            self.sample_from_cfg = sample_from
            self.sample_from_cfg['image_selection'] = {-1: sample_from.get('image_selection', None)}
            self.sample_from_cfg['idx'] = 0
        else:
            raise AttributeError('Unknown sample_from value {}'.format(sample_from))
        self.samples = None

    def _construct_lrscheduler(self, cfg):
        sched_type = cfg.pop('type')
        scheduler = StepLRScheduler(base_lr=self.lr, **cfg)
        cfg['type'] = sched_type
        return scheduler
        # if sched_type.startswith('optim.'):
        #     sched_class = torch.optim.lr_scheduler.__dict__[sched_type.replace('optim.','')]
        # else:
        #     sched_class = lrschedulers.__dict__[sched_type]
        # return sched_class(DummyOptimizer(self.lr), **cfg)

    def _gaussian_sampler(self, img_shape, **kwargs):
        return (torch.randn(*img_shape) / 20.0 + 0.5).clamp_(0,1)

    def _dataset_sampler(self, label, **kwargs):
        # assertions
        # print(self.sample_from_cfg['idx'], self.sample_from_cfg.get('last_label', -1), label)
        if self.sample_from_cfg['idx'] > 0:
            assert self.sample_from_cfg['last_label'] == label, 'idx counter error. Counting for label {}, but got label {}'.format(self.sample_from_cfg['last_label'], label)

        dataset = self.sample_from_cfg['dataset']
        if label not in self.sample_from_cfg['image_selection']:
            if self.sample_from_cfg['image_selection'][-1] is not None:
                self.sample_from_cfg['image_selection'][label] = self.sample_from_cfg['image_selection'][-1]
            else:
                set_indices = torch.randperm(len(dataset))
                indices = []
                for sample_idx in set_indices:
                    _, label_at_idx = dataset[sample_idx]
                    if label_at_idx != label:
                        indices.append(sample_idx.item())
                    if len(indices) == self.num_samples:
                        break
                print(label, indices)
                self.sample_from_cfg['image_selection'][label] = indices

        indices = self.sample_from_cfg['image_selection'][label]
        idx = self.sample_from_cfg['idx']
        sample, _ = dataset[indices[idx]]
        self.sample_from_cfg['idx'] = (self.sample_from_cfg['idx']+1) % len(indices)
        self.sample_from_cfg['last_label'] = label
        return sample

    def _sample_loader(self, dataset, sampler):
        for img_idx in self.image_selection:
            image, label = dataset[img_idx]
            yield image, label

            if self.samples is None:
                self.samples = OrderedDict()
            if label not in self.samples:
                self.samples[label] = [sampler(img_shape=image.shape, label=label) for _ in range(self.num_samples)]

            M = len(self.deltas)
            # We generate the same samples M times
            for delta_idx in range(M):
                for sample in self.samples[label]:
                    yield sample, label
            # for sample_idx in range(self.num_samples):
            #     yield sampler(img_shape=image.shape), label

    def _get_correct_preds(self, model, inputs, labels):
        # initial forward pass to acquire correct predicted samples
        with torch.no_grad():
            output, _ = self._forward_pass(model, inputs)
            _, predictions = output.max(dim=1)
            correct_preds = predictions.eq(labels)

        return correct_preds

    def _invert(self, model, inputs, labels, sample_idx):
        original_inputs = inputs.detach().clone()
        lrs = torch.full((inputs.shape[0],), 1.0).to(inputs.device)

        rmNorm = self._remove_normalization

        lr_scheduler = None
        if self.lr_scheduler_cfg is not None:
            lr_scheduler = self._construct_lrscheduler(self.lr_scheduler_cfg)

        # inputs = inputs.clone().detach().requires_grad_(True)
        # optim = self._init_optimizer([inputs], self.optimizer_cfg)

        for it in range(self.max_iter):
            inputs = inputs.clone().detach().requires_grad_(True)

            outputs = model(inputs)
            ref_rep = outputs['layer_output'][self.match_layer]

            rep_shape = np.arange(len(ref_rep.shape)).tolist()

            rep_diff = torch.norm(ref_rep[1:]-ref_rep[0].detach().unsqueeze(0), p=self.norm_p, dim=rep_shape[1:])
            loss = torch.div(rep_diff, torch.norm(ref_rep[0], p=self.norm_p).detach())
            # loss = (loss**2).mean()
            # loss = rep_diff
            loss = loss.mean()

            # optim.zero_grad()
            loss.backward()

            lr = self.lr
            if lr_scheduler is not None:
                lr = lr_scheduler.get_last_lr()

            with torch.no_grad():
                for delta_idx, delta in enumerate(self.deltas):
                    s = 1+delta_idx * self.num_samples
                    e = 1+(1+delta_idx) * self.num_samples
                    pgd_updates = self._step_and_project(delta, original_inputs, inputs, inputs.grad, lrs*lr)
                    inputs[s:e] = pgd_updates[s:e]

            if lr_scheduler is not None:
                lr_scheduler.step()

            # optim.step()

            # with torch.no_grad():
            #     inputs[:,0].clamp_(-mean[0]/std[0],(1-mean[0])/std[0])
            #     inputs[:,1].clamp_(-mean[1]/std[1],(1-mean[1])/std[1])
            #     inputs[:,2].clamp_(-mean[2]/std[2],(1-mean[2])/std[2])

            if (it+sample_idx*self.max_iter) % ((self.max_iter*len(self.image_selection))//20) == 0:
                if self.logger is not None:
                    pred_values, predictions = outputs['prediction'].max(dim=1)
                    pred_agreement = (predictions[1:] == predictions[0]).sum().float() / (predictions.shape[0] - 1)
                    
                    self.logger.info('RepresentationInversion: [{}/{}: {}/{}] lr {:.4f}, Loss {:.4f}, R-diff: {:.4f}, Im-diff (Delta): {:.4f}, Prediction agreement: {:.4f}'.format(sample_idx, len(self.image_selection), it, self.max_iter, 
                        lr,
                        loss.detach().item(), 
                        rep_diff.mean().item(),
                        torch.norm(rmNorm(original_inputs[1:])-rmNorm(inputs[1:]), p=2, dim=[1,2,3]).mean().item(),
                        pred_agreement.item()))

        pred_values, predictions = outputs['prediction'].max(dim=1)
        

        #######################
        # Final calculations and print out
        avg_rep_diff_per_delta = {}
        min_rep_diff_per_delta = {}
        pred_agreement_per_delta = {}

        prefix = 'RepresentationInversion:'
        if self.logger is not None:
            self.logger.info('{} [{}/{}: {}/{}]'.format(prefix, sample_idx, len(self.image_selection), it+1, self.max_iter))
        for delta_idx, delta in enumerate(self.deltas):
            s = 1+delta_idx * self.num_samples
            e = 1+(1+delta_idx) * self.num_samples
            pred_agreement = (predictions[s:e] == predictions[0]).sum().float() / (e-s)
            r_diff = rep_diff[s:e].mean().item()
            min_r_diff = rep_diff[s:e].min().item()
            avg_rep_diff_per_delta[delta] = r_diff/torch.norm(ref_rep[0], p=self.norm_p)
            min_rep_diff_per_delta[delta] = min_r_diff/torch.norm(ref_rep[0], p=self.norm_p)
            pred_agreement_per_delta[delta] = pred_agreement

            if self.logger is not None:
                self.logger.info('{} delta={} :: R-diff (raw/std): ({:.4f}, {:.4f}) min: {:.4f}, Im-diff (Delta): {:.4f}, Pred agreement: {:.2f}'.format(
                    ''.ljust(len(prefix)),
                    delta,
                    r_diff,
                    avg_rep_diff_per_delta[delta],
                    min_rep_diff_per_delta[delta],
                    torch.norm(rmNorm(original_inputs[s:e])-rmNorm(inputs[s:e]), p=2, dim=[1,2,3]).mean().item(),
                    pred_agreement.item()))
            
        # pred_values, predictions = outputs['prediction'].max(dim=1)
        # pred_agreement = (predictions[1:] == predictions[0]).sum().float() / (predictions.shape[0] - 1)

        return inputs.detach(), pred_agreement_per_delta, rep_diff, avg_rep_diff_per_delta, min_rep_diff_per_delta

    def save_images(self, images, epoch):
        # prepare file system
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # remove image normalization
        images = self._remove_normalization(images)

        batch_size = self.num_samples * len(self.deltas) + 1

        sample_keys = list(self.samples.keys())

        for delta_idx, delta in enumerate(self.deltas):
            image_list = []
            for input_idx in range(len(self.image_selection)):
                s = input_idx * batch_size
                e = (1+input_idx) * batch_size
                image_for_input = images[s:e]

                s = 1 + delta_idx * self.num_samples
                e = 1 + (1+delta_idx) * self.num_samples

                image_for_input_and_delta = image_for_input[s:e]
                if self.sample_from == 'dataset' and self.sample_from_cfg['image_selection'][-1] is None:
                    # source images are differently sampled per label
                    sample_key = sample_keys[input_idx]
                    samples = [torch.zeros_like(self.samples[sample_key][0]).unsqueeze(0)]+[s.unsqueeze(0) for s in self.samples[sample_key]]
                    image_list += samples

                image_list.append(images[input_idx * batch_size].unsqueeze(0))  # reference image
                image_list.append(image_for_input_and_delta)  # images for reference and delta

            if self.sample_from != 'dataset' or self.sample_from_cfg['image_selection'][-1] is not None:
                sample_row = torch.stack([torch.zeros_like(self.samples[0][0])]+self.samples[0])
                images_ = torch.cat([sample_row]+image_list)
            else:
                images_ = torch.cat(image_list)

            grid = make_grid(images_, pad_value=0.5, normalize=False, nrow=self.num_samples+1)
            save_image(grid, fp=os.path.join(self.save_path, 'epoch_{}_delta{}.png'.format(epoch, delta)))


    def save_data(self, avg_rep_diffs_per_delta, min_rep_diffs_per_delta, pred_agreements_per_delta, epoch):
        # prepare file system
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        filep = os.path.join(self.save_path, 'epoch_{}_data.csv'.format(epoch))

        with open(filep, 'w') as f:
            f.write('Difference in representation space\n')
            f.write('{}\n'.format(','.join([str(d) for d in self.deltas])))
            f.write('{}\n'.format(','.join([str(np.mean(r_diffs)) for r_diffs in min_rep_diffs_per_delta.values()])))
            f.write('{}\n'.format(','.join([str(np.std(r_diffs)) for r_diffs in min_rep_diffs_per_delta.values()])))

            f.write('\n')
            f.write('Prediction agreement\n')
            f.write('{}\n'.format(','.join([str(d) for d in self.deltas])))
            f.write('{}\n'.format(','.join([str(np.mean(agreement)) for agreement in pred_agreements_per_delta.values()])))
            f.write('{}\n'.format(','.join([str(np.std(agreement)) for agreement in pred_agreements_per_delta.values()])))

        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(2,1,1)
        xs = self.deltas
        ys = [np.mean(r_diffs) for r_diffs in min_rep_diffs_per_delta.values()]
        err_ys = [np.std(r_diffs) for r_diffs in min_rep_diffs_per_delta.values()]
        plt.errorbar(xs, ys, err_ys, fmt='o-')
        plt.xlabel('$||\\delta||_2$')
        plt.ylabel('Min dist in rep space')
        plt.grid('on')

        plt.subplot(2,1,2)
        ys = [np.mean(agreement) for agreement in pred_agreements_per_delta.values()]
        err_ys = [np.std(agreement) for agreement in pred_agreements_per_delta.values()]
        plt.errorbar(xs, ys, err_ys, fmt='o-')
        plt.xlabel('$||\\delta||_2$')
        plt.ylabel('Class agreement with $x_2$')
        plt.grid('on')

        plt.suptitle(os.path.basename(os.path.dirname(self.save_path)))

        plt.savefig(os.path.join(self.save_path, 'epoch_{}_plots.png'.format(epoch)))

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs['val_dataset']
        epoch = kwargs['epoch']

        all_images = []
        all_labels = []
        all_epsilons = []
        all_rep_diffs = []
        pred_agreements_per_delta = {delta: [] for delta in self.deltas}
        avg_rep_diffs_per_delta = {delta: [] for delta in self.deltas}
        min_rep_diffs_per_delta = {delta: [] for delta in self.deltas}

        batch_size = self.num_samples * len(self.deltas) + 1  # N random samples per delta + original image

        if self.sample_from == 'gaussian_noise':
            sampler = self._gaussian_sampler
        elif self.sample_from == 'dataset':
            self.sample_from_cfg['dataset'] = kwargs[self.sample_from_cfg['type']]
            sampler = self._dataset_sampler

        inputs = []
        labels = []
        for i, (image, label) in enumerate(self._sample_loader(dataset, sampler)):
            inputs.append(image.unsqueeze(0))
            labels.append(label)

            if len(inputs) == batch_size:
                inputs = torch.cat(inputs, dim=0).to(model_device)
                labels = torch.Tensor(labels).long().to(model_device)

                inversions, pred_agreement_per_delta, representation_diffs, avg_rdiff_per_delta, min_rdiff_per_delta = self._invert(model, inputs, labels, (i+1)//batch_size)

                epsilons = torch.norm(self._remove_normalization(inversions[1:])-self._remove_normalization(inversions[0].unsqueeze(0)), p=self.norm_p, dim=[1,2,3])

                all_images.append(inversions.detach().cpu())
                all_labels.append(labels.detach().cpu())
                all_epsilons.append(epsilons.detach().cpu())
                all_rep_diffs.append(representation_diffs.detach().cpu())

                for delta, r_diff in avg_rdiff_per_delta.items():
                    avg_rep_diffs_per_delta[delta].append(r_diff.detach().cpu().item())
                for delta, r_diff in min_rdiff_per_delta.items():
                    min_rep_diffs_per_delta[delta].append(r_diff.detach().cpu().item())
                for delta, agreement in pred_agreement_per_delta.items():
                    pred_agreements_per_delta[delta].append(agreement.detach().cpu().item())

                if self.logger is not None:
                    prefix = 'RepresentationInversion:'
                    self.logger.info('\n{} [{}/{}] Cumulative stats:'.format(prefix, i+1, len(self.image_selection)))
                    for delta in self.deltas:
                        self.logger.info('{} delta={} :: avg R_diff (std): {:.4f}, min R_diff (std): {:.4f}'.format(prefix, delta, np.mean(avg_rep_diffs_per_delta[delta]), np.mean(min_rep_diffs_per_delta[delta])))
                    self.logger.info('\n')


                inputs = []
                labels = []

        all_images = torch.cat(all_images, dim=0)
        self.save_images(all_images, epoch)

        self.save_data(avg_rep_diffs_per_delta, min_rep_diffs_per_delta, pred_agreements_per_delta, epoch)

        if self.logger is not None:
            self.logger.info('RepresentationInversion: [Done]')

        all_epsilons = torch.cat(all_epsilons, dim=0)
        all_rep_diffs = torch.cat(all_rep_diffs)

        result = dict(img_diff=all_epsilons.mean())
        for delta, r_diffs in avg_rep_diffs_per_delta.items():
            result['rdiff_avg_{:.4f}'.format(delta)] = np.mean(r_diffs)
        for delta, r_diffs in min_rep_diffs_per_delta.items():
            result['rdiff_min_{:.4f}'.format(delta)] = np.mean(r_diffs)

        return result


class PGDAttack(PGDAttackEvaluationBase):
    def __init__(self, target_map, num_classes, save_path, lr, max_iter,
        image_selection, batch_size, loss_type,
        eps_robust, projection_method='image_space_bounded',
        eps_thresh=None, maximize_indefinitely=False, norm_p=2, **kwargs):
        super(PGDAttack, self).__init__(**kwargs)
        self.target_map = target_map
        self.num_classes = num_classes
        self.save_path = save_path
        self.lr = lr
        self.max_iter = max_iter
        self.eps_robust = eps_robust
        self.eps_thresh = eps_thresh
        if self.eps_thresh is None:
            self.eps_thresh = 0 #np.infty
        self.image_selection = image_selection
        self.batch_size = batch_size
        self.logger = None
        self.maximize_indefinitely = maximize_indefinitely
        self.projection_method = projection_method
        self.norm_p = norm_p

        self.loss = self._construct_loss(loss_type)

    def _sample_loader(self, dataset):
        if self.target_map == 'pairwise':
            for img_idx in self.image_selection:
                train_data = dataset[img_idx]
                image, label = train_data[0], train_data[1]

                #for target_idx in self.target_map[label]:
                for target in range(self.num_classes):
                    yield image, label, target
        else:
            raise NotImplementedError
            # for img_idx in self.image_selection:
            #     image, label = dataset[img_idx]
            #     #for target_idx in self.target_map[label]:

    def _attack(self, model, inputs, targets):
        lrs = torch.full((inputs.shape[0],), self.lr).to(inputs.device)
        success = (lrs == 0)

        original_inputs = inputs.detach().clone()

        original_predictions = None

        mean = self.data_mean
        std = self.data_std

        iteration = 0
        while (success.sum() < inputs.shape[0] and iteration < self.max_iter) or \
            (self.maximize_indefinitely and iteration < self.max_iter):

            inputs = inputs.clone().detach().requires_grad_(True)

            output, robust = self._forward_pass(model, inputs)

            pred_values, predictions = output.max(dim=1)

            if original_predictions is None:
                original_predictions = predictions.detach()

            with torch.no_grad():
                if np.isinf(self.norm_p):
                    # epsilons = (self._remove_normalization(original_inputs) - \
                    #     self._remove_normalization(inputs)).abs().max()
                    epsilons = (self._remove_normalization(original_inputs) - \
                        self._remove_normalization(inputs)).view(inputs.shape[0], -1).abs().max(1)[0]
                else:
                    epsilons = torch.norm(self._remove_normalization(original_inputs) - \
                        self._remove_normalization(inputs), p=self.norm_p, dim=[1,2,3])

                # success = epsilons >= self.eps_thresh
                success = predictions.eq(targets)
                # success = predictions.eq(targets)
                if robust is not None:
                    # flag only successful when also robust
                    success = success & (pred_values > robust)
                
                if (self.eps_thresh > 0) and (not self.maximize_indefinitely):
                    # stop updating for successful attacks
                    print('setting lr to 0')
                    lrs[success] = 0

            # update
            loss = self.loss(output, targets)
            grad = torch.autograd.grad(loss, [inputs], grad_outputs=torch.ones_like(loss))[0]

            # loss = self.loss(output, targets)
            # # loss = -torch.gather(output, dim=1, index=targets.unsqueeze(1)).mean()
            # loss.backward(retain_graph=False)

            if self.projection_method == 'image_space_bounded':
                with torch.no_grad():
                    inputs -= lrs[:, None, None, None]*inputs.grad
                    inputs[:,0].clamp_(-mean[0]/std[0],(1-mean[0])/std[0])
                    inputs[:,1].clamp_(-mean[1]/std[1],(1-mean[1])/std[1])
                    inputs[:,2].clamp_(-mean[2]/std[2],(1-mean[2])/std[2])
                    inputs.grad.fill_(0)
            elif self.projection_method == 'eps_thresh_bounded':
                with torch.no_grad():
                    inputs = self._step_and_project(self.eps_thresh, original_inputs, inputs, grad, lrs)
                # inputs = self._step_and_project(self.eps_thresh, original_inputs, inputs, inputs.grad, lrs)
            elif self.projection_method == 'instance_eps_thresh_bounded':
                with torch.no_grad():
                    inputs = self._step_and_project(self.eps_thresh, inputs.clone().detach(), inputs, inputs.grad, lrs)


            # success = (lrs == 0)
            # success = predictions.eq(targets)

            iteration += 1

            if iteration % (self.max_iter//10) == 0:
                if self.logger is not None:
                    tgt_values = torch.gather(output, index=targets.unsqueeze(1), dim=1)
                    if self.eps_thresh == 0:
                        self.logger.info('PGDAttack: [{}/{}] Success rate: {:.2f}. Avg tgt confidence: {:.4f}, Avg epsilon: {:.4f}'.format(iteration, self.max_iter, success.sum()/float(inputs.shape[0]), tgt_values.mean().item(), epsilons[success].mean().item()))
                    else:
                        self.logger.info('PGDAttack: [{}/{}] Success rate: {:.2f}. Avg tgt confidence: {:.4f}, Avg epsilon: {:.4f}'.format(iteration, self.max_iter, success.sum()/float(inputs.shape[0]), tgt_values.mean().item(), epsilons.mean().item()))

        if robust is not None:
            robust_preds = pred_values > robust
        else:
            robust_preds = None

        if np.isinf(self.norm_p):
            epsilons = (self._remove_normalization(original_inputs) - \
                self._remove_normalization(inputs)).view(inputs.shape[0], -1).abs().max(1)[0]
        else:
            epsilons = torch.norm(self._remove_normalization(original_inputs) - \
                self._remove_normalization(inputs), p=self.norm_p, dim=[1,2,3])
        
        return inputs, success, epsilons, original_predictions, robust_preds

    def save_results(self, inputs, targets, original_preds, robust_preds, success, epsilons, epoch):
        # prepare file system
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # remove image normalization
        # std = torch.Tensor(self.data_std).to(inputs.device)[None, :, None, None]
        # mean = torch.Tensor(self.data_mean).to(inputs.device)[None, :, None, None]
        # inputs = inputs * std + mean
        inputs = self._remove_normalization(inputs)

        if self.target_map == 'pairwise':
            grid = make_grid(inputs, pad_value=0.5, normalize=False, nrow=self.num_classes)
            save_image(grid, fp=os.path.join(self.save_path, 'epoch_{}.png'.format(epoch)))

            is_robust_str = lambda vals, idx: '1' if robust_preds[idx] else '0'

            with open(os.path.join(self.save_path, 'epoch_{}.csv'.format(epoch)), mode='w') as result_file:
                writer = csv.writer(result_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([str(label) for label in range(self.num_classes)])

                batch_idx = 0
                for i in range(self.num_classes):
                    row = []
                    for target in range(self.num_classes):
                        row.append('P={}, s={}, rob={}, eps={:.3f}'.format(
                            original_preds[batch_idx].item(),
                            '1' if success[batch_idx] else '0',
                            is_robust_str(robust_preds, batch_idx) if robust_preds is not None else 'n/a',
                            epsilons[batch_idx].item()
                            ))
                        batch_idx += 1

                    writer.writerow(row)

        else:
            raise NotImplementedError

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs['val_dataset']
        epoch = kwargs['epoch']

        all_inputs = []
        all_targets = []
        all_labels = []
        all_success = []
        all_epsilons = []
        all_original_preds = []
        all_robust_preds = []

        inputs = []
        targets = []
        for image, label, target in self._sample_loader(dataset):
            inputs.append(image.unsqueeze(0))
            targets.append(target)
            all_labels.append(label)

            if len(inputs) == self.batch_size:
                inputs = torch.cat(inputs, dim=0).to(model_device)
                targets = torch.Tensor(targets).long().to(model_device)

                inputs, success, epsilons, original_preds, robust_preds = self._attack(model, inputs, targets)

                all_inputs.append(inputs.detach().cpu())
                all_targets.append(targets.detach().cpu())
                all_success.append(success.detach().cpu())
                all_epsilons.append(epsilons.detach().cpu())
                all_original_preds.append(original_preds.detach().cpu())
                if robust_preds is not None:
                    all_robust_preds.append(robust_preds.detach().cpu())

                inputs = []
                targets = []

        if len(inputs) > 0:
            success, epsilons, original_preds, robust_preds = self._attack(model, inputs, targets)

            all_inputs.append(inputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_success.append(success.detach().cpu())
            all_epsilons.append(epsilons.detach().cpu())
            all_original_preds.append(original_preds.detach().cpu())
            if robust_preds is not None:
                all_robust_preds.append(robust_preds.detach().cpu())

        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_labels = torch.Tensor(all_labels).to(all_targets.device)
        all_success = torch.cat(all_success, dim=0)
        all_epsilons = torch.cat(all_epsilons, dim=0)
        all_original_preds = torch.cat(all_original_preds, dim=0)
        if len(all_robust_preds) > 0:
            all_robust_preds = torch.cat(all_robust_preds, dim=0)
        else:
            all_robust_preds = None

        if self.logger is not None:
            self.logger.info('PGDAttack: [Done] Success rate: {:.2f}. Avg epsilon: {:.4f}'.format(all_success.sum()/float(all_inputs.shape[0]), all_epsilons[all_success].mean().item()))

        self.save_results(all_inputs, all_targets, all_original_preds, all_robust_preds, all_success, all_epsilons, epoch)

        correct = all_original_preds.eq(all_labels)
        eps_robust = all_epsilons <= self.eps_robust
        correct_and_robust = correct & eps_robust
        robust_acc = correct_and_robust.sum() / float(all_labels.shape[0])

        if self.eps_thresh == 0:
            return dict(avg_epsilon=all_epsilons[all_success].mean(), robust_accuracy=robust_acc)
        else:
            return dict(avg_epsilon=all_epsilons.mean(), robust_accuracy=robust_acc)


class ValidateWithBackwardPass(EvaluationBase):
    def __init__(self, loss_type, *args, dataset='val', **kwargs):
        super(ValidateWithBackwardPass, self).__init__(*args, **kwargs)
        self.loss = self._construct_loss(loss_type)
        self.dataset_key = dict(val='val_loader', train='train_loader')[dataset]

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        logger = kwargs.get('logger', None)
        loader = kwargs[self.dataset_key]
        epoch = kwargs['epoch']

        self.loss = self.loss.cuda()

        optimizer = self._init_optimizer(kwargs['parent'].parameters(), dict(type='SGD', lr=0.1))

        for i, data in enumerate(loader):
            if len(data) == 2:
                input, target = data
            elif len(data) == 3:
                input, target, data_kwargs = data

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            optimizer.zero_grad()
            prediction, _ = self._forward_pass(model, input)
            # print(type(input), type(prediction), type(target))
            loss = self.loss(prediction, target)
            loss.backward()

            if i % (len(loader)//10) == 0:
                if logger is not None:
                    logger.info('ValidateWithBackwardPass :: Progress [{}/{}]'.format(i, len(loader)))

        return {}


class GradientCoherence(EvaluationBase):
    def __init__(self, num_classes, structure_tensor_sigma, save_path, 
        image_selection, batch_size, loss_type,
        data_mean, data_std, **kwargs):
        super(GradientCoherence, self).__init__(**kwargs)

        from skimage.feature import structure_tensor, structure_tensor_eigenvalues

        self.num_classes = num_classes
        self.save_path = save_path
        self.image_selection = image_selection
        self.batch_size = batch_size
        self.data_mean = data_mean
        self.data_std = data_std
        self.structure_tensor_sigma = structure_tensor_sigma

        self.loss = self._construct_loss(loss_type)

    def _remove_normalization(self, inputs):
        # remove image normalization
        std = torch.Tensor(self.data_std).to(inputs.device)[None, :, None, None]
        mean = torch.Tensor(self.data_mean).to(inputs.device)[None, :, None, None]
        inputs = inputs * std + mean
        return inputs

    def save_gradients(self, epoch, gradients, suffix=''):
        # prepare file system
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # g_ = (g_ - g_.min()) / (g_.max() - g_.min())
        # scipy.misc.toimage(g_[0], cmin=0., cmax=1.).save(path + '.png')

        # normalization individual per image
        g_ = gradients.view(gradients.shape[0], -1)
        # print(gradients.shape, g_.shape, g_.min(-1, keepdim=True)[0].shape)
        g_min = g_.min(-1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        g_max = g_.max(-1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        gradients = (gradients - g_min) / (g_max - g_min)

        grid = make_grid(gradients, pad_value=0.5, normalize=False, scale_each=False, nrow=self.num_classes)
        save_image(grid, fp=os.path.join(self.save_path, 'epoch_{}_grads{}.png'.format(epoch, suffix)))

    def save_inputs(self, epoch, inputs):
        # prepare file system
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # remove image normalization
        inputs = self._remove_normalization(inputs)

        grid = make_grid(inputs, pad_value=0.5, normalize=False, nrow=self.num_classes)
        save_image(grid, fp=os.path.join(self.save_path, 'epoch_{}_inputs.png'.format(epoch)))

    def _attack(self, model, inputs, targets):
        from skimage.feature import structure_tensor, structure_tensor_eigenvalues
        # mean = self.data_mean
        # std = self.data_std

        inputs.requires_grad = True
        output, robust = self._forward_pass(model, inputs)

        # update
        # loss = self.loss(output, targets)
        loss = -torch.gather(output, dim=1, index=targets.unsqueeze(1)).mean()
        loss.backward(retain_graph=False)

        with torch.no_grad():
            grads = inputs.grad

            # calculate gradient magnitude per pixel
            grads_np = torch.norm(grads, dim=1, p=2)
            grads_np = (grads_np/grads_np.max()).cpu().numpy()

            avg_eigs = torch.zeros(grads_np.shape[0], 2).to(inputs.device)
            for img_idx in range(grads_np.shape[0]):
                # calculate structure tensor and eigenvalues
                arr = structure_tensor(grads_np[img_idx], sigma=self.structure_tensor_sigma, order='rc')
                eig = structure_tensor_eigenvalues(arr)

                # calculate average eigenvalues across image
                avg_eigs[img_idx, :] = torch.from_numpy(eig.reshape(2, -1).mean(-1))

            grads /= torch.norm(grads, dim=[1,2,3], p=2, keepdim=True)

        return grads.detach(), avg_eigs

    def _sample_loader(self, dataset):
        for img_idx in self.image_selection:
            image, label = dataset[img_idx]
            #for target_idx in self.target_map[label]:
            for target in range(self.num_classes):
                yield image, label, target

    def forward(self, **kwargs):
        model = kwargs['model']
        model_device = torch.ones(1).cuda().device
        self.logger = kwargs.get('logger', None)
        dataset = kwargs['val_dataset']
        loader = kwargs['val_loader']
        epoch = kwargs['epoch']

        ################################
        # 1st, generated gradients for selected images and save them to file

        if self.logger is not None:
            self.logger.info('GradientCoherence: Calculating image mean on training datasets')
        with torch.no_grad():
            dataset_mean = None
            for image, _ in kwargs['train_dataset']:
                if dataset_mean is None:
                    dataset_mean = image
                else:
                    dataset_mean += image
            dataset_mean /= len(kwargs['train_dataset'])
        if self.logger is not None:
            self.logger.info('GradientCoherence: Done')

        inputs = []
        targets = []
        for image, label, target in self._sample_loader(dataset):
            inputs.append(image.unsqueeze(0))
            targets.append(target)

        inputs = torch.cat(inputs, dim=0).to(model_device)
        targets = torch.Tensor(targets).long().to(model_device)

        grads, _ = self._attack(model, inputs, targets)

        self.save_inputs(epoch, inputs.cpu())
        self.save_gradients(epoch, grads.cpu())

        # generate gradients from image mean
        inputs = dataset_mean.repeat(len(targets), 1, 1, 1).to(model_device)
        grads, _ = self._attack(model, inputs, targets)
        self.save_gradients(epoch, grads.cpu(), '_fromImageMean')

        # generate gradients from average value
        inputs = torch.full(inputs.shape, 0.5).to(model_device)
        grads, _ = self._attack(model, inputs, targets)
        self.save_gradients(epoch, grads.cpu(), '_from0.5')

        # generate gradients from average value
        inputs = torch.full(inputs.shape, 0.0).to(model_device)
        grads, _ = self._attack(model, inputs, targets)
        self.save_gradients(epoch, grads.cpu(), '_from0')

        ################################
        # 2nd, generate structure tensor statistics for validation dataset
        all_targets = []
        avg_eigs = []
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(model_device)
            targets = targets.to(model_device)

            _, eigs = self._attack(model, inputs, targets)

            all_targets.append(targets.cpu())
            avg_eigs.append(eigs.cpu())

            if i % (len(loader)//10) == 0:
                if self.logger is not None:
                    self.logger.info('GradientCoherence: [{}/{}]'.format(i, len(loader)))

        # targets = torch.cat(all_targets)
        avg_eigs = torch.cat(avg_eigs, dim=0)
        coherence = ((avg_eigs[0]-avg_eigs[1]) / (avg_eigs[0]+avg_eigs[1]))**2

        if self.logger is not None:
            self.logger.info('GradientCoherence: Average coherence: {}, Average Eigenvalues: {}, {}'.format(coherence.mean(), avg_eigs[:,0].mean(), avg_eigs[:,1].mean()))
            self.logger.info('GradientCoherence: [Done]')

        return dict(coherence_dist=Histogram(coherence), avg_coherence=coherence.mean(), avg_eig1=avg_eigs[:,0].mean(), avg_eig2=avg_eigs[:,1].mean())


class GenerateMaxEntropyDataset(EvaluationBase):
    def __init__(self, p, eps, step_size, iterations, 
        loss_name, data_mean, data_std, save_path, num_images, *args, **kwargs):
        super(GenerateMaxEntropyDataset, self).__init__(*args, **kwargs)
        self.p = p
        self.eps = eps
        self.step_size = step_size
        self.iterations = iterations
        self.loss_name = loss_name
        self.data_mean = data_mean
        self.data_std = data_std
        # self.dataset_key = dict(val='val_loader', train='train_loader', uniform='uniform')[dataset]
        self.save_path = save_path
        self.num_images = num_images

    def uniform_noise_generator(self, input_shape, batch_size):
        for i in range(0, self.num_images, batch_size):
            batch = torch.rand(*((batch_size,)+input_shape))
            yield batch, torch.zeros(batch_size)


    def forward(self, **kwargs):
        from model import AttackerModel

        model = kwargs['parent']
        model_device = torch.ones(1).cuda().device
        logger = kwargs.get('logger', None)

        # loader = kwargs[self.dataset_key]
        input_shape = kwargs['train_dataset'][0][0].shape
        batch_size = 100 #next(iter(kwargs['train_loader']))[0].shape[0]
        loader = partial(self.uniform_noise_generator, input_shape=input_shape, batch_size=batch_size)
        epoch = kwargs['epoch']

        adv_model = AttackerModel(self.p, self.eps, self.step_size, self.iterations, self.loss_name, data_mean=self.data_mean, data_std=self.data_std)

        for name, child in model.named_children():
            setattr(adv_model, name, child)

        data = []
        labels = []

        i = 0
        for input, target in loader():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.no_grad():
                prediction = adv_model._forward_pass(input)['prediction']
                entropy = -torch.sum(torch.log_softmax(prediction, dim=1)*torch.softmax(prediction, dim=1), dim=1)
                if logger is not None:
                    logger.info('GenerateMaxEntropyDataset: [{}/{}] Avg. log entropy before perturbations: {:.2f}'.format(i, self.num_images//batch_size, entropy.log().mean(0).item()))

            perturbed = adv_model.attack(input, target)
            data.append(perturbed.detach().cpu())
            labels.append(target.cpu())

            with torch.no_grad():
                prediction = adv_model._forward_pass(perturbed)['prediction']
                entropy = -torch.sum(torch.log_softmax(prediction, dim=1)*torch.softmax(prediction, dim=1), dim=1)
                if logger is not None:
                    logger.info('GenerateMaxEntropyDataset: [{}/{}] Avg. log entropy after perturbations: {:.2f}'.format(i, self.num_images//batch_size, entropy.log().mean(0).item()))

            i += 1

        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)

        save_filep = os.path.join(self.save_path, 'data.pth')

        if logger is not None:
            logger.info('GenerateMaxEntropyDataset: Saving {} samples to {}'.format(data.shape[0], save_filep))

        os.makedirs(os.path.dirname(save_filep), exist_ok=True)

        torch.save({
            'images': data,
            'labels': labels},
            save_filep)