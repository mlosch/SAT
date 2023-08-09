import os
import torch
import numpy as np
import torch.nn.functional as F


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class LayerOutputInspector(object):
    def __init__(self):
        self.layers = []
        self.handles = []

    def register_hooks(self, net, layers=None):
        # register forward hooks
        if layers is not None:
            for layer in layers:
                module = net
                for subm in layer.split('.'):
                    module = module._modules.get(subm)
                try:
                    self.handles.append(module.register_forward_hook(self.forward_hook))
                except:
                    raise RuntimeError('Cannot register hook at module %s: Not found'%layer)
            self.layers += layers
        else:
            self.handles.append(net.register_forward_hook(self.forward_hook))

    def unregister_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.layers = []

    def close(self):
        self.unregister_hooks()

    def forward_hook(self, module, input, output):
        raise NotImplementedError


def iou(activation, groundtruth, thresholds, channel_selection, num_classes, invalid_class=0, batch_size=4):
    """
    :param activation:
    :param groundtruth:
    :return: Dictionary, assigning each label in groundtruth a mean accuracy per chosen softmax threshold
    """

    num_units = activation.shape[0]

    results = torch.zeros(4, thresholds.shape[-1], num_units, num_classes, dtype=torch.long, device=activation.device)  # 4 for (area_c=1[pred+label], area_c=0[pred+label], intersection_c=1, intersection_c=0)
    thresholds = thresholds.to(activation.device).float()
    if activation.is_cuda:
        groundtruth = groundtruth.cuda()

    if len(thresholds.shape) == 1:
        thresholds = thresholds.unsqueeze(0).expand(num_units, -1)
    thresholds = thresholds.transpose(1,0)

    if len(groundtruth.shape) == 3:
        valid = (groundtruth != invalid_class).float()
        valid = valid.sum(0)
        valid.clamp_(0, 1)
        valid = valid.bool()
    else:
        valid = groundtruth != invalid_class
    valid = valid.view(-1)

    assert valid.sum() > 0, 'At least one pixel must be valid in groundtruth.'

    pos_act = torch.zeros(num_classes, groundtruth.shape[-2]*groundtruth.shape[-1], dtype=torch.float, device=activation.device)
    neg_act = torch.zeros(num_classes, groundtruth.shape[-2]*groundtruth.shape[-1], dtype=torch.float, device=activation.device)
    # for i, label_idx in enumerate(np.arange(invalid_class+1, invalid_class+1+num_classes)):
    # num_classes_offset = 0
    # if invalid_class >= 0 and invalid_class < num_classes:
    #     num_classes_offset += 1
    for i, label_idx in enumerate(range(num_classes)):
        label_idx = int(label_idx)
        pos = (groundtruth == label_idx).float()
        neg = (groundtruth != label_idx).float()
        # pos = groundtruth == label_idx
        # neg = groundtruth != label_idx

        if len(pos.shape) == 3:
            pos = pos.sum(0)
            neg = neg.sum(0)

        pos_act[i] = pos.view(-1)
        neg_act[i] = neg.view(-1)

    if channel_selection is None:
        std_selection = torch.arange(num_units, device=activation.device)
        channel_selection = std_selection

    # area_label_class1 = (pos_act & valid).view(num_classes, -1).sum(-1)
    # area_label_class0 = (neg_act & valid).view(num_classes, -1).sum(-1)
    area_label_class1 = (pos_act * valid).sum(-1)
    area_label_class0 = (neg_act * valid).sum(-1)

    intersection_class1 = area_label_class1.new(thresholds.shape[0], num_units, num_classes)
    intersection_class0 = area_label_class1.new(thresholds.shape[0], num_units, num_classes)

    # Loop over thresholds as each threshold may come with a different set of active channels
    ti_bs = thresholds.shape[0]
    thresholds = thresholds.unsqueeze(2)
    for ti in range(0, thresholds.shape[0], ti_bs):
        for bi in range(0, len(channel_selection), batch_size):
            unit_inds = channel_selection[bi:(bi+batch_size)]

            unit_acts = activation[unit_inds].view(num_units, -1)
            thresholds_ = thresholds[ti:ti+ti_bs, unit_inds]

            unit_act_ = (unit_acts.unsqueeze(0) > thresholds_).float()  # (nthresh, nunits, H, W)
            pred_class1 = unit_act_ * valid
            pred_class0 = (1.-unit_act_) * valid            

            nT, nN, HW = pred_class1.shape
            area_pred_class1 = pred_class1.sum(-1).unsqueeze(2).expand(-1, -1, num_classes)
            area_pred_class0 = pred_class0.sum(-1).unsqueeze(2).expand(-1, -1, num_classes)

            # print(pred_class1.shape, pos_act.shape)
            # intersection_class1 = (pred_class1.unsqueeze(2) * pos_act).view(nT, nN, num_classes, -1).sum(-1)
            # intersection_class0 = (pred_class0.unsqueeze(2) * neg_act).view(nT, nN, num_classes, -1).sum(-1)

            # print(pred_class1.shape, pos_act.shape)
            for ci in range(num_classes):
                intersection_class1[:, :, ci] = (pred_class1 * pos_act[ci].unsqueeze(0).unsqueeze(0)).sum(-1)
                intersection_class0[:, :, ci] = (pred_class0 * neg_act[ci].unsqueeze(0).unsqueeze(0)).sum(-1)

            # ## Attempt with booleans:
            # unit_act_ = (unit_acts.unsqueeze(0) > thresholds_)
            # unit_act_ = unit_act_.unsqueeze(2).expand(-1, -1, pos_act.shape[0], -1, -1)
            # pred_class1 = unit_act_ & valid
            # pred_class0 = (~unit_act_) & valid

            # nT, nN, nC, H, W = pred_class1.shape
            # area_pred_class1 = pred_class1.view(nT, nN, nC, -1).sum(-1)
            # area_pred_class0 = pred_class0.view(nT, nN, nC, -1).sum(-1)

            # pred_class1 &= pos_act
            # intersection_class1 = pred_class1.view(nT, nN, nC, -1).sum(-1)

            # pred_class0 &= neg_act
            # intersection_class0 = pred_class0.view(nT, nN, nC, -1).sum(-1)

            # print(area_pred_class1.shape, area_label_class1.shape, intersection_class1.shape)
            area_union_class1 = area_pred_class1 + area_label_class1 - intersection_class1
            area_union_class0 = area_pred_class0 + area_label_class0 - intersection_class0

            results[0, ti:ti+ti_bs, unit_inds] = area_union_class0.long()
            results[1, ti:ti+ti_bs, unit_inds] = area_union_class1.long()
            results[2, ti:ti+ti_bs, unit_inds] = intersection_class0.long()
            results[3, ti:ti+ti_bs, unit_inds] = intersection_class1.long()

    return results.cpu().numpy()


def apply_nonlinearity(output, which=None, **kwargs):
    if which is None:
        return output
    if which == 'sigmoid':
        output = torch.sigmoid(output)
    elif which == 'softmax':
        output = F.softmax(output, 1)
    elif which == 'attention':
        nSB = output.shape[1]
        output_raw = F.relu(output[:,:nSB//2])
        masks = torch.sigmoid(output[:,nSB//2:])
        output = output_raw * masks
    elif which == 'attention_only_mask':
        nSB = output.shape[1]
        output = torch.sigmoid(output[:,nSB//2:])
    elif which == 'VKQattention':
        nI, nSB, H, W = output.shape
        assert nSB % 4 == 0, 'VKQattention requires 4 heads, but width={} is not divisable by 4'.format(nSB)
        # assert 'gamma' in kwargs
        n = nSB//4

        keys = output[:,n:(2*n)]
        return torch.sigmoid(keys)
    elif which == 'VK-Q-attention':
        nI, nSB, H, W = output.shape
        assert nSB % 4 == 0, 'VK-Q-attention requires 4 heads, but width={} is not divisable by 4'.format(nSB)
        n = nSB//4

        queries = output[:,(2*n):(3*n)]
        return torch.sigmoid(queries)
    elif which == 'VKQmulattention':
        nI, nSB, H, W = output.shape
        assert nSB % 4 == 0, 'VKQattention requires 4 heads, but width={} is not divisable by 4'.format(nSB)
        n = nSB//4
        V = output[:,:n].view(nI, n, H*W)
        K = output[:,n:(2*n)].view(nI, n, H*W)
        Q = output[:,(2*n):(3*n)].view(nI, n, H*W)
        x = output[:,-n:]

        attention = F.softmax(torch.bmm(Q.permute(0,2,1).contiguous(), K), dim=-1)  # nI x (H*W) x (H*W)
        output = torch.bmm(V, attention.permute(0,2,1))
        output = torch.sigmoid(output.view(nI, n, H, W))
        return output * x
    elif which == 'VKQaddattention':
        raise NotImplementedError
        # output = kwargs['gamma'] * output.view(nI, n, H, W) + x
    else:
        raise RuntimeError('Unknown nonlinearity {}'.format(which))
    return output


class LayerOutputVisualizer(LayerOutputInspector):
    def __init__(self, vis_config, grab='output', apply_nonlinearity=None):
        super(LayerOutputVisualizer, self).__init__()

        import matplotlib
        matplotlib.use('Agg')

        self.config = vis_config
        self.grab = grab
        self.apply_nonlinearity = apply_nonlinearity
        self.state = Namespace(image_idx=-1, image=None, groundtruths=None, categories=None)

        # determine channel assignments based on iou_stats
        mious = vis_config['mious']
        self.raw_mious = mious
        mious[np.isnan(mious)]=0
        self.mious = mious.max(0)

        self.max_miou_per_unit = np.argsort(np.max(self.mious, 1))[::-1]

        self.unit_to_classes_sorted = np.argsort(self.mious, 1)[:, ::-1]

        self.class_labels = self.read_class_labels(self.config['domain_config']['visualize']['class_labels'])
        self.unit_to_class_name = self.assign_class_labels(self.mious, self.class_labels, 3)

    def register_input_sample(self, image, groundtruths, categories):
        self.state.image = image
        self.state.groundtruths = groundtruths
        self.state.categories = categories
        self.state.image_idx += 1

    def reset(self):
        self.unregister_hooks()
        self.state = Namespace(image_idx=-1, image=None, groundtruths=None, categories=None)

    @staticmethod
    def assign_class_labels(mious, class_labels, topn=3):
        assignments = []

        sorted_mious = np.argsort(mious, 1)[:, ::-1]

        for ui in range(sorted_mious.shape[0]):
            ui_assignments = []
            for i in range(topn):
                ci = sorted_mious[ui, i]
                ui_assignments.append((ui,ci))
            assignments.append(ui_assignments)
            # print('{} Assignments: {}'.format(ui, ui_assignments))

        return assignments

    @staticmethod
    def read_class_labels(label_fp):
        with open(label_fp, 'r') as f:
            lines = [l.strip() for l in f.readlines() if not '__ignore__' in l]
        return lines

    def get_category_masks(self):
        categories = []
        if self.config['sb_config'] is None:
            if self.config['channel_selection'] is not None:
                numel = len(self.config['channel_selection'])
            else:
                # iou_stats: type, nThresholds, nChannels, nConcepts
                numel = self.config['iou_stats'].shape[2]
            return [np.ones((numel,)).astype(np.bool)]
        if self.config['sb_config'].get('type', 'usb') == 'ssb':
            widths = self.config['sb_config']['widths']
            numel = sum(widths)
            if self.config['domain_config']['visualize'].get('join_bottlenecks', False):
                mask = np.ones((numel,)).astype(np.bool)
                categories.append(mask)
            else:
                offset = 0
                for w in self.config['sb_config']['widths']:
                    cat_mask = np.zeros((numel,)).astype(np.bool)
                    cat_mask[offset:(offset+w)] = True
                    categories.append(cat_mask)
                    offset += w
        else:
            offset = 0
            width = self.config['sb_config']['width']
            numel = width * self.config['sb_config']['num_parallel']
            
            if self.config['domain_config']['visualize'].get('join_bottlenecks', False):
                mask = np.ones((numel,)).astype(np.bool)
                categories.append(mask)
            else:
                for wi in range(self.config['sb_config']['num_parallel']):
                    cat_mask = np.zeros((numel,)).astype(np.bool)
                    cat_mask[offset:(offset+width)] = True
                    categories.append(cat_mask)
                    offset += width

        if self.config['channel_selection'] is not None:
            for i in range(len(categories)):
                categories[i] = categories[i][self.config['channel_selection']]

        return categories

    def plot_individual_channels(self, output):
        import matplotlib.pyplot as plt
        from scipy.ndimage.morphology import binary_dilation
        # units_sorted_by_score = config.units_sorted_by_score
        # unit_to_classid = config.unit_to_classid
        # unit_to_class = config.unit_to_class
        thresholds = self.thresholds
        best_thresholds = np.argmax(self.raw_mious, 0)
        # auc = self.config['auics']
        std = np.array(self.config['std'])[np.newaxis, np.newaxis, ...]
        mean = np.array(self.config['mean'])[np.newaxis, np.newaxis, ...]
        input_image = ((self.state.image * std) + mean) / 255.
        input_image = input_image.clip(0,1)
        output_ = output.numpy()

        units_sorted_by_score = np.argsort(np.max(self.mious, 1))[::-1]

        save_folder = os.path.join(self.config['save_folder'], 'image_%05d'%self.state.image_idx)
        os.makedirs(save_folder, exist_ok=True)

        for k, ui in enumerate(units_sorted_by_score):
            ci = self.unit_to_class_name[ui][0][1]
            # print(thresholds.shape, best_thresholds.shape)
            # print(ui,ci, best_thresholds[ui, ci])
            thresh = max(0, thresholds[ui, best_thresholds[ui, ci]])
            mask = np.clip(output_[ui], a_min=thresh, a_max=None) / thresholds[ui, -1]
            binary_mask = (mask>=thresh).astype(np.float32)
            border = binary_dilation(binary_mask) - binary_mask
            border = border[..., np.newaxis].repeat(4, axis=-1)
            mask = mask[..., np.newaxis]

            fig = plt.figure(figsize=(20, 6))
            composition = np.concatenate((input_image, mask.clip(0.2,1)), axis=2)
            plt.imshow(np.zeros_like(input_image))
            plt.imshow(composition)
            plt.imshow(border)
            # plt.gca().set_facecolor((0.0, 0.0, 0.0))
            plt.axis('off')
            plt.savefig(os.path.join(save_folder, 'rank%04d_unit%04d.png' % (k, ui)), bbox_inches='tight', pad_inches=0)
            plt.close(fig)


    def plot_argmax(self, output, topns):
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        # print('max_miou_per_unit', self.max_miou_per_unit)
        # self.max_miou_per_unit

        class_names = self.unit_to_class_name

        maccs = self.mious

        colormap = None #config.colormap  # TODO
        # unit_to_concept = self.max_miou_per_unit
        unit_to_concept = np.argmax(self.mious, 1)

        categories = self.get_category_masks()

        nrows = 1
        ncols = 2

        image_kws = [[dict(arr=None, cmap=None, vmin=0, vmax=None) for topn in topns] for _ in categories]
        legend_figs = [[plt.subplots(1, 1, figsize=(6,4), dpi=150) for topn in topns] for _ in categories]

        cat_offset = 0
        for category, mask in enumerate(categories):
            output_ = output.numpy()
            
            # units_sorted_by_score = self.max_miou_per_unit[mask])[::-1]
            units_sorted_by_score = np.argsort(np.max(self.mious, 1)[mask])[::-1]
            output_ = output_[mask][units_sorted_by_score]

            class_names_ = np.array(class_names)[mask][units_sorted_by_score]
            concepts = unit_to_concept[mask][units_sorted_by_score]
            maccs_ = maccs[mask][units_sorted_by_score]

            for topni, topn in enumerate(topns):

                if topn > 0:
                    output__ = output_[:topn]
                    class_names__ = class_names_[:topn]
                    concepts = concepts[:topn]
                    maccs__ = maccs_[:topn]
                else:
                    output__ = output_[topn:][::-1]
                    class_names__ = class_names_[topn:][::-1]
                    concepts = concepts[topn:][::-1]
                    maccs__ = maccs_[topn:][::-1]

                N = output__.shape[0]

                if colormap is not None and topn > 0:
                    cmaplist = []
                    for concept in concepts:
                        # concept = unit_to_concept[ui]
                        if concept not in colormap:
                            raise RuntimeError('Concept %d has no assigned color in colormap' % concept)
                        cmaplist.append(colormap[concept])
                else:
                    if N > 11:
                        cmaplist = plt.cm.get_cmap('tab20c', N).colors
                    else:
                        cmaplist = plt.cm.get_cmap('tab10', N).colors

                    class_names__ = class_names__.tolist()

                im_amax = np.argmax(output__, axis=0)

                # define the bins and normalize
                bounds = np.linspace(0, N, N+1)

                cmap = mpl.colors.LinearSegmentedColormap.from_list('SB cmap', cmaplist, N)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

                # make the plot
                image_kws[category][topni]['arr'] = im_amax
                image_kws[category][topni]['cmap'] = cmap
                # image_kws[category][topni]['vmax'] = topn-1

                # -----------------------------------------------------------------
                # define second axes for the colorbar
                ax = legend_figs[category][topni][1]
                ax.set_position([0, 0, 0.06, 1])

                cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                               spacing='proportional', ticks=bounds + 0.5, boundaries=bounds, format='%1i')

                legend_labels = []
                for k, assignments in enumerate(class_names__):
                    legend_label = []
                    for ui,ci in assignments:
                        cmiou = float(self.mious[ui,ci])*100
                        cname = self.class_labels[ci]
                    # for cmiou, cname in cnames: #cij in range(len(cnames)):
                        # cmiou = float(cmiou)*100
                        legend_label.append('%d%%: %s'%(int(cmiou), cname))

                    legend_labels.append(','.join(legend_label))
                # if os.path.basename(config.model).startswith('SB'):
                if topn > 40:
                    _ = cb.ax.set_yticklabels(legend_labels, fontdict=dict(size=6))
                else:
                    _ = cb.ax.set_yticklabels(legend_labels)
                ax.invert_yaxis()

            cat_offset += np.sum(mask)

        for ci, kws in enumerate(image_kws):
            for ti, topn in enumerate(topns):
                plt.imsave(os.path.join(self.config['save_folder'], 'image_%05d-cat%d-argmax-top%d.png' % (self.state.image_idx, ci, topn)), **kws[ti])
                legend_figs[ci][ti][0].savefig(os.path.join(self.config['save_folder'], 'image_%05d-cat%d-argmax-topn%d-colorbar.png' % (self.state.image_idx, ci, topn)))


    def visualize(self, output):
        import matplotlib.pyplot as plt
        if self.config['channel_selection'] is not None:
            output = output[self.config['channel_selection']]
            thresholds = self.config['thresholds'].copy()
            self.thresholds = thresholds[self.config['channel_selection']]
        else:
            self.thresholds = self.config['thresholds']

        kwargs = dict(
            mode = 'bilinear',
            align_corners = True)
        if self.config['sb_config'] is not None:            
            if self.config['sb_config'].get('type', 'usb') == 'usb':
                kwargs['mode'] = 'nearest'
                del kwargs['align_corners']

        self.config['logger'].info('Visualizing image {}'.format(self.state.image_idx))

        output = F.interpolate(output.unsqueeze(0), size=self.state.groundtruths[0].shape[-2:], **kwargs).squeeze().cpu()

        # plot argmax
        topns = self.config['domain_config'].get('topns', [5, 10, 20, 40])
        self.plot_argmax(output, topns)
        plt.close('all')

        # plot individual
        self.plot_individual_channels(output)


    def forward_hook(self, module, input, output):
        if self.grab == 'input':
            output = input
            assert type(output) is tuple
            output = output[0]

        if type(output) is tuple and isinstance(output[0], torch.Tensor):
            output = output[0]

        output = apply_nonlinearity(output, which=self.apply_nonlinearity)

        assert output.shape[0] == 1
        output = output[0]

        self.visualize(output)



class LayerOutputLimits(LayerOutputInspector):
    def __init__(self, grab='output', apply_nonlinearity=None):
        super(LayerOutputLimits, self).__init__()
        self.grab = grab
        self.mins = None
        self.maxs = None
        self.apply_nonlinearity = apply_nonlinearity
        assert apply_nonlinearity is None
        # print('Applying nonlinearity {}'.format(self.apply_nonlinearity))

    def forward_hook(self, module, input, output):
        if self.grab == 'input':
            output = input
            assert type(output) is tuple
            output = output[0]

        N, K, H, W = output.shape
        if self.mins is None:
            self.mins = output.view(N, K, -1).min(-1)[0].min(0)[0]
            self.maxs = output.view(N, K, -1).max(-1)[0].min(0)[0]
        else:
            mins = output.view(N, K, -1).min(-1)[0].min(0)[0]
            maxs = output.view(N, K, -1).max(-1)[0].min(0)[0]
            self.mins = torch.min(mins, self.mins)
            self.maxs = torch.max(maxs, self.maxs)

        # raise ForwardStopException


class AUIC(LayerOutputInspector):
    def __init__(self, thresholds, num_classes, invalid_classes, 
        channel_selections=None, grab='output', batchsize=4,
        apply_nonlinearity=None):
        """
        :param thresholds: list of floats
        :param num_classes: dictionary, assigning scalar to each category
        :param invalid_classes: dictionary, assigning scalar to each category
        """
        super(AUIC, self).__init__()
        self.results_by_category = {}
        self.channel_miou_by_category = {}
        self.state = Namespace(image=None, groundtruths=None, categories=None)
        self.thresholds = thresholds
        if channel_selections is not None:
            assert len(self.thresholds) == len(channel_selections)
            self.thresholds = thresholds[channel_selections]
        self.num_classes = num_classes
        self.invalid_class = invalid_classes
        self.grab = grab
        self.bs = batchsize
        self.channel_selections = channel_selections
        self.apply_nonlinearity = apply_nonlinearity
        self.dump_idx = 0
        # print('Applying nonlinearity {}'.format(self.apply_nonlinearity))

    def register_input_sample(self, image, groundtruths, categories):
        self.state.image = image
        self.state.groundtruths = groundtruths
        self.state.categories = categories

    def reset(self):
        self.unregister_hooks()
        self.state = Namespace(image=None, groundtruths=None, categories=None)

    def dump_outputs(self, output, prefix=''):
        import os
        from PIL import Image
        from torchvision.utils import make_grid

        if not os.path.exists('dumped_outputs'):
            os.makedirs('dumped_outputs')

        grid = make_grid(output[0].unsqueeze(1), pad_value=0.5, normalize=True, scale_each=True)
        grid = grid.permute(1,2,0)
        grid = (grid*255).long()
        im = Image.fromarray(grid.cpu().numpy().astype(np.uint8))
        im.save(os.path.join('dumped_outputs', prefix+'output_%03d.png'%self.dump_idx))
        self.dump_idx += 1

    def forward_hook(self, module, input, output):
        if self.grab == 'input':
            output = input
            assert type(output) is tuple
            output = output[0]

        if type(output) is tuple and isinstance(output[0], torch.Tensor):
            output = output[0]

        if type(self.apply_nonlinearity) is str:
            output = apply_nonlinearity(output, which=self.apply_nonlinearity)
        elif self.apply_nonlinearity is not None:
            output = self.apply_nonlinearity(output)
        # output = apply_nonlinearity(output, which=self.apply_nonlinearity)

        # self.dump_outputs(output)

        if self.channel_selections is not None and len(self.channel_selections.shape)==1:
            # print('Active channel selection. Original output: {}'.format(output.shape))
            output = output[:, self.channel_selections]
            # print('Reduced output: {}'.format(output.shape))

        num_units = output.shape[1]
        unitbs = min(num_units, self.bs)
        # assert (num_units%unitbs)==0, 'Unit-batchsize must be divider of number of units. #Units: {}, bs: {}'.format(num_units, unitbs)

        for category, groundtruth in zip(self.state.categories, self.state.groundtruths):
            if len(groundtruth.shape) == 2:
                groundtruth = groundtruth.unsqueeze(0)
            num_classes = self.num_classes[category]

            # if groundtruth.shape[1]%output.shape[-2] == 0:
            #     ds = groundtruth.shape[1]//output.shape[-2]
            #     gt = torch.from_numpy(majority_kernel(groundtruth[0].numpy(), ds, ds, set_even_to=self.invalid_class[category]))
            # else:
            #     gt = torch.from_numpy(imresize(groundtruth.numpy(), output.shape[-2:], 'nearest', mode='F'))

            gt = groundtruth.to(output.device)

            self.process(category, gt, num_classes, num_units, output, unitbs)

        # raise ForwardStopException

    def process(self, category, gt, num_classes, num_units, output, unitbs):
        results = np.zeros((4, self.thresholds.shape[-1], num_units, num_classes))
        invalid_class = self.invalid_class[category]

        # activation_ = F.interpolate(output, size=gt.shape[-2:], mode='bilinear', align_corners=True)
        # torch.save(dict(target=gt.cpu(), output=activation_.cpu(), image=torch.from_numpy(self.state.image)), 'auic_state.pth')
        # print('Saved')
        # raise RuntimeError

        # gt_ = gt.unsqueeze(0).float()
        # gt_[gt_==255] = -1
        # gt_ += 1
        # self.dump_outputs(gt_/255., prefix='gt_')

        # resize gt to match layer output shape
        gt = F.interpolate(gt.unsqueeze(0).float(), size=output.shape[-2:], mode='bilinear', align_corners=True).long().squeeze(0)

        uniques = np.unique(gt.cpu().numpy())
        if len(uniques) > 1 or uniques[0] != invalid_class:
            # print(uniques)
            for ui in range(0, num_units, unitbs):
                activation = output[:, ui:ui + unitbs]
                # print('activation shape', activation.shape, output.shape)

                # # resize layer output to match groundtruth shape
                # activation = F.interpolate(activation, size=gt.shape[-2:], mode='bilinear', align_corners=True)

                assert activation.shape[0] == 1
                activation = activation[0]

                result = iou(activation, gt, self.thresholds[ui:ui+unitbs], None, num_classes, invalid_class, batch_size=self.bs)
                results[:, :, ui:ui + unitbs] = result

        pos = (results[3] / results[1])
        neg = (results[2] / results[0])
        mious = 0.5*(pos + neg)
        self.channel_miou_by_category[category] = mious

        if category not in self.results_by_category:
            self.results_by_category[category] = results
        else:
            self.results_by_category[category] += results
            