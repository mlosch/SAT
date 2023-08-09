from torch.utils.data import Dataset
import torch
import torchvision
from PIL import Image
import numpy as np
import os


# class SUN397(torchvision.datasets.SUN397):
#     def __init__(self, *args, split, split_length=0.7, **kwargs):
#         super(SUN397, self).__init__(*args, **kwargs)
#         generator = torch.Generator().manual_seed(123)
#         assert split in ['train', 'val']
#         assert 0 < split_length <= 1
#         subsets = torch.utils.data.random_split(self, [split_length, 1-split_length], generator=generator)
#         if split == 'train':
#             subset = subsets[0]
#         else:
#             subset = subsets[1]

#         self.subset_indices = subset.indices

#     def __getitem__(self, idx):
#         if hasattr(self, 'subset_indices'):
#             if isinstance(idx, list):
#                 return super(SUN397, self).__getitem__([[self.subset_indices[i] for i in idx]])
#             return super(SUN397, self).__getitem__(self.subset_indices[idx])
#         else:
#             return super(SUN397, self).__getitem__(idx)

#     def __len__(self):
#         if hasattr(self, 'subset_indices'):
#             return len(self.subset_indices)
#         else:
#             return super(SUN397, self).__len__()

class Caltech256(torchvision.datasets.ImageFolder):
    def __init__(self, root, split, **kwargs):
        super(Caltech256, self).__init__(root, **kwargs)
        data_list = os.path.join(os.path.dirname(root.rstrip('/')), split+'.txt')

        imgs = []
        targets = []
        with open(data_list, 'r') as f:
            for line in f.readlines():
                filepath, label = line.strip().split(' ')
                imgs.append((filepath, int(label)))
                targets.append(int(label))

        self.imgs = imgs
        self.samples = imgs
        self.targets = targets


class CIFAR10CatVsCar(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFAR10CatVsCar, self).__init__(*args, **kwargs)

        #filter entries not of interest
        new_data = []
        new_targets = []
        for i in range(len(self)):
            if self.targets[i] in [1, 3]:
                new_data.append(self.data[i])
                new_targets.append(self.targets[i])
        self.data = new_data
        self.targets = new_targets

class CIFAR10Subset(torchvision.datasets.CIFAR10):
    def __init__(self, domain, *args, **kwargs):
        super(CIFAR10Subset, self).__init__(*args, **kwargs)
        self.domain = domain

        label_map = dict()
        new_idx = 0
        for class_idx in range(10):
            if class_idx in domain:
                label_map[class_idx] = new_idx
                new_idx += 1

        assert new_idx == len(domain)

        new_data = []
        new_targets = []
        for i in range(len(self)):
            label = self.targets[i]
            if label in label_map:
                new_data.append(self.data[i])
                new_targets.append(label_map[label])
        self.data = new_data
        self.targets = new_targets


class CIFAR10Oversampled(torchvision.datasets.CIFAR10):
    def __init__(self, repetition_per_class, *args, **kwargs):
        super(CIFAR10Oversampled, self).__init__(*args, **kwargs)

        new_data = []
        new_targets = []
        for i in range(len(self)):
            label = self.targets[i]
            for n in range(repetition_per_class[label]):
                new_data.append(self.data[i])
                new_targets.append(self.targets[i])
        self.data = new_data
        self.targets = new_targets


class CIFAR10WithMaxEntropyData(torchvision.datasets.CIFAR10):
    def __init__(self, max_ent_filep, *args, **kwargs):
        super(CIFAR10WithMaxEntropyData, self).__init__(*args, **kwargs)
        import torch
        import numpy as np

        max_ent = torch.load(max_ent_filep)
        max_ent_data = max_ent['images'].numpy()
        # max_ent_targets = max_ent['labels'].numpy().tolist()

        # permute order
        data = []
        targets = []
        for i in range(len(max_ent_data)):
            data.append((max_ent_data[i].transpose(1,2,0)*255).astype(np.uint8)[None, ...])
            targets.append(10)

        data = np.concatenate(data, axis=0)

        # print(type(self.data[0]))
        # print(self.data[0].shape)
        # print(self.targets[0])

        self.data = np.concatenate([self.data, data], axis=0)
        self.targets += targets


class CIFAR10WithIndex(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        x,y = super(CIFAR10WithIndex, self).__getitem__(index)
        return x, y, {'imids': index}


class CIFAR100Subset(torchvision.datasets.CIFAR100):
    def __init__(self, domain, *args, **kwargs):
        super(CIFAR100Subset, self).__init__(*args, **kwargs)
        self.domain = domain

        label_map = dict()
        new_idx = 0
        for class_idx in range(100):
            if class_idx in domain:
                label_map[class_idx] = new_idx
                new_idx += 1

        assert new_idx == len(domain)

        new_data = []
        new_targets = []
        for i in range(len(self)):
            label = self.targets[i]
            if label in label_map:
                new_data.append(self.data[i])
                new_targets.append(label_map[label])
        self.data = new_data
        self.targets = new_targets

        print('CIFAR100 Subset [{}] :: Number of instances {}'.format(domain, len(self.data)))


class CIFAR100InstanceSubset(torchvision.datasets.CIFAR100):
    def __init__(self, domain, *args, **kwargs):
        super(CIFAR100InstanceSubset, self).__init__(*args, **kwargs)
        self.domain = domain

        new_data = []
        new_targets = []
        stats = [0]*100
        for i in range(len(self)):
            if i in domain:
                new_data.append(self.data[i])
                new_targets.append(self.targets[i])
                stats[self.targets[i]] += 1
        self.data = new_data
        self.targets = new_targets

        print('CIFAR100 Instance Subset has the following number of images per class:')
        for i in range(len(stats)):
            if stats[i] == 0:
                print('\t{:02d} - {} = #{} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(i, self.classes[i], stats[i]))
            else:
                print('\t{:02d} - {} = #{}'.format(i, self.classes[i], stats[i]))


# class CIFAR100InstanceSubsetWithIndex(CIFAR100InstanceSubset):
#   def __getitem__(self, index):
#       x,y = super(CIFAR100InstanceSubsetWithIndex, self).__getitem__(index)
#       return x, y, {'imids': index}


class CIFAR100WithIndex(torchvision.datasets.CIFAR100):
    def __getitem__(self, index):
        x,y = super(CIFAR100WithIndex, self).__getitem__(index)
        return x, y, {'imids': index}


class CIFAR10And100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super(CIFAR10And100, self).__init__(*args, **kwargs)

        cifar10 = torchvision.datasets.CIFAR10(*args, **kwargs)

        new_data = []
        new_targets = []
        for i in range(len(self)):
            new_data.append(self.data[i])
            new_targets.append(self.targets[i])
        for i in range(len(cifar10)):
            new_data.append(cifar10.data[i])
            new_targets.append(cifar10.targets[i]+100)

        self.data = new_data
        self.targets = new_targets


class CIFAR10PlusOne(torchvision.datasets.CIFAR10):
    def __init__(self, *args, c100_classes=[], **kwargs):
        super(CIFAR10PlusOne, self).__init__(*args, **kwargs)

        cifar100 = torchvision.datasets.CIFAR100(*args, **kwargs)

        new_data = []
        new_targets = []
        additional_label_idx = 10
        label_map = dict()
        for i in range(len(self)):
            new_data.append(self.data[i])
            new_targets.append(self.targets[i])
        for i in range(len(cifar100)):
            target = cifar100.targets[i]
            if target in c100_classes:
                new_data.append(cifar100.data[i])
                new_targets.append(additional_label_idx)
                

        self.data = new_data
        self.targets = new_targets
        print('unique labels:', np.unique(new_targets))


class CIFAR10AndSubsetCIFAR100(torchvision.datasets.CIFAR10):
    def __init__(self, *args, c100_subset=[], **kwargs):
        super(CIFAR10AndSubsetCIFAR100, self).__init__(*args, **kwargs)

        cifar100 = torchvision.datasets.CIFAR100(*args, **kwargs)

        new_data = []
        new_targets = []
        additional_label_idx = 10
        label_map = dict()
        for i in range(len(self)):
            new_data.append(self.data[i])
            new_targets.append(self.targets[i])
        for i in range(len(cifar100)):
            target = cifar100.targets[i]
            if target in c100_subset:
                if target not in label_map:
                    label_map[target] = additional_label_idx
                    additional_label_idx += 1
                new_data.append(cifar100.data[i])
                new_targets.append(label_map[target])
                

        self.data = new_data
        self.targets = new_targets
        print('unique labels:', np.unique(new_targets))


class RestrictedCIFAR10And100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, num_samples_per_cifar10_class=np.inf, **kwargs):
        super(RestrictedCIFAR10And100, self).__init__(*args, **kwargs)

        cifar10 = torchvision.datasets.CIFAR10(*args, **kwargs)

        new_data = []
        new_targets = []
        for i in range(len(self)):
            new_data.append(self.data[i])
            new_targets.append(self.targets[i])

        nitems_per_class = np.zeros((10,))
        for i in range(len(cifar10)):
            if nitems_per_class[cifar10.targets[i]] >= num_samples_per_cifar10_class:
                continue
            new_data.append(cifar10.data[i])
            new_targets.append(cifar10.targets[i]+100)
            nitems_per_class[cifar10.targets[i]] += 1

        self.data = new_data
        self.targets = new_targets


class MNISTWithIndex(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        x,y = super(MNISTWithIndex, self).__getitem__(index)
        return x, y, {'imids': index}


RESTRICTED_IMAGNET_RANGES = [
    (151, 268), # dog
    (281, 285), # cat
    (30, 32), # frog
    (33, 37), # turtle
    (80, 100), # bird
    (365, 382), # monkey
    (389, 397), # fish
    (118, 121), # crab
    (300, 319) # insect
]

class RestrictedImageNet(torchvision.datasets.ImageNet):
    def __init__(self, root, split = 'train', **kwargs):
        super(RestrictedImageNet, self).__init__(root, split, **kwargs)

        in_ids = dict()
        for class_idx, (a,b) in enumerate(RESTRICTED_IMAGNET_RANGES):
            in_ids[class_idx] = list(range(a, b+1))

        # map 
        new_samples = []
        for sample in self.samples:
            for class_idx, range_inds in in_ids.items():
                if sample[1] in range_inds:
                    new_samples.append((sample[0], class_idx))
        
        self.samples = new_samples

class RestrictedImageNetWithIndices(RestrictedImageNet):
    def __getitem__(self, index):
        x,y = super(RestrictedImageNetWithIndices, self).__getitem__(index)
        return x, y, {'imids': index}


class SubsetImageNet(torchvision.datasets.ImageNet):
    def __init__(self, root, split = 'train', wnid_selection=[], **kwargs):
        super(SubsetImageNet, self).__init__(root, split, **kwargs)

        in_ids = dict()
        for idx, wnid in enumerate(wnid_selection):
            in_ids[self.wnid_to_idx[wnid]] = idx

        new_samples = []
        targets = []
        classes = set()
        for sample in self.samples:
            if sample[1] in in_ids:
                new_class_idx = in_ids[sample[1]]
                new_samples.append((sample[0], new_class_idx))
                targets.append(new_class_idx)
                classes.add(new_class_idx)

        self.targets = targets
        self.samples = new_samples
        self.classes = list(classes)

class SubsetImageNetWithIndices(SubsetImageNet):
    def __getitem__(self, index):
        x,y = super(SubsetImageNetWithIndices, self).__getitem__(index)
        return x, y, {'imids': index}


class OxfordFlowers102(Dataset):
    def __init__(self, root, split, transform=None):
        super(OxfordFlowers102, self).__init__()
        self.root = root
        assert os.path.exists(os.path.join(root, 'imagelabels.mat'))
        assert os.path.exists(os.path.join(root, 'setid.mat'))
        assert os.path.exists(os.path.join(root, 'jpg')), 'Image folder missing'
        assert split in ['train', 'val', 'test']

        from scipy.io import loadmat
        setid_key = dict(train='trnid', val='valid', test='tstid')[split]
        split_indices = loadmat(os.path.join(root, 'setid.mat'))[setid_key]
        self.image_ids = split_indices.squeeze()

        self.labels = loadmat(os.path.join(root, 'imagelabels.mat'))['labels'].squeeze()
        self.labels -= 1 # index in .mat starts at 1
        self.labels = self.labels.astype(np.int64)

        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        label = self.labels[img_id]

        img = self.pil_loader(os.path.join(self.root, 'jpg', 'image_{:05d}.jpg'.format(img_id)))

        if self.transform is not None:
            img = self.transform(img)

        return img, label