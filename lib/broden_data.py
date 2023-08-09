import numpy as np
import os
import cv2

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class BrodenData(Dataset):
    def __init__(self, split, concept_class, data_root, transform=None, num_samples=-1):
        self.split = split
        self.concept_class = concept_class
        self.num_classes = self.read_concept_count(data_root)
        self.data_list = self.assemble_data_list(data_root)
        self.transform = transform

        if num_samples is not None and num_samples > 0:
            print('Shorting Broden data list length from {} to {}'.format(len(self.data_list), num_samples))
            np.random.seed(0)
            np.random.shuffle(self.data_list)
            self.data_list = self.data_list[:num_samples]

    def read_concept_count(self, data_root):
        num_classes = {}
        with open(os.path.join(data_root, 'category.csv'), 'r') as f:
            f.readline()  # skip header
            for line in f.readlines():
                entries = line.split(',')
                num_classes[entries[0].strip()] = int(entries[3].strip())
        assert self.concept_class in num_classes
        return num_classes[self.concept_class]

    def assemble_data_list(self, data_root):
        split = self.split
        concept_class = self.concept_class
        data_list = []
        with open(os.path.join(data_root, 'index.csv'), 'r') as f:
            header = f.readline()
            columns = {}
            for i, entry in enumerate(header.split(',')):
                columns[entry.strip()] = i

            for line in f.readlines():
                data_pair = []
                entries = [v.strip() for v in line.split(',')]
                
                if entries[columns[concept_class]] and entries[columns['split']] == split:
                    fname = os.path.join(data_root, 'images', entries[columns['image']])
                    for segm in entries[columns[concept_class]].split(';'):
                        if concept_class != 'texture':
                            segm = os.path.join(data_root, 'images', segm)
                            assert os.path.exists(fname), fname
                            assert os.path.exists(segm), segm
                        else:
                            segm = int(segm)
                        data_pair = (fname, segm)
                        data_list.append(data_pair)

        assert len(data_list) > 0

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        if type(label_path) is int:
            texture_id = label_path
            label = np.zeros((image.shape[:2])) + texture_id
        else:
            label = cv2.imread(label_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray with shape H * W * 3
            label = label.astype(np.int32)
            label = label[:,:,2] + label[:,:,1] * 256  # According to https://github.com/CSAILVision/NetDissect-Lite/blob/master/loader/data_loader.py#L202
            label = cv2.resize(label, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        # print('BrodenData', self.concept_class, np.unique(label))
        # if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        #     raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label