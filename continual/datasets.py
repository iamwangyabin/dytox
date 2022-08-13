# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import json
import os
import warnings

from continuum import ClassIncremental
from continuum.datasets import CIFAR100, ImageNet100, ImageFolderDataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms import functional as Fv

from continuum.datasets import _ContinuumDataset
from continuum.tasks import TaskType
from typing import List, Tuple, Union
import numpy as np

try:
    interpolation = Fv.InterpolationMode.BICUBIC
except:
    interpolation = 3


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder




class DomainnetDataset(_ContinuumDataset):
    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = True,
            data_type: TaskType = TaskType.IMAGE_PATH,
    ):
        self.data_path = data_path
        self._data_type = data_type
        super().__init__(data_path=data_path, train=train, download=download)

        allowed_data_types = (TaskType.IMAGE_PATH, TaskType.SEGMENTATION)
        if data_type not in allowed_data_types:
            raise ValueError(f"Invalid data_type={data_type}, allowed={allowed_data_types}.")

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[None, np.ndarray]]:
        # rootdir = '/home/wangyabin/workspace/datasets/domainnet'
        train_txt = './domainnet_split/train.txt'
        test_txt = './domainnet_split/test.txt'
        dataset = []
        if self.train:
            with open(train_txt, 'r') as dict_file:
                for line in dict_file:
                    (value, key) = line.strip().split(' ')
                    dataset.append((os.path.join(self.data_path, value), int(key)))
        else:
            with open(test_txt, 'r') as dict_file:
                for line in dict_file:
                    (value, key) = line.strip().split(' ')
                    dataset.append((os.path.join(self.data_path, value), int(key)))

        self.dataset = dataset
        x, y, t = self._format(self.dataset)
        self.list_classes = np.unique(y)
        return x, y, t

    @staticmethod
    def _format(raw_data: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray, None]:
        x = np.empty(len(raw_data), dtype="S255")
        y = np.empty(len(raw_data), dtype=np.int16)

        for i, (path, target) in enumerate(raw_data):
            x[i] = path
            y[i] = target

        return x, y, None



class FiveDatasetsDataset(_ContinuumDataset):
    """Continuum dataset for 5 datasets.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = True,
            data_type: TaskType = TaskType.IMAGE_ARRAY,
    ):
        self.data_path = data_path
        self._data_type = data_type
        super().__init__(data_path=data_path, train=train, download=download)

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[None, np.ndarray]]:


        dataset = []

        img_size=64

        if self.train:
            cifar10_train_dataset = datasets.cifar.CIFAR10('./data', train=True, download=True)
            for img, target in zip(cifar10_train_dataset.data, cifar10_train_dataset.targets):
                dataset.append((np.array(Image.fromarray(img).resize((img_size, img_size))), target))

            minist_train_dataset = datasets.MNIST('./data', train=True, download=True)
            for img, target in zip(minist_train_dataset.data.numpy(), minist_train_dataset.targets.numpy()):
                dataset.append((np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')),
                               target + 10))

            classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            tarin_dir = "./data/notMNIST_large"
            for idx, cls in enumerate(classes):
                image_files = os.listdir(os.path.join(tarin_dir, cls))
                for img_path in image_files:
                    try:
                        image = np.array(Image.open(os.path.join(tarin_dir, cls, img_path)).resize((img_size, img_size)).convert('RGB'))
                        dataset.append((image, idx+20))
                    except:
                        print(os.path.join(tarin_dir, cls, img_path))

            fminist_train_dataset = datasets.FashionMNIST('./data', train=True, download=True)
            for img, target in zip(fminist_train_dataset.data.numpy(), fminist_train_dataset.targets.numpy()):
                dataset.append((np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')), target+30))

            svhn_train_dataset = datasets.SVHN('./data', split='train', download=True)
            for img, target in zip(svhn_train_dataset.data, svhn_train_dataset.labels):
                dataset.append((np.array(Image.fromarray(img.transpose(1, 2, 0)).resize((img_size, img_size))),
                                target + 40))

        else:
            cifar10_test_dataset = datasets.cifar.CIFAR10('./data', train=False, download=True)
            for img, target in zip(cifar10_test_dataset.data, cifar10_test_dataset.targets):
                dataset.append((np.array(Image.fromarray(img).resize((img_size, img_size))), target))

            minist_test_dataset = datasets.MNIST('./data', train=False, download=True)
            for img, target in zip(minist_test_dataset.data.numpy(), minist_test_dataset.targets.numpy()):
                dataset.append((np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')),
                               target + 10))

            classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            test_dir = "./data/notMNIST_small"
            for idx, cls in enumerate(classes):
                image_files = os.listdir(os.path.join(test_dir, cls))
                for img_path in image_files:
                    try:
                        image = np.array(Image.open(os.path.join(test_dir, cls, img_path)).resize((img_size, img_size)).convert('RGB'))
                        dataset.append((image, idx+20))
                    except:
                        print(os.path.join(test_dir, cls, img_path))

            fminist_test_dataset = datasets.FashionMNIST('./data', train=False, download=True)
            for img, target in zip(fminist_test_dataset.data.numpy(), fminist_test_dataset.targets.numpy()):
                dataset.append((np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')), target+30))

            svhn_test_dataset = datasets.SVHN('./data', split='test', download=True)
            for img, target in zip(svhn_test_dataset.data, svhn_test_dataset.labels):
                dataset.append((np.array(Image.fromarray(img.transpose(1, 2, 0)).resize((img_size, img_size))),
                                target + 40))

        self.dataset = dataset
        x, y, t = self._format(self.dataset)
        self.list_classes = np.unique(y)
        return x, y, t

    @staticmethod
    def _format(raw_data: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray, None]:
        x = np.empty(len(raw_data))
        y = np.empty(len(raw_data), dtype=np.int16)

        for i, (path, target) in enumerate(raw_data):
            x[i] = path
            y[i] = target

        return x, y, None





def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set.lower() == 'cifar':
        dataset = CIFAR100(args.data_path, train=is_train, download=True)
    elif args.data_set.lower() == 'imagenet100':
        dataset = ImageNet100(
            args.data_path, train=is_train,
            data_subset=os.path.join('./imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
        )
    elif args.data_set.lower() == 'imagenet1000':
        dataset = ImageNet1000(args.data_path, train=is_train)
    elif args.data_set.lower() == 'domainnet':
        dataset = DomainnetDataset(args.data_path, train=is_train, download=True)
    elif args.data_set.lower() == '5datasets':
        dataset = FiveDatasetsDataset(args.data_path, train=is_train, download=True)
    else:
        raise ValueError(f'Unknown dataset {args.data_set}.')

    scenario = ClassIncremental(
        dataset,
        initial_increment=args.initial_increment,
        increment=args.increment,
        transformations=transform.transforms,
        class_order=args.class_order
    )
    nb_classes = scenario.nb_classes

    return scenario, nb_classes


def build_transform(is_train, args):
    if args.aa == 'none':
        args.aa = None

    with warnings.catch_warnings():
        resize_im = args.input_size > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)

            if args.input_size == 32 and args.data_set == 'CIFAR':
                transform.transforms[-1] = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            return transform

        t = []
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                transforms.Resize(size, interpolation=interpolation),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        if args.input_size == 32 and args.data_set == 'CIFAR':
            # Normalization values for CIFAR100
            t.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        else:
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

        return transforms.Compose(t)
