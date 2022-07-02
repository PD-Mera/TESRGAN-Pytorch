from xml.sax import default_parser_list
from matplotlib.pyplot import get
import torchvision.transforms.functional as F
import torch
import random
from os import listdir
from os.path import join
from torch.utils.data import Dataset
from skimage.feature import canny
from PIL import Image, ImageFilter
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize, InterpolationMode

# Normalization parameters for pre-trained PyTorch models
#mean = torch.tensor([0.485, 0.456, 0.406])
#std = torch.tensor([0.229, 0.224, 0.225])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class TrainDataset(Dataset):
    def __init__(self,
                 data_path,
                 mode,
                 lr_size):  # lr_size must be valid
        super(TrainDataset, self).__init__()
        w, h = lr_size
        self.thermal_HR_filenames = get_image(data_path=data_path,
                                              data_type_KAIST='lwir_HR',
                                              data_type_FLIR='thermal_HR',
                                              data_type_LLVIP='thermal_HR',
                                              mode=mode)
        self.thermal_LR_filenames = get_image(data_path=data_path,
                                              data_type_KAIST='lwir_LR',
                                              data_type_FLIR='thermal_LR',
                                              data_type_LLVIP='thermal_LR',
                                              mode=mode)
        self.rgb_filenames = get_image(data_path=data_path,
                                       data_type_KAIST='visible',
                                       data_type_FLIR='visible',
                                       data_type_LLVIP='visible',
                                       mode=mode)

        # self.lr_transform = Compose(
        #     [Resize(lr_size, interpolation=InterpolationMode.BICUBIC), ToTensor()])  # @Thuan: add normalize

        self.hr_transform = Compose([ToTensor()])
        self.lr_transform = Compose([Resize((w, h), interpolation=InterpolationMode.BICUBIC),ToTensor()])
        self.edge_downscale = Compose(
            [Resize((w*2,h*2), interpolation=InterpolationMode.BICUBIC)])  # @Thuan: add normalize
        self.edge_transform = Compose([ToTensor()])

    def __getitem__(self, index):

        lr_image = Image.open(self.thermal_LR_filenames[index])
        hr_image = Image.open(self.thermal_HR_filenames[index])
        if self.thermal_HR_filenames[index].endswith('jpeg'):
            lr_image = lr_image.convert('RGB')
        if self.thermal_HR_filenames[index].endswith('jpeg'):
            hr_image = hr_image.convert('RGB')
        rgb_image = Image.open(self.rgb_filenames[index])
        edge_map = self.edge_downscale(rgb_image)
        edge_map = edge_map.convert("L")  # Convert to  gray scale
        edge_map = edge_map.filter(ImageFilter.FIND_EDGES)
        edge_map = self.edge_transform(edge_map)
        lr_image = self.lr_transform(lr_image)
        target = self.hr_transform(hr_image)

        # edge_map = edge_map * 2 - 1
        # lr_image = lr_image * 2 - 1
        # target = target * 2 - 1
        return lr_image, edge_map, target

    def __len__(self):
        return len(self.thermal_HR_filenames)


def display_transform():
    return Compose([
        ToPILImage(),
        ToTensor()
    ])


BLUR_list = ['BLUR_1', 'BLUR_2', 'BLUR_3', 'BLUR_4']

# data_path = '/kaist-cvpr15/images'


def get_image_KAIST(data_path, data_type, mode, train_set=6, test_set=6, max_set=13):
    """
    data_type
    - 'lwir_LR' -> thermal image LR
    - 'lwir_HR' -> thermal image HR
    - 'visible' -> rgb image
    mode
    - 'train'
    - 'test'
    - 'test_all_set'
    """
    if mode == 'train':
        data_set = ["set{:02d}".format(i) for i in range(0, train_set)]
    elif mode == 'test':
        data_set = ["set{:02d}".format(i) for i in range(test_set, max_set)]

    data_list = []

    for BLUR_x in listdir(data_path):
        if BLUR_x in BLUR_list:
            for setxx in listdir(join(data_path, BLUR_x, 'KAIST')):
                if setxx in data_set:
                    for Vxxx in listdir(join(data_path, BLUR_x, 'KAIST', setxx)):
                        for x in listdir(join(data_path, BLUR_x, 'KAIST', setxx, Vxxx, data_type)):
                            if is_image_file(x):
                                data_list.append(
                                    join(data_path, BLUR_x, 'KAIST', setxx, Vxxx, data_type, x))
    return data_list


def get_image_FLIR_LLVIP(data_path, data_type, mode, dataset):
    """
    data_type
    - 'thermal_LR' -> thermal image LR
    - 'thermal_HR' -> thermal image HR
    - 'visible' -> rgb image
    mode
    - 'train'
    - 'test'
    dataset
    - 'FLIR'
    - 'LLVIP'
    """
    data_list = []
    for BLUR_x in listdir(data_path):
        if BLUR_x in BLUR_list:
            for x in listdir(join(data_path, BLUR_x, dataset, mode, data_type)):
                data_list.append(
                    join(data_path, BLUR_x, dataset, mode, data_type, x))
    return data_list


def get_image(data_path,
              data_type_KAIST,
              data_type_FLIR,
              data_type_LLVIP,
              mode='train',
              KAIST_train_set=6, KAIST_test_set=6, KAIST_max_set=13):
    '''
    MODE
    - 'train'
    - 'test'
    data_type_KAIST
    - 'lwir_LR' -> thermal image LR
    - 'lwir_HR' -> thermal image HR
    - 'visible' -> rgb image
    data_type_FLIR, data_type_LLVIP
    - 'thermal_LR' -> thermal image LR
    - 'thermal_HR' -> thermal image HR
    - 'visible' -> rgb image
    '''
    data_list = []
    data_list.extend(get_image_KAIST(data_path=data_path,
                                     data_type=data_type_KAIST,
                                     mode=mode,
                                     train_set=KAIST_train_set,
                                     test_set=KAIST_test_set,
                                     max_set=KAIST_max_set))
    data_list.extend(get_image_FLIR_LLVIP(data_path=data_path,
                                          data_type=data_type_FLIR,
                                          mode=mode,
                                          dataset='FLIR'))
    data_list.extend(get_image_FLIR_LLVIP(data_path=data_path,
                                          data_type=data_type_LLVIP,
                                          mode=mode,
                                          dataset='LLVIP'))
    return data_list


class Opt(object):
    lr = 0.00005
    n_critic = 5
    clip_value = 0.01
