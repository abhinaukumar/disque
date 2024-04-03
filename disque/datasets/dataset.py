import os

import numpy as np
import random
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
from tonemaplib import get_tmoclass
from videolib import Frame, standards

from .iqa_distortions import *
from ..utils.png import read_png, write_png



class SDRSemiGridDataset(data.Dataset):
    def __init__(self, txt_path, base_dir='sdr_dataset', max_n_distortions=3, patch_size=160, mode='train'):
        super().__init__()
        self.base_dir = base_dir
        rng = np.random.default_rng(0)
        with open(txt_path, 'rt') as f:
            self.image_name = f.read().splitlines()
        rng.shuffle(self.image_name)
        self._n_train = int(0.8*len(self.image_name))
        self._n_val = len(self.image_name) - self._n_train
        self.mode = mode
        if self.mode not in ['train', 'val']:
            raise ValueError(f'Invalid mode {self.mode}')
        if self.mode == 'train':
            self.image_name = self.image_name[:self._n_train]
        elif self.mode == 'val':
            self.image_name = self.image_name[self._n_train:]
        else:
            pass

        self.max_n_distortions = max_n_distortions
        self.patch_size = patch_size
        self._iqa_transformations = [
            imidentity,
            imblurgauss,
            imblurlens,
            imblurmotion,
            imcolordiffuse,
            imcolorshift,
            imcolorsaturate,
            imsaturate,
            imcompressjpeg,
            imnoisegauss,
            imnoisecolormap,
            imnoiseimpulse,
            imnoisemultiplicative,
            imdenoise,
            imbrighten,
            imdarken,
            immeanshift,
            imsharpenHi,
            imcontrastc,
            impixelate,
            imnoneccentricity,
            imjitter,
            imresizedist_nearest,
            imresizedist_bilinear,
            imresizedist_bicubic,
            imresizedist_lanczos,
        ]
        self.n_transforms = len(self._iqa_transformations)
        self.n_levels = 5
        self.crop_scale = (0.5, 1.0)
        self.crop_ratio = (3/4, 4/3)
        self.crop_transform = transforms.RandomResizedCrop((self.patch_size, self.patch_size), scale=self.crop_scale, ratio=self.crop_ratio, interpolation=transforms.InterpolationMode.BICUBIC)

    @property
    def n_img(self):
        return len(self.image_name)

    def __len__(self):
        return self.n_img

    def _apply_transformation(self, img, choices, levels):
        img_trans = img
        for choice, level in zip(choices, levels):
            img_trans = self._iqa_transformations[choice](img_trans, level)
        return img_trans

    def _read_till_success(self, id):
        while True:
            try:
                image = Image.open(os.path.join(self.base_dir, self.image_name[id])).convert('RGB')
                return image
            except OSError:
                id = random.randint(0, self.n_img-1)

    def __getitem__(self, idx) :
        id1 = idx
        id2 = random.randint(0, self.n_img-1)

        image1 = self._read_till_success(id1)
        image2 = self._read_till_success(id2)

        n_dist = random.randint(1, self.max_n_distortions)
        choices = [random.randint(0, self.n_transforms-1) for _ in range(n_dist)]
        levels = [random.randint(0, self.n_levels-1) for _ in range(n_dist)]

        image1_1 = transforms.ToTensor()(image1)
        image2_1 = transforms.ToTensor()(image2)

        image1_2 = transforms.ToTensor()(self._apply_transformation(image1, choices, levels))
        image2_2 = transforms.ToTensor()(self._apply_transformation(image2, choices, levels))

        image1_crop_params = transforms.RandomResizedCrop.get_params(image1_1, self.crop_scale, self.crop_ratio)
        image2_crop_params = transforms.RandomResizedCrop.get_params(image2_1, self.crop_scale, self.crop_ratio)

        image1_1 = F.resized_crop(image1_1, *image1_crop_params, (self.patch_size, self.patch_size), interpolation=transforms.InterpolationMode.BICUBIC)
        image1_2 = F.resized_crop(image1_2, *image1_crop_params, (self.patch_size, self.patch_size), interpolation=transforms.InterpolationMode.BICUBIC)
        image2_1 = F.resized_crop(image2_1, *image2_crop_params, (self.patch_size, self.patch_size), interpolation=transforms.InterpolationMode.BICUBIC)
        image2_2 = F.resized_crop(image2_2, *image2_crop_params, (self.patch_size, self.patch_size), interpolation=transforms.InterpolationMode.BICUBIC)

        return image1_1, image1_2, image2_1, image2_2


class HDRSemiGridDataset(data.Dataset):
    def __init__(self, txt_path, base_dir='hdr_dataset', base_tm_dir=None, patch_size=160, mode='train'):  # Do not add '/' at the end of the directory names
        super().__init__()
        self.base_dir = base_dir
        self.base_tm_dir = base_tm_dir if base_tm_dir else base_dir + '_tonemapped'
        rng = np.random.default_rng(0)
        with open(txt_path, 'rt') as f:
            self.image_name = f.read().splitlines()
        rng.shuffle(self.image_name)
        self._n_train = int(0.8*len(self.image_name))
        self._n_val = len(self.image_name) - self._n_train
        self.mode = mode
        if self.mode not in ['train', 'val']:
            raise ValueError(f'Invalid mode {self.mode}')
        if self.mode == 'train':
            self.image_name = self.image_name[:self._n_train]
        elif self.mode == 'val':
            self.image_name = self.image_name[self._n_train:]
        else:
            pass

        # Either all should end in '/' or none should.
        # self._hdr_image_base_dir = 'hdr_dataset'
        # self._tm_image_base_dir = 'hdr_dataset_tonemapped'

        self._tmo_names = [
            'Durand02',
            'Eilertsen15',
            'Hable',
            'ITU21',
            'Oskarsson17',
            'Rana19',
            'Reinhard02',
            'Reinhard12',
            'Shan12',
            'Yang21'
        ]

        self._tmo_arg_names = {
            'Durand02': 'base_contrast',
            'Eilertsen15': 'bin_width',
            'Hable': 'desat',
            'ITU21': 'peak_hdr',
            'Oskarsson17': 'num_clusters',
            'Rana19': 'desat',
            'Reinhard02': 'desat',
            'Reinhard12': 'viewing_cond',
            'Shan12': 'levels',
            'Yang21': 'desat'
        }

        self._tmo_arg_values = {
            'Durand02': [10, 100, 1000, 10000],
            'Eilertsen15': [0.01, 0.03, 0.1, 0.3],
            'Hable': [0.0, 0.25, 0.5, 0.75],
            'ITU21': [1e3, 1e4, 1e5, 1e6],
            'Oskarsson17': [8, 16, 32, 64],
            'Rana19': [0.0, 0.25, 0.5, 0.75],
            'Reinhard02': [0.0, 0.25, 0.5, 0.75],
            'Reinhard12': ['neutral', 'red', 'blue', 'green'],
            'Shan12': [1, 2, 3, 4],
            'Yang21': [0.0, 0.25, 0.5, 0.75],
        }

        self._tmo_arg_dict = {tmo_name: [{'video_mode': 'framewise', self._tmo_arg_names[tmo_name]: arg_value} for arg_value in self._tmo_arg_values[tmo_name]] for tmo_name in self._tmo_names}

        self._tmos = {}
        for tmo_name in self._tmo_names:
            if tmo_name in ['Rana19', 'Yang21']:
                continue
            TMOClass = get_tmoclass(tmo_name)
            self._tmos[tmo_name] = []
            for tmo_args in self._tmo_arg_dict[tmo_name]:
                self._tmos[tmo_name].append(TMOClass(**tmo_args))

        self._comp_levels = [70, 43, 36, 24, 7]
        self.n_comp_levels = len(self._comp_levels) + 1 # + 1 for lossless

        self.n_tmos = len(self._tmo_names)
        self.n_levels = len(self._tmo_arg_values[self._tmo_names[0]])
        self.crop_scale = (0.5, 1.0)
        self.crop_ratio = (3/4, 4/3)
        self.patch_size = patch_size

    @property
    def n_img(self):
        return len(self.image_name)

    def __len__(self):
        return self.n_img

    def _jpeg_compress(self, img, level):
        if level == 0:
            return img
        amount = self._comp_levels[level-1]
        im = Image.fromarray(img, mode='RGB')
        imgByteArr = io.BytesIO()
        im.save(imgByteArr, format='JPEG',quality=amount)
        im1 = Image.open(imgByteArr)
        return np.array(im1)

    def _apply_transformation(self, img, img_name, choice, level, compress_level):
        tmo_name = self._tmo_names[choice]
        out_dir = os.path.join(self.base_tm_dir, tmo_name, str(level))
        if self.base_dir[-1] == '/' and out_dir[-1] != '/':
            out_dir += '/'
        elif self.base_dir[-1] != '/' and out_dir[-1] == '/':
            out_dir = out_dir[:-1]

        out_path = os.path.join(out_dir, img_name)
        out_path = os.path.splitext(out_path)[0] + '.png'
        if os.path.isfile(out_path):
            try:
                out_img = read_png(out_path).astype('uint8')
            except RuntimeError:
                return None
        else:
            hdr_frame = Frame(standards.rec_2100_pq)
            hdr_frame.rgb = img.astype('float64')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                tmo = self._tmos[tmo_name][level]
            except KeyError:
                return None
            out_frame = tmo(hdr_frame)
            out_img = out_frame.rgb.astype('uint8')  # Quantize and dequantize to match reading images
            write_png(out_img, out_path)

        return self._jpeg_compress(out_img, compress_level).astype('float64')

    def _read_till_success(self, id):
        read_success = False
        while not read_success:
            img_name = self.image_name[id]
            hdr_image_path = os.path.join(self.base_dir, img_name)
            if os.path.isfile(hdr_image_path):
                try:
                    image = read_png(hdr_image_path).astype('float64')
                    return image, img_name
                except RuntimeError:
                    raise RuntimeError(hdr_image_path)
            else:
                print(f'{hdr_image_path} does not exist')
                id = random.randint(0, self.n_img-1)

    def __getitem__(self, idx) :
        id1 = idx
        id2 = random.randint(0, self.n_img-1)

        image1 = None
        image2 = None
        image1_trans = None
        image2_trans = None
        while any(map(lambda x: x is None, [image1, image1_trans, image2, image2_trans])):
            if image1 is None:
                image1, image1_name = self._read_till_success(id1)
            if image2 is None:
                image2, image2_name = self._read_till_success(id2)

            fail_count = -1
            max_fail_count = 20
            while any(map(lambda x: x is None, [image1_trans, image2_trans])):
                fail_count += 1
                if fail_count == max_fail_count:
                    image1 = None
                    image1_trans = None
                    image2 = None
                    image2_trans = None
                    id1 = random.randint(0, self.n_img-1)
                    id2 = random.randint(0, self.n_img-1)
                    break
                choice = random.randint(0, self.n_tmos-1)
                level = random.randint(0, self.n_levels-1)
                compress_level = random.randint(0, self.n_comp_levels-1)

                image1_trans = self._apply_transformation(image1, image1_name, choice, level, compress_level)
                image2_trans = self._apply_transformation(image2, image2_name, choice, level, compress_level)

        image1_1 = transforms.ToTensor()(image1 / 1023).float()
        image2_1 = transforms.ToTensor()(image2 / 1023).float()

        image1_2 = transforms.ToTensor()(image1_trans / 255).float()
        image2_2 = transforms.ToTensor()(image2_trans / 255).float()

        image1_crop_params = transforms.RandomResizedCrop.get_params(image1_1, self.crop_scale, self.crop_ratio)
        image2_crop_params = transforms.RandomResizedCrop.get_params(image2_1, self.crop_scale, self.crop_ratio)

        image1_1 = F.resized_crop(image1_1, *image1_crop_params, (self.patch_size, self.patch_size), interpolation=transforms.InterpolationMode.BICUBIC)
        image1_2 = F.resized_crop(image1_2, *image1_crop_params, (self.patch_size, self.patch_size), interpolation=transforms.InterpolationMode.BICUBIC)
        image2_1 = F.resized_crop(image2_1, *image2_crop_params, (self.patch_size, self.patch_size), interpolation=transforms.InterpolationMode.BICUBIC)
        image2_2 = F.resized_crop(image2_2, *image2_crop_params, (self.patch_size, self.patch_size), interpolation=transforms.InterpolationMode.BICUBIC)

        return image1_1, image1_2, image2_1, image2_2
