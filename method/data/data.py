import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch
from data.utils_ import crop_by_mask, square_crop_shortest_side, square_crop_with_mask
from PIL import Image, ImageFile, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SynCDDataset(Dataset):
    def __init__(
        self,
        rootdir=None,
        mode='rigid',
        numref=3,
        drop_im=0.0,
        drop_txt=0.0,
        filter_aesthetics=0.,
        filter_dino=0.,
        img_size=1024,
        random_crop=False,
        dilate_mask=False,
        repeat=1,
        train=True,
        crop_global=False,
        drop_both=0.0,
        random_permute=True,
        default_dir=None,
        kernel_size=3,
        drop_mask=0.0,
        high_quality=False,
        transform=None,
        cropped_image=False,
        **kwargs
    ):
        self.rootdir = rootdir
        self.numref = numref
        self.img_size = img_size
        self.drop_im = drop_im
        self.drop_txt = drop_txt
        self.drop_both = drop_both
        self.drop_mask = drop_mask
        self.random_crop = random_crop
        self.dilate_mask = dilate_mask
        self.train = train
        self.crop_global = crop_global
        self.filter_dino = filter_dino
        self.filter_aesthetics = filter_aesthetics
        self.train = train
        self.repeat = repeat
        self.random_permute = random_permute
        self.default_dir = default_dir
        self.kernel_size = kernel_size
        self.high_quality = high_quality
        self.transform = transform
        self.cropped_image = cropped_image
        self.name = mode

        if self.transform is None:
            self.transform = transforms.Compose(
                    [
                        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
                        transforms.CenterCrop(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                )
        self.transform_mask = transforms.Compose(
                [
                    transforms.Resize(img_size // 8, interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.CenterCrop(img_size // 8),
                    transforms.ToTensor(),
                ]
            )
        self.transform_refimages = transforms.Compose(
                [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

        self.kernel_tensor = torch.ones((1, 1, kernel_size, kernel_size))
        self.setup_metadata()

    def __getitem__(self, index):
        batch = self.getdata(index)
        drop_both = False
        drop_im = False
        drop_txt = False
        if self.train:
            chance = np.random.uniform(0, 1)
            drop_both = chance < self.drop_both
            drop_im = chance >= self.drop_both and chance < self.drop_both + self.drop_im
            drop_txt = chance >= self.drop_both + self.drop_im and chance < self.drop_both + self.drop_im + self.drop_txt

        batch["prompts"] = ["" for _ in range(len(batch["prompts"]))] if (drop_txt or drop_both) else batch["prompts"]
        batch["masks"] = torch.ones_like(batch["masks"]) if np.random.uniform(0, 1) < self.drop_mask else batch["masks"]
        batch["ref_images"] = torch.zeros_like(batch["ref_images"]) if (drop_im or drop_both) else batch["ref_images"]
        batch["drop_im"] = torch.Tensor([drop_im*1.])
        return batch

    @staticmethod
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        keys = list(batch[0].keys())
        collate_batch = {x: [] for x in keys}

        keys_as_list = ['prompts', 'filenames', 'name']
        for batch_obj in batch:
            for key in collate_batch.keys():
                collate_batch[key].append(batch_obj[key])
        for key in collate_batch.keys():
            if key in keys_as_list:
                collate_batch[key] = [item for sublist in collate_batch[key] for item in sublist]
            else:
                collate_batch[key] = torch.cat(collate_batch[key], dim=0)

        return collate_batch

    def setup_metadata(self,):
        self.images = defaultdict(list)
        self.prompts = defaultdict(list)
        self.metadata = []
        self.metadata_files = []
        counter = 0
        for DIR in self.rootdir:
            rootname = str(Path(DIR).stem)
            rootrootname = str(Path(DIR).parent)
            if self.filter_aesthetics >= 0 and self.filter_dino >= 0:
                assert os.path.exists(f'{rootrootname}/{rootname}_aesthetics.pt')
                aesthetics_scores = torch.load(f'{rootrootname}/{rootname}_aesthetics.pt')
                dino_scores = torch.load(f'{rootrootname}/{rootname}_dino.pt')

            metadata_files = glob.glob(f'{DIR}/*metadata*.json')
            for metadata_file in metadata_files:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    for item in metadata:
                        valid_clique = [1] * len(item['filenames'])
                        if self.filter_dino > 0 and self.filter_aesthetics > 0:
                            dino_score = dino_scores['+'.join([Path(x).stem for x in item['filenames']])]
                            valid_clique = [aesthetics_scores[Path(x).stem] >= self.filter_aesthetics for x in item['filenames']]
                            valid_indices = [list(set(x).union(set(y))) for (x, y) in [((dino_score >= self.filter_dino) * (dino_score <= 0.98)).nonzero()]][0]
                            valid_clique = [x * (i in valid_indices) for i, x in enumerate(valid_clique)]

                        if np.sum(valid_clique) > 1:
                            self.metadata.append(counter)
                            self.images[counter] = [item['filenames'][i] for i in range(len(item['filenames'])) if valid_clique[i] > 0]
                            self.prompts[counter] = [item['prompts'][i] for i in range(len(item['filenames'])) if valid_clique[i] > 0]
                            counter += 1
                            self.metadata_files.append(item)

        random.shuffle(self.metadata)

    def __len__(self):
        return len(self.metadata) * self.repeat

    def getdata(self, index):
        index = index % len(self.metadata)
        batch = {'images': [],
                 'ref_images': [],
                 'prompts': [],
                 'filenames': [],
                 'masks': [],
                 'original_size_as_tuple': []
                 }

        fn = lambda x: 255 if x > 0 else 0

        num_images = len(self.images[self.metadata[index]])

        if self.numref == -1:
            indices_selected = list(np.arange(num_images))
        elif num_images >= self.numref:
            # sufficient images, so randomly sample from the indices
            indices_selected = list(np.random.choice(np.arange(0, num_images), self.numref, replace=False))
        else:
            # insufficient images, resample some indices with horizontal flip
            values = list(np.arange(0, num_images)) + list(-1*np.arange(1, num_images+1))
            indices_selected = list(np.random.choice(values, self.numref, replace=False))

        global_ref_index = [indices_selected.index(np.random.choice([num for num in indices_selected if num not in [x, -(x+1), -x-1]])) for x in indices_selected][:1] + np.arange(1, len(indices_selected)).tolist()

        for i in indices_selected:
            if i < 0:
                i = -1*i - 1
                flip = True
            else:
                flip = False
            filename = self.images[self.metadata[index]][i]
            image = Image.open(filename).convert('RGB')
            parentfolder = str(Path(filename).parent)
            imagestem = str(Path(filename).stem)
            mask = ImageOps.grayscale(Image.open(f'{parentfolder}/masks/{imagestem}.jpg')).convert('L').point(fn, mode='1').resize(image.size)
            mask = Image.fromarray((np.array(mask) * 255).astype(np.uint8))
            prompt = self.prompts[self.metadata[index]][i]

            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            ref_image, _, _ = square_crop_with_mask(image, mask, random=True)

            if self.cropped_image:
                image, mask2 = crop_by_mask(image, mask)
                mask2 = mask2.filter(ImageFilter.ModeFilter(size=20))
            elif self.random_crop:
                image, mask, scale = square_crop_with_mask(image, mask, random=True)
            else:
                image = square_crop_shortest_side(image)
                mask = square_crop_shortest_side(mask)

            orig_h, orig_w = image.size
            image = self.transform(image)[None]
            ref_image = self.transform_refimages(ref_image)[None]
            mask = self.transform_mask(mask)[None]
            if self.dilate_mask:
                mask = torch.clamp(torch.nn.functional.conv2d(mask.float(), self.kernel_tensor, padding='same'), 0, 1)

            batch['images'].append(image)
            batch['ref_images'].append(ref_image)
            batch['masks'].append(mask)
            batch['prompts'].append(prompt)
            batch['filenames'].append(filename)
            batch['original_size_as_tuple'].append(torch.tensor([orig_h, orig_w]))

        batch['images'] = torch.cat(batch['images'])
        batch['ref_images'] = torch.cat(batch['ref_images'])[global_ref_index] if self.random_permute else torch.cat(batch['ref_images'])
        batch['masks'] = torch.cat(batch['masks'])
        batch['original_size_as_tuple'] = torch.stack(batch['original_size_as_tuple'])
        batch['crop_coords_top_left'] = torch.zeros_like(batch['original_size_as_tuple'])
        batch['target_size_as_tuple'] = self.img_size*torch.ones_like(batch['original_size_as_tuple'])
        batch['name'] = [self.metadata[index]]

        return batch


class DummyDataset(SynCDDataset):
    def __init__(self, image_paths, prompt, num_images_per_prompt, cat, **kwargs):
        super().__init__(**kwargs)
        self.images = image_paths
        self.prompt = prompt
        self.num_images_per_prompt = num_images_per_prompt
        self.cat = cat

    def setup_metadata(self):
        pass

    def __len__(self):
        return self.num_images_per_prompt

    def getdata(self, index):
        batch = {'images': [],
                 'ref_images': [],
                 'prompts': [],
                 'filenames': [],
                 'masks': [],
                 'original_size_as_tuple': []
                 }

        fn = lambda x : 255 if x > 0 else 0

        for i in range(len(self.images)+1):
            if i == 0:
                # a random image for global feature injection
                filename = self.images[np.random.choice(np.arange(0, len(self.images)))]
            else:
                filename = self.images[i-1]
            image = Image.open(filename).convert('RGB')
            parentfolder = str(Path(filename).parent)
            imagestem = str(Path(filename).stem)
            mask = ImageOps.grayscale(Image.open(f'{parentfolder}/masks/{imagestem}.jpg')).convert('L').point(fn, mode='1').resize(image.size)
            mask = Image.fromarray((np.array(mask) * 255).astype(np.uint8))

            image, mask, _ = square_crop_with_mask(image, mask, random=True)
            ref_image = image

            orig_h, orig_w = image.size
            image = self.transform(image)[None]
            ref_image = self.transform_refimages(ref_image)[None]
            mask = self.transform_mask(mask)[None]
            mask = torch.clamp(torch.nn.functional.conv2d(mask.float(), self.kernel_tensor, padding='same'), 0, 1)

            batch['images'].append(image)
            batch['ref_images'].append(ref_image)
            batch['masks'].append(mask)
            batch['prompts'].append(self.prompt if i == 0 else f"photo of a {self.cat}")
            batch['filenames'].append(filename)
            batch['original_size_as_tuple'].append(torch.tensor([orig_h, orig_w]))

        batch['images'] = torch.cat(batch['images'])
        batch['ref_images'] = torch.cat(batch['ref_images'])
        batch['masks'] = torch.cat(batch['masks'])
        batch['original_size_as_tuple'] = torch.stack(batch['original_size_as_tuple'])
        batch['crop_coords_top_left'] = torch.zeros_like(batch['original_size_as_tuple'])
        batch['target_size_as_tuple'] = self.img_size*torch.ones_like(batch['original_size_as_tuple'])
        return batch


class ConcatDataset(Dataset):
    def __init__(
        self,
        mode='rigid',
        rootdir=None,
        **kwargs
    ):
        self.dataset = []
        self.length = 0
        self.split = []
        for i, mode_ in enumerate(mode.split('+')):
            self.dataset.append(SynCDDataset(rootdir=rootdir[i], mode=mode_, **kwargs))
            self.length += len(self.dataset[-1])
            self.split.append(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        whichdata = next(x[0] for x in enumerate(self.split) if x[1] > index)
        return self.dataset[whichdata].__getitem__(index - self.split[whichdata-1] if whichdata > 0 else index)


class CustomLoader(L.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers=2,
        shuffle=True,
        regularization=0.,
        mode='mv',
        **kwargs
    ):
        super().__init__()

        data_class = ConcatDataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.collate_fn = SynCDDataset.collate_fn
        self.train_dataset = data_class(mode=mode,
                                        regularization=regularization,
                                        train=True,
                                        **kwargs)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )
