import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

PHRASES_TO_REMOVE = "worn; low-poly; secluded; rustic; figurine; distinctive; retro-style; quirky; retro; nostalgic; historic; peaceful; wondrous; mystical; in a magical kingdom, ; fantastical; majestic; vibrant; tranquil; cartoon; industrial-chic; traditional; chic; vintage-inspired; ornate; rendering; cozy; flying in the air; picturesque; futuristic; high-tech; fantasy world; from the future; quaint; vintage; low poly number; legendary; striking; serene; bold and graphic; stylish; lush; exotic; trendy; modern; minimalist; stunning; iconic; neon; antique; charming; impressive; whimsical; bustling; gleaming"


def cartesian_to_spherical(xyz):
    xy = xyz[:,0]**2 + xyz[:,2]**2
    z = np.sqrt(xy + xyz[:,1]**2)
    theta = np.arctan2(xyz[:,1], np.sqrt(xy)) # for elevation angle defined from Z-axis down
    azimuth = np.arctan2(xyz[:,2], xyz[:,0])
    return np.rad2deg(theta), np.rad2deg(azimuth), z


def getw2cpy(transforms):
    x_vector = transforms['x']
    y_vector = transforms['y']
    z_vector = transforms['z']
    origin = transforms['origin']

    rotation_matrix = np.array([x_vector, y_vector, z_vector]).T

    translation_vector = np.array(origin)

    rt_matrix = np.eye(4)
    rt_matrix[:3, :3] = rotation_matrix
    rt_matrix[:3, 3] = translation_vector

    R, T = rotation_matrix, translation_vector

    st = np.array([[-1., 0., 0.],[0., 0., 1.],[0., -1, 0.]])
    st1 = np.array([[-1., 0., 0.],[0., -1., 0.],[0., 0, 1.]])

    R = (st@R@st1.T)
    T = st@T
    rt_matrix_py = np.eye(4)
    rt_matrix_py[:3, :3] = R
    rt_matrix_py[:3, 3] = T
    w2c_py = np.linalg.inv(rt_matrix_py)

    return w2c_py


def clean_prompt(prompt):
    for each in PHRASES_TO_REMOVE.split(';'):
        prompt = prompt.replace(each, '')
    return prompt


def get_depth(imagename, scale, shift):
    namestem = Path(imagename).stem
    nameparent = Path(imagename).parent
    image = scale_shift( f'{nameparent}/{namestem}_depth.png', scale, shift, depth=True)[0]
    if image is None:
        return None
    image = torch.from_numpy(np.array(image)).unsqueeze(0).unsqueeze(1)

    depth_map = 1/image
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def shift_scale_correspondences(correspondences, scale_factor1, shift_position1, scale_factor2, shift_position2):
    # Initialize new correspondences array
    new_correspondences = torch.zeros_like(correspondences)
    
    # Shift and scale each pair of correspondences
    new_correspondences[:, 0] = scale_factor1 * correspondences[:, 0] + ((1 - 2*(shift_position1[1] / 1024)) - scale_factor1)
    new_correspondences[:, 1] = scale_factor1 * correspondences[:, 1] + ((1 - 2*(shift_position1[0] / 1024)) - scale_factor1)
    
    # Second image coordinates (shifted and scaled)
    new_correspondences[:, 2] = scale_factor2 * correspondences[:, 2] + ((1 - 2*(shift_position2[1] / 1024)) - scale_factor2)
    new_correspondences[:, 3] = scale_factor2 * correspondences[:, 3] + ((1 - 2*(shift_position2[0] / 1024)) - scale_factor2)
    
    return new_correspondences


# random scale and shift the rendered image
def scale_shift(path, scale=None, shift=None, depth=False):
    if not os.path.exists(path):
        return [None]*4
    if depth:
        original_image = Image.open(path).resize((1024, 1024))
        mask = original_image.convert('L')
        background_image = Image.new("I", (original_image.width, original_image.height), 65535 )
    else:
        original_image = Image.open(path).convert('RGBA').resize((1024, 1024))
        mask = (original_image.split()[-1]).convert('L')
        original_image = Image.open(path).convert('RGB').resize((1024, 1024))
        background_color = (0, 0, 0)  # White color in RGB
        background_image = Image.new("RGB", (original_image.width, original_image.height), background_color)

    # If the image is too small, ignore it. Otherwise define the scaling factor (e.g., scale by 50%)
    area = (np.array(mask)/255.).mean()
    if area < 0.1:
        return [None]*4
    if area < 0.2:
        min_scale = 0.9
    else:
        min_scale = 0.7
    scale = np.random.uniform(min_scale, 0.95) if scale is None else scale
    scaled_width = int(original_image.width * scale)
    scaled_height = int(original_image.height * scale)

    # Resize the image
    scaled_image = original_image.resize((scaled_width, scaled_height))
    scaled_mask = mask.resize((scaled_width, scaled_height))

    # Create a new white background image
    background_mask = Image.new("L", (original_image.width, original_image.height), 0)

    # Paste the scaled image onto the background at a random location
    if shift is not None:
        paste_position = shift
    else:
        paste_position = (
            random.randint(0, background_image.width - scaled_image.width),
            random.randint(max(background_image.height - scaled_image.height - 100, 0), background_image.height - scaled_image.height),
        )
    background_image.paste(scaled_image, paste_position)
    background_mask.paste(scaled_mask, paste_position)
    return background_image, background_mask, scale, paste_position


def is_elev_inrange(imagenames, min=0, max=60):
    json_files = [x.replace('.png', '.json') for x in imagenames]

    w2c_pys = []
    for each in json_files:
        with open(each, 'r') as f:
            transforms = json.load(f)
        w2c_py = getw2cpy(transforms)
        w2c_pys.append(w2c_py)

    w2c_pys = np.stack(w2c_pys)
        
    camera_center = torch.einsum('bhw,bwc->bhc', -1*torch.from_numpy(w2c_pys[:, :3, :3]).permute(0,2,1), torch.from_numpy(w2c_pys[:, :3, 3]).reshape(-1, 3, 1))
    elev, _, _ = cartesian_to_spherical(camera_center)
    if torch.all(elev > min) and torch.all(elev < max):
        return True
    else:
        return False


def get_minimum_k_overlap(corresp, imagenames, k=0):
    if len(corresp) < k:
        return None, None
    matches = [(x[0].shape[0] + x[1].shape[0] + x[2].shape[0], i) for i,x in enumerate(corresp) if is_elev_inrange(imagenames[i])] # and x.shape[0] < 70000 ]
    if len(matches) - 1 < k:
        return None, None
    matches = sorted(matches, key=lambda x: x[0], reverse=False)
    index = [x[1] for x in matches][k]
    return corresp[index], imagenames[index]


def remove_unreadable_files(corresp, imagenames):
    valid_i = []
    for i in range(len(imagenames)):
        try:
            json_files = [x.replace('.png', '.json') for x in imagenames[i]]
            _ =  [json.load(open(x, 'r')) for x in json_files]
            valid_i.append(i)
        except:
            continue
    return [corresp[x] for x in valid_i], [imagenames[x] for x in valid_i]


class ObjaverseDataset(Dataset):
    def __init__(self, objects_ids, rootdir, promptpath, warping=True, img_size=1024, repeat_each_id=2):
        self.objects_ids = objects_ids
        self.rootdir = rootdir
        self.num = 3
        self.prompts = torch.load(promptpath)
        self.warping = warping

        # maximum times the same asset can be used (in pratice it can be less if we didn't get triplets with sufficient correspondence in get_corresp.py)
        self.repeat_each_id=repeat_each_id 

        self.transform = transforms.Compose(
                [
                    transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.ToTensor()
                ]
            )
        self.transform_mask = transforms.Compose(
                [
                    transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.ToTensor(),
                ]
            )
    
    def _pointlambda_(self, x):
        if x > 0:
            return 255
        else:
            return 0
    
    def __getitem__(self, index):
        try:
            i = index //  self.repeat_each_id
            k = index % self.repeat_each_id

            uid = self.objects_ids[i].split("/")[-1].split(".glb")[0]
            if self.warping:
                corresp, imagenames = torch.load(f"{self.rootdir}/correspondence/{uid}.pt", map_location='cpu')
                
                # make sure imagenames is correctly formatted with parent dir
                for j in range(len(imagenames)):
                    for m in range(len(imagenames[j])):
                        base = imagenames[j][m].split('/')[-1]
                        stem = Path(imagenames[0][0]).parent.stem
                        imagenames[j][m] = f"{self.rootdir}/objaverse_rendering/{stem}/{base}"

                corresp, imagenames = remove_unreadable_files(corresp, imagenames)

                # check if we have kth valid triplet and corresponding prompt.
                corresp, imagenames = get_minimum_k_overlap(corresp, imagenames, k=k)
            else:
                corresp = None
                imagenames = [x for x in os.listdir(f"{self.rootdir}/objaverse_rendering/{uid}/") if x.endswith('.png') ]
                # check if filepath stem if of the form f'{0:5d}.png'
                imagenames = [f"{self.rootdir}/objaverse_rendering/{uid}/{x}" for x in imagenames if x.split('.')[0].isdigit()]
                imagenames = [[imagenames[i+x] for x in range(self.num)] for i in np.arange(0, len(imagenames)-self.num, self.num)]
                if len(imagenames) < k:
                    return {}
                else:
                    imagenames = imagenames[k]
            possible_prompts = self.prompts[uid]
            if (corresp is None and self.warping) or len(possible_prompts) < self.num * (k + 1):
                return {}
            
            possible_prompts = possible_prompts[min(len(possible_prompts)-self.num, self.num*k): ]
            images = []
            masks = []
            depths = []
            prompts = []
            scales = []
            shifts = []

            for counter in range(len(imagenames)): 
                orig_image, mask, scale, shift = scale_shift(imagenames[counter])
                depth = get_depth(imagenames[counter], scale, shift)
                if orig_image is None or depth is None:
                    return {}
                mask = ImageOps.grayscale(orig_image.convert('L').point(self._pointlambda_, mode='1'))
                prompt = possible_prompts[counter % len(possible_prompts)].lower()
                prompt = f'{prompt}'
                prompt = clean_prompt(prompt)

                masks += [self.transform_mask(mask)]
                depths += [self.transform_mask(depth)]
                images += [self.transform(orig_image)]
                prompts += [prompt]
                scales.append(scale)
                shifts.append(shift)
                counter+=1

            batch = {
                'images': 2*torch.stack(images) -1.,
                'depths': torch.stack(depths),
                'masks': torch.cat(masks),
                'prompts': prompts,
                'imagenames': imagenames,
                'uid': [uid]
            }

            if self.warping:
                correspondences = []
                counter_cc = []
                for j, comb_ in enumerate(itertools.combinations(np.arange(self.num), 2)):
                    corresp_ = shift_scale_correspondences(corresp[j], scales[comb_[0]], shifts[comb_[0]], scales[comb_[1]], shifts[comb_[1]])
                    correspondences += [corresp_]
                    counter_cc.append(corresp_.shape[0])
                correspondences = torch.cat(correspondences)
                counter_cc = torch.tensor(counter_cc)
                batch.update({'correspondence': correspondences, 'counter_cc': counter_cc})
            return batch
        except:
            return {}

    def __len__(self):
        return len(self.objects_ids) * self.repeat_each_id

    @staticmethod
    def collate_fn(batch):
        """A function to collate the data across batches. This function must be passed to pytorch's DataLoader to collate batches.
        Args:
            batch(list): List of objects returned by this class' __getitem__ function. This is given by pytorch's dataloader that calls __getitem__
                         multiple times and expects a collated batch.
        Returns:
            dict: The collated dictionary representing the data in the batch.
        """
        result = {x: [] for x in list(batch[0].keys())}
        for batch_obj in batch:
            if len(batch_obj.keys()) <=0:
                return None
        for batch_obj in batch:
            for key in result.keys():
                result[key].append(batch_obj[key])
        for key in result.keys():
            if key in ['prompts', 'imagenames', 'uid']:
                result[key] = [item for sublist in result[key] for item in sublist]
            else:
                result[key] = torch.cat(result[key], dim=0)
        
        return result
