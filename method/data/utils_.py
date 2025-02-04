import random

import numpy as np
import torchvision
from PIL import Image, ImageFile, ImageOps
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def square_crop_with_mask(img, mask, random=False, scale=None, min_padding=200, max_padding=None):
    # Calculate dimensions to get the largest possible square
    # width, height = img.size
    # if width != height:
    #     img = square_crop_shortest_side(img)
    width, height = img.size
    max_dim = max(width, height)
    # bbox = np.array([0,0,max_dim,max_dim])
    
    bbox = mask_to_bbox(np.array(mask)/255, square=False)
    if bbox is None:
        return img, Image.new('L', color = 255, size=img.size), 1.0

    if random:
        newbbox = _jitter_bbox(bbox, width, height, min_padding=min_padding, max_padding=max_padding) #max_dim, max_dim)

        if newbbox[2] - newbbox[0]  > width or newbbox[3] - newbbox[1] > height:
            print(newbbox, width, bbox)
        img_cropped = transforms.functional.crop(
                img,
                top=newbbox[1],
                left=newbbox[0],
                height=newbbox[3] - newbbox[1],
                width=newbbox[2] - newbbox[0],
            )
        mask_cropped = transforms.functional.crop(
                    mask,
                    top=newbbox[1],
                    left=newbbox[0],
                    height=newbbox[3] - newbbox[1],
                    width=newbbox[2] - newbbox[0],
                )
    else:
        img_cropped = img
        mask_cropped = mask
    return img_cropped, mask_cropped, scale


def _jitter_bbox(bbox, w, h, min_padding=200, max_padding=None):
    if max_padding is not None:
        ul0 = np.random.randint(max(0, bbox[0] - max_padding), max(1, bbox[0] - min_padding))
        ul1 = np.random.randint(max(0, bbox[1] - max_padding), max(1, bbox[1] - min_padding))
    else:
        ul0 = np.random.randint(0, max(1, bbox[0] - min_padding))
        ul1 = np.random.randint(0, max(1, bbox[1] - min_padding))

    if max_padding is not None:
        lr0 = np.random.randint(min(bbox[2] + min_padding, w-1), min(w, bbox[2] + max_padding))
        lr1 = np.random.randint(min(bbox[3] + min_padding, h-1), min(h, bbox[3] + max_padding))
    else:
        lr0 = np.random.randint(min(bbox[2] + min_padding, w-1), w)
        lr1 = np.random.randint(min(bbox[3] + min_padding, h-1), h)


    bbox = np.array([ul0, ul1, lr0, lr1])
    center = ((bbox[:2] + bbox[2:]) / 2).round().astype(int)
    extents = (bbox[2:] - bbox[:2]) / 2
    s = min( lr0 - ul0, lr1 - ul1) // 2
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=np.float32,
    )
    
    return square_bbox


def square_bbox(bbox, padding=0.0, astype=None):
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = ((bbox[:2] + bbox[2:]) / 2).round().astype(int)
    extents = (bbox[2:] - bbox[:2]) / 2
    s = (max(extents) * (1 + padding)).round().astype(int)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )

    return square_bbox


def mask_to_bbox(mask, square=False):
    """
    xyxy format
    """

    mask = mask > 0.1
    if not np.any(mask):
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    if cmax <= cmin or rmax <= rmin:
        return None
    bbox = np.array([int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1])
    if square: 
        bbox = square_bbox(bbox.astype(np.float32))
    return bbox


def crop_by_mask(img, mask):
    width, height = img.size
    max_dim = max(width, height)
    # bbox = np.array([0,0,max_dim,max_dim])
    maskarray = np.array(mask)/255
    bbox = mask_to_bbox(maskarray, square=True)
    # print(bbox, maskarray.min(), maskarray.max())
    if bbox is None:
        return img, Image.new('L', color = 255, size=img.size)
    if not np.any(bbox - np.array([0,0,max_dim,max_dim])):
        mask = ImageOps.invert(mask)
        bbox = mask_to_bbox(np.array(mask)/255, square=True)

    image_crop = transforms.functional.crop(
                img,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
    mask_crop = transforms.functional.crop(
                mask,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
    return image_crop, mask_crop


def random_crop_to_mask(image, mask, scale_factor=None, min_percent=15, max_percent=30):
    """
    Crop the image around the mask where the mask occupies a random percentage 
    between min_percent and max_percent of the image area.

    Args:
    image (PIL.Image): The original image to be cropped.
    mask (PIL.Image): A binary mask image (same size as the original image).
    min_percent (int): Minimum percentage of the area to be occupied by the mask.
    max_percent (int): Maximum percentage of the area to be occupied by the mask.

    Returns:
    PIL.Image: Cropped image around the mask.
    """
    # Convert mask to a numpy array
    mask_array = np.array(mask)

    # Find all non-zero points (where the mask is)
    rows, cols = np.where(mask_array > 0)
    if not len(rows) or not len(cols):
        raise ValueError("The mask does not contain any positive values.")

    # Get bounding box coordinates for the mask
    top, left = np.min(rows), np.min(cols)
    bottom, right = np.max(rows), np.max(cols)

    # Calculate height and width of the mask bounding box
    height, width = bottom - top, right - left
    img_width, img_height = mask.size

    # Calculate the scaling factor to randomly adjust the size of the crop
    if scale_factor is None:
        scale_factor = random.uniform(min_percent / 100, max_percent / 100)
    new_height = int(height / scale_factor)
    new_width = int(width / scale_factor)

    # Calculate new bounding box dimensions
    new_top = max(0, top - (new_height - height) // 2)
    new_left = max(0, left - (new_width - width) // 2)
    new_bottom = min(img_height, new_top + new_height)
    new_right = min(img_width, new_left + new_width)

    # Crop the image
    cropped_image = image.crop((new_left, new_top, new_right, new_bottom))

    return cropped_image, scale_factor


def square_crop(img):
    # Calculate dimensions to get the largest possible square
    width, height = img.size
    max_dim = max(width, height)

    img_cropped = Image.new('RGB', (max_dim, max_dim))
    x_offset = (max_dim - width) // 2
    y_offset = (max_dim - height) // 2
    img_cropped.paste(img, (x_offset, y_offset))
    return img_cropped


def square_crop_shortest_side(img):
    # Calculate dimensions to get the largest possible square
    width, height = img.size
    max_dim = min(width, height)

    # Calculate new bounding box dimensions
    top = left = 0
    new_top = max(0, top - (max_dim - height) // 2)
    new_left = max(0, left - (max_dim - width) // 2)
    new_bottom = min(height, new_top + max_dim)
    new_right = min(width, new_left + max_dim)
    img = img.crop((new_left, new_top, new_right, new_bottom))
    return img



def shortest_side_resize(image, size):
    width, height = image.size
    min_dim = min(width, height)

    scale = size / min_dim
    new_height = int(height * scale)
    new_width = int(width * scale)

    image = image.resize((new_width, new_height))
    return image


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def get_video_metadata(video_path):
    reader = torchvision.io.VideoReader(video_path)
    try:
        meta = reader.get_metadata()
        assert len(meta["video"]["fps"]) == 1
        assert len(meta["video"]["duration"]) == 1
        duration = meta["video"]["duration"][0]
        fps = meta["video"]["fps"][0]
    except:
        duration=10
        fps=1
    return duration, fps

def read_video(video_path, start_sec = 0, end_sec = 2):
   video = EncodedVideo.from_path(video_path)
   clip = video.get_clip(start_sec, end_sec)
   video_frames = clip["video"]
   return video_frames


def getw2cpy(transforms):
    x_vector = transforms['x']
    y_vector = transforms['y']
    z_vector = transforms['z']
    origin = transforms['origin']
    fov_select = transforms["x_fov"]

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

def cartesian_to_spherical(xyz):
    xy = xyz[:,0]**2 + xyz[:,2]**2
    z = np.sqrt(xy + xyz[:,1]**2)
    theta = np.arctan2(xyz[:,1], np.sqrt(xy)) # for elevation angle defined from Z-axis down
    azimuth = np.arctan2(xyz[:,2], xyz[:,0])
    return np.rad2deg(theta), np.rad2deg(azimuth), z
