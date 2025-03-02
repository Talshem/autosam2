import torch
import PIL
from PIL import Image, ImageEnhance
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
from datasets.tfs import get_transform
import cv2
from tqdm import tqdm


def _load_img_as_tensor(img_path, mask_path, image_size, vertical_flip_prob, horizontal_flip_prob, rotation_degree, scale_factor, brightness, contrast, saturation, hue):
    img_pil = Image.open(img_path)
    mask_pil = Image.open(mask_path)
    video_width, video_height = img_pil.size 

    mask_pil = mask_pil.convert("L")
    img_pil = img_pil.convert("RGB")

    if horizontal_flip_prob > 0.5 :
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

    if vertical_flip_prob > 0.5 :
        img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
        mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)

    img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
    img_pil = ImageEnhance.Color(img_pil).enhance(saturation)

    img_pil = img_pil.convert("HSV")
    np_img = np.array(img_pil)
    np_img[..., 0] = (np_img[..., 0].astype(int) + int(hue * 180)) % 180
    img_pil = Image.fromarray(np_img, "HSV").convert("RGB") 

    img_pil = img_pil.resize((int(image_size*scale_factor), int(image_size*scale_factor)))
    mask_pil = mask_pil.resize((int(image_size*scale_factor), int(image_size*scale_factor)))

    img_pil = img_pil.rotate(rotation_degree, expand=True)
    mask_pil = mask_pil.rotate(rotation_degree, expand=True)

    left = (img_pil.size[0] - image_size) // 2
    top = (img_pil.size[1] - image_size) // 2
    right = (img_pil.size[0] + image_size) // 2
    bottom = (img_pil.size[1] + image_size) // 2

    img_pil = img_pil.crop((left, top, right, bottom))

    left = (mask_pil.size[0] - image_size) // 2
    top = (mask_pil.size[1] - image_size) // 2
    right = (mask_pil.size[0] + image_size) // 2
    bottom = (mask_pil.size[1] + image_size) // 2

    mask_pil = mask_pil.crop((left, top, right, bottom))

    img_np = np.array(img_pil)
    if img_np.dtype == np.uint8: 
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")

    mask_np = np.array(mask_pil)
    if mask_np.dtype == np.uint8: 
        mask_np = mask_np / 255.0
        mask_np[mask_np > 0.5] = 1
        mask_np[mask_np <= 0.5] = 0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")

    img = torch.from_numpy(img_np).permute(2, 0, 1)
    mask = torch.from_numpy(mask_np).unsqueeze(0)

    return img, mask, video_height, video_width

def load_videos_from_jpg_images(
    video_path,
    mask_path,
    image_size,
    start_idx,
    sequence_length,
    train,
    offload_video_to_cpu=False,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
):
    """
    Load frames from a PyTorch video_path containing video frame directory.
    
    Parameters:
        video_path (torch.utils.data.DataLoader): video_path with video frames directory.
        image_size (int): Size to resize frames to (square resolution).
        offload_video_to_cpu (bool): If True, keeps video tensors on CPU.
        img_mean (tuple): Mean for normalization.
        img_std (tuple): Standard deviation for normalization.
        async_loading_frames (bool): If True, uses asynchronous frame loading.
        compute_device (torch.device): Device to load tensors to (CPU/GPU).
    
    Returns:
        images (torch.Tensor): images of video, shape (T, 3, H, W).
        video_heights (list): Height of original video.
        video_widths (list): Width of original video.
    """

    if not os.path.isdir(video_path):
        raise ValueError(f"Invalid directory: {video_path}")
            
    frame_names = sorted(
        [p for p in os.listdir(video_path) if p.lower().endswith(('.jpg', '.jpeg'))],
        key=lambda p: int(os.path.splitext(p)[0])
    )
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"No images found in {video_path}")

    img_paths = [os.path.join(video_path, fname) for fname in frame_names]
    mask_paths = [os.path.join(mask_path, fname) for fname in frame_names]

    valid_frames = tuple(f"/{i}.jpg" for i in range(start_idx, start_idx + sequence_length))

    images = torch.zeros(sequence_length, 3, image_size, image_size, dtype=torch.float32)
    masks = torch.zeros(sequence_length, 1, image_size, image_size, dtype=torch.float32)
  
    if train:
        vertical_flip_prob = random.random()
        horizontal_flip_prob = random.random()
        rotation_degree = random.uniform(-90, 90)
        scale_factor = random.uniform(0.75, 1.25)
        brightness = random.uniform(1 - 0.4, 1 + 0.4)
        contrast = random.uniform(1 - 0.4, 1 + 0.4)
        saturation = random.uniform(1 - 0.4, 1 + 0.4)
        hue = random.uniform(-0.1, 0.1)
    else:
        vertical_flip_prob = 0
        horizontal_flip_prob = 0
        rotation_degree = 0
        scale_factor = 1
        brightness = 1
        contrast = 1
        saturation = 1
        hue = 0
        
    i = 0
    for img_path, mask_path in zip(img_paths, mask_paths):
        if img_path.endswith(valid_frames):
            images[i], masks[i], video_height, video_width = _load_img_as_tensor(img_path, mask_path, image_size, vertical_flip_prob, horizontal_flip_prob, rotation_degree, scale_factor, brightness, contrast, saturation, hue)
            i += 1
            
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].to(compute_device)
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].to(compute_device)

    images -= img_mean
    images /= img_std

    return images, masks, (video_height, video_width)

import os
import torch
from torch.utils.data import Dataset
import random
import torch.nn.functional as F

class VideoDataset(Dataset):
    def __init__(self, root, image_size, transform=None, target_transform=None, train=False, sequence_length=8, loops=1):
        self.root = root
        if train:
            self.video_root = os.path.join(self.root, 'Training', 'images')
            self.masks_root = os.path.join(self.root, 'Training', 'masks')
        else:
            self.video_root = os.path.join(self.root, 'Test', 'images')
            self.masks_root = os.path.join(self.root, 'Test', 'masks')
        
        self.paths = os.listdir(self.video_root)
        self.transform = transform
        self.image_size = image_size
        self.loops = loops
        self.target_transform = target_transform
        self.train = train
        self.sequence_length = sequence_length

        self.index_mapping = []
        for video_name in self.paths:
            video_path = os.path.join(self.video_root, video_name)
            frames = sorted(os.listdir(video_path))
            num_frames = len(frames)
            num_sequences = num_frames - sequence_length + 1
            for start_idx in range(num_sequences):
                self.index_mapping.append((video_name, start_idx))
        
        self.total_items = len(self.index_mapping)
        
        if train:
             print("Number of train data sequences:", self.total_items)
        else:
             print("Number of test data sequences:", self.total_items)

    def __len__(self):
        return self.total_items * self.loops

    def __getitem__(self, index):
        index = index % self.total_items
        video_name, start_idx = self.index_mapping[index]

        video_path = os.path.join(self.video_root, video_name)
        mask_path = os.path.join(self.masks_root, video_name)

        videos, gts, size = load_videos_from_jpg_images(
                            video_path,                                       
                            mask_path,
                            self.image_size,
                            start_idx,
                            self.sequence_length,
                            self.train
                            )

        return videos, gts, torch.Tensor(size)
    
    def get_full_sequence(self, index):
        video_path = os.path.join(self.video_root, index)
        mask_path = os.path.join(self.masks_root, index)

        frames = sorted(os.listdir(video_path))

        videos, gts, size = load_videos_from_jpg_images(
                            video_path,                                       
                            mask_path,
                            self.image_size,
                            0,
                            len(frames),
                            self.train
                            )

        return videos, gts, torch.Tensor(size)



def get_dataset(args):
    datadir = args['task']
    transform_train, transform_test = get_transform(datadir)
    ds_train = VideoDataset(f"datasets/{datadir}", train=True, sequence_length=args['sequence_length'] ,transform=transform_train, image_size=args['size'], loops=1)
    ds_test = VideoDataset(f"datasets/{datadir}", train=False, sequence_length=args['sequence_length'] ,transform=transform_test, image_size=args['size'])
    return ds_train, ds_test


if __name__ == "__main__":
    import argparse
    import os
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from sam2.utils.transforms import SAM2Transforms

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-Idim', '--Idim', default=512, help='learning_rate', required=False)
    parser.add_argument('-pSize', '--pSize', default=4, help='learning_rate', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='learning_rate', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='learning_rate', required=False)
    parser.add_argument('-rotate', '--rotate', default=20, help='learning_rate', required=False)
    args = vars(parser.parse_args())

    sam_args = {
        'sam_checkpoint': "../cp/sam_vit_b.pth",
        'model_type': "vit_b",
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': 0,
    }
    sam = SAM2VideoPredictor().from_pretrained("facebook/sam2.1-hiera-large")
    sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    sam_trans = SAM2Transforms(sam.image_encoder.img_size, 0.5)
    ds_train, ds_test = get_dataset(args, sam_trans)
    ds = torch.utils.data.DataLoader(ds_train,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=True,
                                     drop_last=True)
    pbar = tqdm(ds)
    mean0_list = []
    mean1_list = []
    mean2_list = []
    std0_list = []
    std1_list = []
    std2_list = []
    for i, (img, mask, _, _) in enumerate(pbar):
        a = img.mean(dim=(0, 2, 3))
        b = img.std(dim=(0, 2, 3))
        mean0_list.append(a[0].item())
        mean1_list.append(a[1].item())
        mean2_list.append(a[2].item())
        std0_list.append(b[0].item())
        std1_list.append(b[1].item())
        std2_list.append(b[2].item())
    print(np.mean(mean0_list))
    print(np.mean(mean1_list))
    print(np.mean(mean2_list))

    print(np.mean(std0_list))
    print(np.mean(std1_list))
    print(np.mean(std2_list))

        # a = img.squeeze().permute(1, 2, 0).cpu().numpy()
        # b = mask.squeeze().cpu().numpy()
        # a = (a - a.min()) / (a.max() - a.min())
        # cv2.imwrite('kaki.jpg', 255*a)
        # cv2.imwrite('kaki_mask.jpg', 255*b)
