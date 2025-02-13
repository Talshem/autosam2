import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
from datasets.tfs import get_transform
import cv2

def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width

def load_videos_from_jpg_images(
    video_paths,
    image_size,
    mask,
    offload_video_to_cpu=False,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
):
    """
    Load frames from a PyTorch video_paths containing video frame directories (batch processing).
    
    Parameters:
        video_paths (torch.utils.data.DataLoader): video_paths with video frame directories.
        image_size (int): Size to resize frames to (square resolution).
        offload_video_to_cpu (bool): If True, keeps video tensors on CPU.
        img_mean (tuple): Mean for normalization.
        img_std (tuple): Standard deviation for normalization.
        async_loading_frames (bool): If True, uses asynchronous frame loading.
        compute_device (torch.device): Device to load tensors to (CPU/GPU).
    
    Returns:
        videos (torch.Tensor): Batch of videos, shape (B, T, 3, H, W).
        video_heights (list): Heights of original videos.
        video_widths (list): Widths of original videos.
    """
    batch_images = []
    batch_heights = []
    batch_widths = []
    
    for batch in tqdm(video_paths, desc="Loading videos in batch"):
        for video_path in batch:
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
            img_mean_tensor = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
            img_std_tensor = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

            images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
            for n, img_path in enumerate(img_paths):
                images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
            
            if not offload_video_to_cpu:
                images = images.to(compute_device)
                img_mean_tensor = img_mean_tensor.to(compute_device)
                img_std_tensor = img_std_tensor.to(compute_device)
            
            # Normalize by mean and std
            if not mask:
                images = (images - img_mean_tensor) / img_std_tensor
            
            batch_images.append(images)
            batch_heights.append(video_height)
            batch_widths.append(video_width)
    
    # Stack videos into a single tensor (B, T, C, H, W)
    videos = torch.nn.utils.rnn.pad_sequence(batch_images, batch_first=True)
    video_heights = torch.tensor(batch_heights, dtype=torch.int32)
    video_widths = torch.tensor(batch_widths, dtype=torch.int32)
    
    return videos, (video_heights, video_widths)


def cv2_loader(path, is_mask):
    files = os.listdir(path)
    frames = []    
    for f in sorted(files, key=lambda x: int(os.path.splitext(x)[0])):
        if os.path.splitext(f)[-1] in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            frame = cv2.imread(path, 0)
            if is_mask:
                frame[frame > 0] = 1
            else:
                frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            frames.append(frame)
    return frames


class VideoLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=cv2_loader,
                 sam_trans=None, loops=1):
        self.root = root
        if train:
            self.video_root = os.path.join(self.root, 'Training', 'frames')
            self.masks_root = os.path.join(self.root, 'Training', 'mask')
        else:
            self.video_root = os.path.join(self.root, 'Test', 'frames')
            self.masks_root = os.path.join(self.root, 'Test', 'mask')
        self.paths = os.listdir(self.video_root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.loops = loops
        self.sam_trans = sam_trans
        print('num of data:{}'.format(len(self.paths)))

    def __getitem__(self, index):
        index = index % len(self.paths)
        file_path = self.paths[index]

        video_path = os.path.join(self.video_root, file_path)
        mask_path = os.path.join(self.masks_root, file_path)

        videos, size = load_videos_from_jpg_images(video_path, args=['Idim'], mask=False)
        gts, gts_size = load_videos_from_jpg_images(mask_path, args=['Idim'], mask=True)


        # return self.sam_trans.preprocess(frames), self.sam_trans.preprocess(masks), torch.Tensor(
        #     original_size), torch.Tensor(image_size)

        return videos, gts, torch.Tensor(size)
        
    def __len__(self):
        return len(self.paths) * self.loops


def get_dataset(args, transofrm):
    datadir = args['task']
    # transform_train, transform_test = get_something_transform(args)
    ds_train = VideoLoader(datadir, train=True, sam_trans=sam_trans, loops=5)
    ds_test = VideoLoader(datadir, train=False, sam_trans=sam_trans)
    return ds_train, ds_test


if __name__ == "__main__":
    from tqdm import tqdm
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
