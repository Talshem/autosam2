import torch.optim as optim
import torch.utils.data
import torch
import os
import numpy as np
from models.model_single import ModelEmb
import torch.nn.functional as F
from datasets.dataset import get_dataset
from sam2.utils.transforms import SAM2Transforms
from sam2.sam2_video_predictor import SAM2VideoPredictor
import json
from torch.cuda import set_device
import shutil
import gc
from PIL import Image
import torchvision.transforms as transforms
import supervision as sv
import cv2
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

NO_OBJ_SCORE = -1024.0

def get_bounding_box_and_middle(mask):
    y_indices, x_indices = np.where(mask > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None, None  

    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    bbox = ([x_min, y_min, x_max, y_max])
    middle_point = ([(x_center, y_center)], [1])

    return bbox, middle_point


def ensure_empty_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x

def sam2_call(videos, sam2, dense_embeddings, output_dict, t, shape, state=None, input=None):
    B, T, C , H , W = shape
    current_out = {}
    with torch.no_grad():
        image_embeddings = sam2.image_encoder(videos[:, t])
        
        image_embeddings["backbone_fpn"][0] = sam2.sam_mask_decoder.conv_s0(
            image_embeddings["backbone_fpn"][0]
        )
        image_embeddings["backbone_fpn"][1] = sam2.sam_mask_decoder.conv_s1(
            image_embeddings["backbone_fpn"][1]
        )

        image_embeddings, vision_feats, vision_pos_embeds, feat_sizes = sam2._prepare_backbone_features(image_embeddings)


        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        sam_point_coords = torch.zeros(B, 1, 2, device=sam2.device)
        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=sam2.device)

        if state == 'autosam2':
            sparse_embeddings_none, dense_embeddings_none = sam2.sam_prompt_encoder(points=(sam_point_coords,sam_point_labels), boxes=None, masks=None)
        elif state == 'sam2_point':
            sam_point_coords = torch.tensor(input[0], dtype=torch.float32, device=sam2.device).unsqueeze(0)  
            sam_point_labels = torch.tensor(input[1], dtype=torch.int32, device=sam2.device).unsqueeze(0)
            sparse_embeddings_none, dense_embeddings = sam2.sam_prompt_encoder(points=(sam_point_coords,sam_point_labels), boxes=None, masks=None)
        elif state == 'sam2_gt':
            sparse_embeddings_none, dense_embeddings = sam2.sam_prompt_encoder(points=(sam_point_coords,sam_point_labels), boxes=None, masks=input.to(sam2.device))
        elif state == 'sam2_bb':
            boxes = torch.tensor(input, dtype=torch.float32, device=sam2.device).unsqueeze(0)  
            sparse_embeddings_none, dense_embeddings = sam2.sam_prompt_encoder(points=(sam_point_coords,sam_point_labels), boxes=boxes, masks=None)
    
    pix_feat = sam2._prepare_memory_conditioned_features(
        frame_idx=t,
        is_init_cond_frame=t==0,
        current_vision_feats=vision_feats[-1:],
        current_vision_pos_embeds=vision_pos_embeds[-1:],
        feat_sizes=feat_sizes[-1:],
        output_dict=output_dict,
        num_frames=T,
    )    

    low_res_multimasks, ious, sam_output_tokens, object_score_logits = sam2.sam_mask_decoder(
    image_embeddings=pix_feat,
    image_pe=sam2.sam_prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings_none,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=False,
    repeat_image=False,  
    high_res_features=high_res_features,
)   

    is_obj_appearing = object_score_logits > 0

    low_res_multimasks = torch.where(
        is_obj_appearing[:, None, None],
        low_res_multimasks,
        NO_OBJ_SCORE,
    )
    low_res_multimasks = low_res_multimasks.float()
    high_res_multimasks = F.interpolate(
        low_res_multimasks,
        size=(sam2.image_size, sam2.image_size),
        mode="bilinear",
        align_corners=False,
    )
    sam_output_token = sam_output_tokens[:, 0]

    obj_ptr = sam2.obj_ptr_proj(sam_output_token)

    if sam2.pred_obj_scores:
        if sam2.soft_no_obj_ptr:
            lambda_is_obj_appearing = object_score_logits.sigmoid()
        else:
            lambda_is_obj_appearing = is_obj_appearing.float()

        if sam2.fixed_no_obj_ptr:
            obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * sam2.no_obj_ptr

    current_out["pred_masks"] = low_res_multimasks
    current_out["pred_masks_high_res"] = high_res_multimasks
    current_out["obj_ptr"] = obj_ptr

    maskmem_features, maskmem_pos_enc = sam2._encode_new_memory(
        vision_feats,
        feat_sizes,
        high_res_multimasks,
        object_score_logits,
        False,
    )
    
    if t == 0:
        output_dict["cond_frame_outputs"][t] = {
            "obj_ptr": obj_ptr,
            "maskmem_features":maskmem_features,
            "maskmem_pos_enc":maskmem_pos_enc
            }
    else:
        output_dict["non_cond_frame_outputs"][t] = {
            "obj_ptr": obj_ptr,
            "maskmem_features":maskmem_features,
            "maskmem_pos_enc":maskmem_pos_enc
            }

    return low_res_multimasks

def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0)) 
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def inference(frames, gts, size, model, sam2, transform, state, input, args):
    Idim = int(args['Idim'])
    frames = frames.unsqueeze(0).to(sam2.device)
    gts = gts.unsqueeze(0).to(sam2.device)
    output_dict = {
        "cond_frame_outputs": {}, 
        "non_cond_frame_outputs": {},
    }
    sequence_masks = []
    sequence_gts = []
    sequence_videos = []
    sequence_dice = []
    sequence_iou = []
    frames = frames[:, 21:]
    dense_embeddings = model(frames[:, 0])
    for t in range(frames.shape[1]):
        mask = norm_batch(sam2_call(frames, sam2, dense_embeddings, output_dict, t, frames.shape, state, input))
        mask = transform.postprocess_masks(mask, size)
        gt = transform.postprocess_masks(gts[:, t], size)
        mask = F.interpolate(mask, (Idim, Idim), mode='bilinear', align_corners=True)
        gt = F.interpolate(gt, (Idim, Idim), mode='nearest')
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        sequence_masks.append(mask.cpu())
        sequence_gts.append(gts[:, t].cpu())
        sequence_videos.append(frames[:, t].cpu())
        dice, ji = get_dice_ji(mask.squeeze().detach().cpu().numpy(),
                        gt.squeeze().detach().cpu().numpy())
        sequence_dice.append(dice)
        sequence_iou.append(ji)

    return sequence_videos, sequence_masks, sequence_gts, sequence_iou, sequence_dice

def main(args=None, sam_args=None, annotations=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = torch.load(args['model_path']).to(device)

    sam2 = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large").to(device)
    
    transform = SAM2Transforms(sam2.image_size, 0.5)
    trainset, testset = get_dataset(args) 
    
    results = {
    "autosam2": {
        "iou": [],
        "iou_mean": [],
        "dice": [],
        "dice_mean": []
        },
    "sam2_point": {
        "iou": [],
        "iou_mean": [],
        "dice": [],
        "dice_mean": []
        },
    "sam2_bb": {
        "iou": [],
        "iou_mean": [],
        "dice": [],
        "dice_mean": []
        },
    "sam2_gt": {
        "iou": [],
        "iou_mean": [],
        "dice": [],
        "dice_mean": []
        }
}

    for i in range(1, len(testset)+1):
        print(i)
        masks_sam2_point_path = os.path.join(args['masks_sam2_point_path'], str(i))
        masks_sam2_bb_path = os.path.join(args['masks_sam2_bb_path'], str(i))
        masks_sam2_gt_path = os.path.join(args['masks_sam2_gt_path'], str(i))
        masks_autosam2_path = os.path.join(args['masks_autosam2_path'], str(i))
        autosam2_annotated_path = os.path.join(args['autosam2_annotated_path'], str(i))
        masks_gt_path = os.path.join(args['masks_gt_path'], str(i))
        
        directories = [
        masks_sam2_point_path,
        masks_sam2_bb_path,
        masks_sam2_gt_path,
        masks_autosam2_path,
        autosam2_annotated_path,
        masks_gt_path
    ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True) 
            ensure_empty_directory(directory)

        frames, gts, size = testset[i-1]
        original_size =  tuple([int(x) for x in size.unsqueeze(0)[0].squeeze().tolist()])
        to_pil = transforms.ToPILImage()

        data_path = f"datasets/polypgen/Test/images/{i}"

        with torch.no_grad():
            videos_list, masks_list, gts_list, iou_list, dice_list = inference(frames, gts, original_size, model.eval(), sam2.eval(), transform, 'autosam2', None, args)
            first_frame = Image.open(f"{data_path}/{1}.jpg")
            results["autosam2"]["iou"].append(iou_list)
            results["autosam2"]["iou_mean"].append(np.mean(iou_list))
            results["autosam2"]["dice"].append(dice_list)
            results["autosam2"]["dice_mean"].append(np.mean(dice_list))
            gif_pred = []
            gif_gt = []
            for t in range(len(videos_list)):
                mask_path = os.path.join(masks_autosam2_path, f"{t+1}.png")
                mask_sized = F.interpolate(masks_list[t].squeeze(2), size=original_size, mode='nearest')
                pred_mask = to_pil(mask_sized.squeeze())
                pred_mask.save(mask_path)
                gt_sized = F.interpolate(gts_list[t].squeeze(2), size=original_size, mode='nearest')
                gt_path = os.path.join(masks_gt_path, f"{t+1}.png")
                gt_mask = to_pil(gt_sized.squeeze())
                gt_mask.save(gt_path)
                frame = Image.open(f"{data_path}/{t+1}.jpg")
                color = (0, 153, 255)
                color_mask_pred = Image.new("RGBA", (original_size[1], original_size[0]), color + (0,))
                color_mask_gt = Image.new("RGBA", (original_size[1], original_size[0]), color + (0,))
                mask_rgba = Image.new("RGBA", (original_size[1], original_size[0]), color + (int(255 * 0.5),))
                color_mask_pred.paste(mask_rgba, (0, 0), pred_mask)
                color_mask_gt.paste(mask_rgba, (0, 0), gt_mask)
                frame = frame.convert("RGBA")
                overlayed_image_pred = Image.alpha_composite(frame, color_mask_pred)
                overlayed_image_pred.convert("RGB").save(os.path.join(autosam2_annotated_path, f"pred_{t+1}.png"))
                overlayed_image_gt = Image.alpha_composite(frame, color_mask_gt)
                overlayed_image_gt.convert("RGB").save(os.path.join(autosam2_annotated_path, f"gt_{t+1}.png"))
                gif_pred.append(overlayed_image_pred)
                gif_gt.append(overlayed_image_gt)
            first_frame_path_1 = os.path.join(masks_autosam2_path, "annotated.png")
            first_frame_path_2 = os.path.join(masks_gt_path, "annotated.png")
            first_frame.save(first_frame_path_1)
            first_frame.save(first_frame_path_2)
            gif_pred_path = os.path.join(autosam2_annotated_path, "pred_gif.gif")
            gif_gt_path = os.path.join(autosam2_annotated_path, "gt_gif.gif")
            gif_pred[0].save(gif_pred_path, save_all=True, append_images=gif_pred[1:], loop=0, duration=40)
            gif_gt[0].save(gif_gt_path, save_all=True, append_images=gif_gt[1:], loop=0, duration=40)
            bbox, point = get_bounding_box_and_middle(F.interpolate(gts_list[0].squeeze(2), size=original_size, mode='nearest').squeeze().numpy())
            gt = F.interpolate(
                        gts_list[0].float(),
                        size=sam2.sam_prompt_encoder.mask_input_size,
                        align_corners=False,
                        mode="bilinear",
                        antialias=True, 
                    )
            _, masks_sam2_gt, _, iou_list_sam2_gt, dice_list_sam2_gt = inference(frames, gts, original_size, model.eval(), sam2.eval(), transform, 'sam2_gt', gt, args)
            results["sam2_gt"]["iou"].append(iou_list_sam2_gt)
            results["sam2_gt"]["iou_mean"].append(np.mean(iou_list_sam2_gt))
            results["sam2_gt"]["dice"].append(dice_list_sam2_gt)
            results["sam2_gt"]["dice_mean"].append(np.mean(dice_list_sam2_gt))
            for t in range(len(videos_list)):
                mask_path = os.path.join(masks_sam2_gt_path, f"{t+1}.png")
                mask_sized = F.interpolate(masks_sam2_gt[t].squeeze(2), size=original_size, mode='nearest')
                pred_mask = to_pil(mask_sized.squeeze())
                pred_mask.save(mask_path)
            frame = Image.open(f"{data_path}/{1}.jpg")
            color = (0, 153, 255)
            color_mask = Image.new("RGBA", (original_size[1], original_size[0]), color + (0,))
            mask_rgba = Image.new("RGBA", (original_size[1], original_size[0]), color + (int(255),))
            gt_sized = F.interpolate(gts_list[0].squeeze(2), size=original_size, mode='nearest')
            gt_pil = to_pil(gt_sized.squeeze())
            color_mask.paste(mask_rgba, (0, 0), gt_pil)
            frame = frame.convert("RGBA")
            overlayed_image = Image.alpha_composite(frame, color_mask)
            overlayed_image.convert("RGB").save(os.path.join(masks_sam2_gt_path, "annotated.png"))

            _, masks_sam2_point, _, iou_list_sam2_point, dice_list_sam2_point = inference(frames, gts, original_size, model.eval(), sam2.eval(), transform, 'sam2_point', point, args)
            results["sam2_point"]["iou"].append(iou_list_sam2_point)
            results["sam2_point"]["iou_mean"].append(np.mean(iou_list_sam2_point))
            results["sam2_point"]["dice"].append(dice_list_sam2_point)
            results["sam2_point"]["dice_mean"].append(np.mean(dice_list_sam2_point))
            for t in range(len(videos_list)):
                mask_path = os.path.join(masks_sam2_point_path, f"{t+1}.png")
                mask_sized = F.interpolate(masks_sam2_point[t].squeeze(2), size=original_size, mode='nearest')
                pred_mask = to_pil(mask_sized.squeeze())
                pred_mask.save(mask_path)
            first_frame = Image.open(f"{data_path}/{1}.jpg")
            annotated_frame_path = os.path.join(masks_sam2_point_path, "annotated.png")
            draw_1 = ImageDraw.Draw(first_frame)
            point_x, point_y = point[0][0]
            radius = 30
            draw_1.ellipse((point_x - radius, point_y - radius, point_x + radius, point_y + radius), fill="green", outline="black", width=5)
            first_frame.save(annotated_frame_path)
            
            _, masks_sam2_bb, _, iou_list_sam2_bb, dice_list_sam2_bb = inference(frames, gts, original_size, model.eval(), sam2.eval(), transform, 'sam2_bb', bbox, args)
            results["sam2_bb"]["iou"].append(iou_list_sam2_bb)
            results["sam2_bb"]["iou_mean"].append(np.mean(iou_list_sam2_bb))
            results["sam2_bb"]["dice"].append(dice_list_sam2_bb)
            results["sam2_bb"]["dice_mean"].append(np.mean(dice_list_sam2_bb))
            for t in range(len(videos_list)):
                mask_path = os.path.join(masks_sam2_bb_path, f"{t+1}.png")
                mask_sized = F.interpolate(masks_sam2_bb[t].squeeze(2), size=original_size, mode='nearest')
                pred_mask = to_pil(mask_sized.squeeze())
                pred_mask.save(mask_path)
            first_frame = Image.open(f"{data_path}/{1}.jpg")
            annotated_frame_path = os.path.join(masks_sam2_bb_path, "annotated.png")
            draw_2 = ImageDraw.Draw(first_frame)
            x_min, y_min, x_max, y_max = bbox 
            radius = 10
            draw_2.rectangle(
                [(x_min, y_min), (x_max, y_max)], 
                outline="blue", width=8
            )
            first_frame.save(annotated_frame_path)

        with open(args['results'], 'w') as f:
            json.dump(results, f, indent=4)
            f.flush()
        
        directories = [
        f"inference/masks_sam2_bb/{i}",
        f"inference/masks_sam2_gt/{i}",
        f"inference/masks_sam2_point/{i}",
        f"inference/masks_autosam2/{i}",
        f"inference/masks_gt_path/{i}"
    ]

        row_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

        num_images = 9

        fig, axes = plt.subplots(5, num_images+1, figsize=(15, 6))

        for row, (directory, label) in enumerate(zip(directories, row_labels)):
            image_files = ["annotated.png"] + [f"{j}.png" for j in range(1, 9)]
            
            for col, image_file in enumerate(image_files, start=1):
                image_path = os.path.join(directory, image_file)
                if os.path.exists(image_path):
                    img = mpimg.imread(image_path)
                    axes[row, col].imshow(img, cmap="gray")
                
                axes[row, col].axis("off")
            
            axes[row, 0].annotate(label, xy=(1.05, 0.5), xycoords='axes fraction',
                                        fontsize=14, ha='left', va='center', fontweight='bold')
            axes[row, 0].axis("off")

        plt.tight_layout()
        plt.savefig(f'inference/figures/{i}.png')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,5"
    torch.cuda.empty_cache()
    gc.collect()
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.0003, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=3, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=100, help='number of epoches', required=False)
    parser.add_argument('-accumulation_steps', '--accumulation_steps', default=3, help='number of epoches', required=False)
    parser.add_argument('-world_size', '--world_size', default=4, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='polypgen', help='evaluation iteration', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-sequence_length', '--sequence_length', default=8, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    parser.add_argument('-size', '--size', default=1024, help='image size', required=False)
    parser.add_argument('-index', '--index', default='14', help='image size', required=False)
    args = vars(parser.parse_args())
    os.makedirs('inference', exist_ok=True)
    args['masks_autosam2_path'] = os.path.join('inference','masks_autosam2')
    args['masks_sam2_point_path'] = os.path.join('inference','masks_sam2_point')
    args['masks_sam2_bb_path'] = os.path.join('inference','masks_sam2_bb')
    args['masks_sam2_gt_path'] = os.path.join('inference','masks_sam2_gt')
    args['autosam2_annotated_path'] = os.path.join('inference','autosam2_annotated')
    args['figures'] = os.path.join('inference','figures')
    args['masks_gt_path'] = os.path.join('inference','masks_gt_path')
    args['results'] = os.path.join('inference','inference.json')
    args['model_path'] = os.path.join('results','gpu0','net_best.pth')

    os.makedirs(args['masks_autosam2_path'], exist_ok=True)
    os.makedirs(args['autosam2_annotated_path'], exist_ok=True)
    os.makedirs(args['masks_sam2_point_path'], exist_ok=True)
    os.makedirs(args['masks_sam2_bb_path'], exist_ok=True)
    os.makedirs(args['masks_sam2_gt_path'], exist_ok=True)
    os.makedirs(args['masks_gt_path'], exist_ok=True)
    os.makedirs(args['figures'], exist_ok=True)
    
    sam_args = {
        'sam_checkpoint': "sam2/checkpoints/sam2.1_hiera_large.pth",
        'sam_config': "sam2/sam2_hiera_l.yaml",
        'model_type': "vit_h",
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

    main(args=args, sam_args=sam_args)
