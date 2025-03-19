import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from models.model_single import ModelEmb
import torch.nn.functional as F
from datasets.dataset import get_dataset
from sam2.utils.transforms import SAM2Transforms
from sam2.sam2_video_predictor import SAM2VideoPredictor
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda import set_device
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import shutil
import gc
import json 

NO_OBJ_SCORE = -1024.0

def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t =  prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()
            
def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0)) 
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))

import os

def train_single_epoch(rank, dataset, model, sam2, epoch, conn, args):
    try:
        setup(rank, int(args['world_size']))
        model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        model.train()
        optimizer = optim.Adam(model.parameters(),
                        lr=float(args['learning_rate']),
                        weight_decay=float(args['WD'])
                        )
        sam2.to(rank)
        sam2.eval()
        sampler = DistributedSampler(dataset, num_replicas=int(args['world_size']), rank=rank, shuffle=True)
        sampler.set_epoch(epoch)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(args['Batch_size']), sampler=sampler, drop_last=True)
        progress_bar = tqdm(dataloader, desc=f'(Train | {args["task"]}) epoch {epoch}', disable=rank != 0)
        loss_list = []
        criterion = FocalLoss()
        optimizer.zero_grad()
        for ix, (videos, gts, size) in enumerate(progress_bar):
            try:
                videos = videos.to(rank)
                gts = gts.to(rank)
                output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
                dense_embeddings = model(videos[:, 0])
                for t in range(args['sequence_length']):
                    masks = norm_batch(sam2_call(videos, sam2, dense_embeddings, output_dict, t, videos.shape))
                    if t == args['sequence_length'] - 1:
                        B, _, H, W = masks.shape
                        gts_sized = F.interpolate(gts[:, t].squeeze(2), size=(H, W), mode='nearest')
                        loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)
                        loss.backward()
                        if (ix + 1) % args['accumulation_steps'] == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                loss_item = loss.item()
                loss_list.append(loss_item)

            except Exception as e:
                torch.save(model.module.state_dict(), args['path_recent'])
                raise e
                
            if rank == 0:
                conn.send(np.mean(loss_list))
                progress_bar.set_description(
                    f'Train | {args["task"]} epoch {epoch} :: loss {np.mean(loss_list):.4f}'
                )
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        raise e

    finally:
        dist.destroy_process_group()

def sam2_call(videos, sam2, dense_embeddings, output_dict, t, shape):
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

    pix_feat = sam2._prepare_memory_conditioned_features(
        frame_idx=t,
        is_init_cond_frame=t==0,
        current_vision_feats=vision_feats[-1:],
        current_vision_pos_embeds=vision_pos_embeds[-1:],
        feat_sizes=feat_sizes[-1:],
        output_dict=output_dict,
        num_frames=T,
    )

    sam_point_coords = torch.zeros(B, 1, 2, device=sam2.device)
    sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=sam2.device)

    sparse_embeddings_none, dense_embeddings_none = sam2.sam_prompt_encoder(points=(sam_point_coords,sam_point_labels), boxes=None, masks=None)
    
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

    low_res_multimasks_ = torch.where(
        is_obj_appearing[:, None, None],
        low_res_multimasks.clone(),
        NO_OBJ_SCORE,
    )

    low_res_multimasks_ = low_res_multimasks_.float()

    high_res_multimasks = F.interpolate(
        low_res_multimasks_,
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

    output_dict["cond_frame_outputs"][t] = {
        "obj_ptr": obj_ptr,
        "maskmem_features":maskmem_features,
        "maskmem_pos_enc":maskmem_pos_enc
        }

    return low_res_multimasks

def inference(dataset, model, sam2, transform, epoch, args):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    progress_bar = tqdm(dataloader, desc=f'(Inference | {args["task"]}) Epoch {epoch}')
    model.eval()
    sam2.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    all_masks = []
    all_gts = []
    all_videos = []
    
    for videos, gts, size in progress_bar:
        try:
            videos = videos.to(sam2.device)
            gts = gts.to(sam2.device)
            output_dict = {
                "cond_frame_outputs": {}, 
                "non_cond_frame_outputs": {},
            }
            sequence_masks = []
            sequence_gts = []
            sequence_videos = []
            original_size = tuple([int(x) for x in size[0].squeeze().tolist()])
            dice_total, ji_total = 0.0, 0.0
            dense_embeddings = model(videos[:, 0])
            for t in range(videos.shape[1]):
                mask = norm_batch(sam2_call(videos, sam2, dense_embeddings, output_dict, t, videos.shape))
                mask = transform.postprocess_masks(mask, original_size)
                gt = transform.postprocess_masks(gts[:, t], original_size)
                mask = F.interpolate(mask, (Idim, Idim), mode='bilinear', align_corners=True)
                gt = F.interpolate(gt, (Idim, Idim), mode='nearest')
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                sequence_masks.append(mask.cpu())
                sequence_gts.append(gts[:, t].cpu())
                sequence_videos.append(videos[:, t].cpu())
                dice, ji = get_dice_ji(mask.squeeze().detach().cpu().numpy(),
                                         gt.squeeze().detach().cpu().numpy())
                dice_total += dice
                ji_total += ji
                
            all_masks.append(sequence_masks)
            all_gts.append(sequence_gts)
            all_videos.append(sequence_videos)
            iou_list.append(ji_total / videos.shape[1])
            dice_list.append(dice_total / videos.shape[1])

            progress_bar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: IoU {iou:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice=np.mean(dice_list),
                iou=np.mean(iou_list)))
            
        except Exception as e:
            print(f"Error processing video: {e}")
            continue

        
    return all_videos, all_masks, all_gts, iou_list, dice_list


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    set_device(rank)

def main(args=None, sam_args=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args['restart']:
        results = {
        "recent_epoch": 0, 
        "loss": {},
        "iou": {},
        "dice": {},
        "best_epoch": 0,
        "best_mean_iou": -float('inf'),
        "best_mean_dice": float('inf')
    }
        recent_epoch = 0
        model = ModelEmb(args=args).to(device)

    else:
        with open(args['path_results'], "r") as f:
            results = json.load(f)
        recent_epoch = results['recent_epoch'] + 1
        model = torch.load(args['path_recent']).to(device)

    sam2 = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")
    
    transform = SAM2Transforms(sam2.image_size, 0.5)

    trainset, testset = get_dataset(args) 

    for epoch in range(recent_epoch, int(args['epoches'])):
        parent_conn, child_conn = mp.Pipe()
        try:
            mp.spawn(train_single_epoch, args=(trainset, model, sam2, epoch, child_conn, args), nprocs=int(args['world_size']), join=True)
        except Exception as e:
            print(f"mp.spawn failed on epoch {epoch} with error: {e}")
            model.load_state_dict(torch.load(args['path_recent']))
        loss_list = []
        while parent_conn.poll():
            loss_list.append(parent_conn.recv())
        results['loss'][f"epoch_{epoch}"] = loss_list
        model.to(device)
        sam2.to(device)
        with torch.no_grad():
            videos, masks, gts, IoU_list, dice_list = inference(testset, model, sam2, transform, epoch, args)
            IoU_val = np.mean(IoU_list)
            results["iou"][f"epoch_{epoch}"] = IoU_list
            results["dice"][f"epoch_{epoch}"] = dice_list
            torch.save(model, args['path_recent'])
            results['recent_epoch'] = epoch
            if IoU_val > results['best_mean_iou']:
                torch.save(model, args['path_best'])
                results['best_mean_iou'] = IoU_val
                results['best_mean_dice'] = np.mean(dice_list)
                results['best_epoch'] = epoch
                print('best results: ' + str(results['best_mean_iou']))
                if os.path.exists(args['figures_folder']):
                    shutil.rmtree(args['figures_folder']) 
                os.makedirs(args['figures_folder'], exist_ok=True)  
                for j in range(len(videos)):
                    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
                    fig.suptitle(f"IoU: {IoU_list[j]}, Dice Loss: {dice_list[j]}", fontsize=16)
                    for i in range(args['sequence_length']):
                        frame = (videos[j][i].squeeze().permute(1, 2, 0)).clamp(0,1)
                        axes[0, i].imshow(frame.numpy()) 
                        axes[0, i].axis('off')

                        axes[1, i].imshow(masks[j][i].squeeze().numpy(), cmap="gray") 
                        axes[1, i].axis('off')

                        axes[2, i].imshow(gts[j][i].squeeze().numpy(), cmap="gray") 
                        axes[2, i].axis('off')

                    plt.subplots_adjust(wspace=0.05, hspace=0.05)
                    plt.tight_layout()
                    fig_path = os.path.join(args['figures_folder'], f"{j}.png")
                    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
                    plt.close(fig) 


        with open(args['path_results'], 'w') as f:
            json.dump(results, f, indent=4)
            f.flush()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,5"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_TIMEOUT"] = "60"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1" 
    torch.cuda.empty_cache()
    gc.collect()
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.0003, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=2, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=100, help='number of epoches', required=False)
    parser.add_argument('-accumulation_steps', '--accumulation_steps', default=1, help='number of epoches', required=False)
    parser.add_argument('-world_size', '--world_size', default=4, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='polypgen', help='evaluation iteration', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-sequence_length', '--sequence_length', default=8, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    parser.add_argument('-size', '--size', default=1024, help='image size', required=False)
    parser.add_argument('-restart', '--restart', default=False, help='image size', required=False)
    args = vars(parser.parse_args())
    os.makedirs('results', exist_ok=True)
    # folder = open_folder('results')
    folder = '0'
    args['folder'] = folder
    args['path_best'] = os.path.join('results',
                                     'gpu' + folder,
                                     'net_best.pth')
    args['path_recent'] = os.path.join('results',
                                     'gpu' + folder,
                                     'net_recent.pth')
    args['path_results'] = os.path.join('results',
                                    'gpu' + folder,
                                    'results.json')

    args['figures_folder'] = os.path.join('results', 'gpu' + args['folder'], 'figures')
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

