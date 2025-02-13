import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from models.model_single import ModelEmb

from datasets.dataset import get_dataset
import torch.nn.functional as F
from sam2.utils.transforms import SAM2Transforms
from sam2.sam2_video_predictor import SAM2VideoPredictor

def norm_batch(x):
    B, T, C, H, W = x.shape
    min_value = x.view(B, T, -1).min(dim=2, keepdim=True)[0].view(B, T, 1, 1, 1)
    max_value = x.view(B, T, -1).max(dim=2, keepdim=True)[0].view(B, T, 1, 1, 1)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(2, 3, 4))
    fn = torch.sum(y_true * (1 - y_pred), dim=(2, 3, 4))
    fp = torch.sum((1 - y_true) * y_pred, dim=(2, 3, 4))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0), axis=(2, 3)) 
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0), axis=(2, 3))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0), axis=(2, 3))
    ji_per_frame = np.nan_to_num(tp / (tp + fp + fn)) 
    dice_per_frame = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
    dice = np.mean(dice_per_frame)
    ji = np.mean(ji_per_frame)
    return dice, ji



def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step(optimizer, gts, masks, criterion, accumulation_steps, step):
    B, T, _, H, W = masks.shape
    gts_sized = F.interpolate(gts.unsqueeze(2), size=(H, W), mode='nearest') 
    loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)
    loss = loss / T
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()


def get_input_dict(frames):
    B, F, C, H, W = frames.shape
    batched_input = []
    for b in range(B):
        video_input = []
        for f in range(F):
            single_input = {
                'image': frames[b, f, :, :, :],
                'point_coords': None,
                'point_labels': None,
            }
            video_input.append(single_input)
        batched_input.append(video_input)
    return batched_input


def postprocess_masks(masks_dict):
    masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).cuda()
    ious = torch.zeros(len(masks_dict)).cuda()
    for i in range(len(masks_dict)):
        cur_mask = masks_dict[i]['low_res_logits'].squeeze()
        cur_mask = (cur_mask - cur_mask.min()) / (cur_mask.max() - cur_mask.min())
        masks[i, 0] = cur_mask.squeeze()
        ious[i] = masks_dict[i]['iou_predictions'].squeeze()
    return masks, ious


def train_single_epoch(ds, model, sam2, optimizer, epoch):
    loss_list = []
    progress_bar = tqdm(ds)
    criterion = nn.BCELoss()
    Idim = int(args['Idim'])
    optimizer.zero_grad()
    for ix, (videos, gts, size) in enumerate(progress_bar):
        # tensor of frames per video
        videos = videos.to(sam2.device)

        # tensor of ground truth masks per video
        gts = gts.to(sam2.device)

        # AutoSAM automatic prompt embedding based on the first frame
        dense_embeddings = model(videos[:, 0])

        # Generate Normalized Output
        masks = norm_batch(sam2_call(videos, sam2, dense_embeddings))
        loss = gen_step(optimizer, gts, masks, criterion, accumulation_steps=4, step=ix)
        loss_list.append(loss)
        progress_bar.set_description(
            '(train | {}) epoch {epoch} ::'
            ' loss {loss:.4f}'.format(
                'Medical',
                epoch=epoch,
                loss=np.mean(loss_list)
            ))
    return np.mean(loss_list)

def sam2_call(videos, sam2, dense_embeddings):
    with torch.no_grad():
        B, F, C, H, W = videos.shape
        memory_bank = []
        predicted_masks = []
        for f in range(F):
            
            # image embedding using sam2 image encoder
            image_embeddings = sam2.image_encoder(videos[:, f])
            # empty prompt embedding using sam2 prompt encoder
            sparse_embeddings_none, dense_embeddings_none = sam2.sam_prompt_encoder(points=None, boxes=None, masks=None)

            # Optional - I dont know if its true, encoding the first frame with no mask
            # We should ask aviad how the memory attention handles the initial state where
            # No memory is yet accumulated
            encoded_memory = sam2.memory_encoder(image_embeddings, torch.zeros_like(videos[:, f]))

            if len(memory_bank) == 0:
                memory_bank.append({
                    # appending to memory the features of the first frame
                    "features": encoded_memory["vision_features"],
                    # takes positional encoding with highest resolution
                    "pos_enc": encoded_memory["vision_pos_enc"]
                })
           
            # Converting to torch tensor
            memory = torch.stack([m["features"] for m in memory_bank], dim=0)
            memory_pos = torch.stack([m["pos_enc"] for m in memory_bank], dim=0)
            
            # Updating features of the next frame based on previous memory
            updated_features = sam2.memory_attention(
                curr=image_embeddings["vision_features"].permute(0, 2, 3, 1), # Permuting since dims class in memory attention
                curr_pos=image_embeddings["vision_pos_enc"][-1].permute(0, 2, 3, 1), # Permuting since dims class in memory attention
                memory=memory,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=0 # Since no point prompt was actually inserted
            )

            # Decoding the mask based on the memory-based updated embeddings and initial frame AutoSAM automatic-prompt embedding
            output_mask, _ = sam2.mask_decoder(
                image_embeddings=updated_features["vision_features"],
                image_pe=updated_features["vision_pos_enc"][-1],
                sparse_prompt_embeddings=sparse_embeddings_none,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                )
            
            predicted_masks.append(output_mask)

            # Encoding the memory of the recent frame based on its features and predicted mask
            encoded_memory = sam2.memory_encoder(updated_features, output_mask)
            
            # makes sure memory bank do not exceed the number of masks in memory parameted initiated in the model
            if len(memory_bank) >= sam2.num_maskmem:
                memory_bank.pop(1)
            
            # Appending in frames {sam2.memory_temporal_stride_for_eval} intervals the encoded memory
             
            if f % sam2.memory_temporal_stride_for_eval == 0:
                memory_bank.append({
                    "features": encoded_memory["vision_features"],
                    "pos_enc": encoded_memory["vision_pos_enc"][-1]
                })

        return torch.stack(predicted_masks, dim=1)

def inference_ds(ds, model, sam2, transform, epoch, args):
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    for videos, gts, size in pbar:
        videos = videos.to(sam2.device)
        gts = gts.to(sam2.device)
        dense_embeddings = model(videos[:, 0])
        masks = norm_batch(sam2_call(videos, sam2, dense_embeddings))
        original_size =  tuple([int(x) for x in size[0].squeeze().tolist()])
        masks = transform.postprocess_masks(masks, original_size)
        gts = transform.postprocess_masks(gts.unsqueeze(dim=0), original_size)
        masks = F.interpolate(masks, (Idim, Idim), mode='bilinear', align_corners=True)
        gts = F.interpolate(gts, (Idim, Idim), mode='nearest')
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0
        dice, ji = get_dice_ji(masks.squeeze().detach().cpu().numpy(),
                               gts.squeeze().detach().cpu().numpy())
        iou_list.append(ji)
        dice_list.append(dice)
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: IoU {iou:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice=np.mean(dice_list),
                iou=np.mean(iou_list)))
    model.train()
    return np.mean(iou_list)


def main(args=None, sam_args=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ModelEmb(args=args).to(device)

    sam2 = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")
    
    transform = SAM2Transforms(sam2.image_size, 0.5)
    optimizer = optim.Adam(model.parameters(),
                           lr=float(args['learning_rate']),
                           weight_decay=float(args['WD'])
                           )
    trainset, testset = get_dataset(args) 

    ds = torch.utils.data.DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                                     num_workers=int(args['nW']), drop_last=True)
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)
    best = 0
    path_best = 'results/gpu' + str(args['folder']) + '/best.csv'
    f_best = open(path_best, 'w')
    for epoch in range(int(args['epoches'])):
        # train_single_epoch(ds, model.train(), sam.eval(), optimizer, transform, epoch)
        train_single_epoch(ds, model.train(), sam2.eval(), optimizer, epoch)
        with torch.no_grad():
            # IoU_val = inference_ds(ds_val, model.eval(), sam, transform, epoch, args)
            IoU_val = inference_ds(ds_val, model.eval(), sam2, epoch, args)
            if IoU_val > best:
                torch.save(model, args['path_best'])
                best = IoU_val
                print('best results: ' + str(best))
                f_best.write(str(epoch) + ',' + str(best) + '\n')
                f_best.flush()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.0003, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=1, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=100, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='polypgen', help='evaluation iteration', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    args = vars(parser.parse_args())
    os.makedirs('results', exist_ok=True)
    folder = open_folder('results')
    args['folder'] = folder
    args['path'] = os.path.join('results',
                                'gpu' + folder,
                                'net_last.pth')
    args['path_best'] = os.path.join('results',
                                     'gpu' + folder,
                                     'net_best.pth')
    args['vis_folder'] = os.path.join('results', 'gpu' + args['folder'], 'vis')
    os.mkdir(args['vis_folder'])
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

