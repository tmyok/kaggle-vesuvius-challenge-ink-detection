import argparse
import cv2
import gc
import numpy as np
import os
import PIL.Image as Image
import random
import torch

from pathlib import Path
from tqdm import tqdm

from model_inference import UNet3D, ResidualUNetSE3D, SegFormer

IS_ROT_TTA = False
IS_FLIP_TTA = False

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_images(input_dir, img_depth, z_start, z_end, resize_ratio):

    # all images
    image_path_list = sorted(list(Path(input_dir).glob('*.tif')))
    # select candidates
    image_path_list = image_path_list[z_start:z_end + img_depth]

    img_stack = []
    for image_path in tqdm(image_path_list, total=len(image_path_list), dynamic_ncols=True, desc="Loading images"):
        img = np.array(Image.open(str(image_path)), dtype=np.uint16)
        img = (img >> 8).astype(np.uint8)
        img = cv2.resize(img, dsize=None, fx=1.0/resize_ratio, fy=1.0/resize_ratio, interpolation=cv2.INTER_AREA)
        img_stack.append(img)
        del img
        gc.collect()

    img_stack = np.stack(img_stack, axis=0)

    return img_stack

def get_scanning_positions(mask, roi_size, stride, offset=0):

    h, w = mask.shape
    x_pos_list = []
    y_pos_list = []
    for y in range(offset, h-roi_size+1, stride):
        for x in range(offset, w-roi_size+1, stride):
            if mask[y:y+roi_size, x:x+roi_size].mean() > 0.1:
                x_pos_list.append(x)
                y_pos_list.append(y)

    return x_pos_list, y_pos_list

def process_tile(model, device, img_tile, is_rot_tta, is_flip_tta):

    img_tile = torch.as_tensor(img_tile, dtype=torch.float32).unsqueeze(0).to(device)

    predictions = []
    with torch.no_grad():
        pred = model(img_tile)
        predictions.append(pred.cpu())

        if is_rot_tta:
            # rotation TTA
            for i in range(1,4):
                pred = model(torch.rot90(img_tile, k=i, dims=(-2, -1)))
                pred = torch.rot90(pred, k=-i, dims=(-2, -1)) # Rotate back
                predictions.append(pred.cpu())

        if is_flip_tta:
            # flip TTA
            pred = model(torch.flip(img_tile, dims=(-2,)))
            pred = torch.flip(pred, dims=(-2,)) # Flip back
            predictions.append(pred.cpu())

            pred = model(torch.flip(img_tile, dims=(-1,)))
            pred = torch.flip(pred, dims=(-1,)) # Flip back
            predictions.append(pred.cpu())

    predictions = torch.stack(predictions).numpy()

    return np.mean(predictions, axis=0)

def normalize_img(img):

    img = img.astype(np.float32)
    if img.sum() != 0:
        img[img > 0] = (img[img > 0] - img[img > 0].mean()) / (img[img > 0].std() + 1e-8)

    return img

def inference(
    img_stack, mask_resized,
    z_start, z_end, img_depth,
    num_instances,
    roi_size, stride,
    model, device,
    is3d,
    is_rot_tta, is_flip_tta,
):
    # Get the scanning position of the image
    x_pos_list, y_pos_list = get_scanning_positions(mask_resized, roi_size, stride)

    # inference
    _, input_h, input_w = img_stack.shape

    output_img = np.zeros((input_h, input_w), dtype=np.float32)
    count_img = np.zeros((input_h, input_w), dtype=np.float32)

    weight = np.ones((roi_size, roi_size), dtype=np.float32)
    weight[0:roi_size//4, :] *= np.linspace(0, 1, num=roi_size//4, dtype=np.float32).reshape(-1, 1)
    weight[-roi_size//4:, :] *= np.linspace(1, 0, num=roi_size//4, dtype=np.float32).reshape(-1, 1)
    weight[:, 0:roi_size//4] *= np.linspace(0, 1, num=roi_size//4, dtype=np.float32).reshape(1, -1)
    weight[:, -roi_size//4:] *= np.linspace(1, 0, num=roi_size//4, dtype=np.float32).reshape(1, -1)

    for y, x in tqdm(zip(y_pos_list, x_pos_list), total=len(y_pos_list), dynamic_ncols=True, desc="Inference"):

        z_start_list = np.linspace(0, z_end-z_start, num=num_instances, dtype=np.int32)

        img_tile = []
        for z in z_start_list:
            img = img_stack[z:z + img_depth, y:y + roi_size, x:x + roi_size]
            img = (img / 255.0).astype(np.float32)
            if is3d:
                img = img[np.newaxis, ...] # DHW -> CDHW
            img_tile.append(normalize_img(img))
        img_tile = np.stack(img_tile, axis=0)

        output = process_tile(model, device, img_tile, is_rot_tta, is_flip_tta)

        output_img[y:y+roi_size, x:x+roi_size] += output * weight
        count_img[y:y+roi_size, x:x+roi_size] += weight

    # averaging
    output_img = output_img / np.maximum(count_img, 1e-5)

    torch.cuda.empty_cache()

    return output_img

def inference_fold(
    fold_id,
    image_path,
    mask_path,
    model_dir,
    device,
    is_rot_tta, is_flip_tta,
):

    mask_org = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_org[mask_org>0] = 1

    #-------------------------------
    # stage1: CNN based segmentation
    print(f"fold{fold_id}: stage1")
    resize_ratio = 2
    z_start = 20
    z_end = 55
    img_depth = 8
    num_instances = 12

    roi_size = 384
    stride = roi_size // 2

    # weight is calculated by forward selection
    weight0 = 0.49

    # loading
    img_stack = load_images(image_path, img_depth, z_start, z_end, resize_ratio)
    mask_resized = cv2.resize(mask_org, dsize=None, fx=1.0/resize_ratio, fy=1.0/resize_ratio, interpolation=cv2.INTER_NEAREST)

    # padding
    img_stack = np.pad(img_stack, [(0, 0), (0, stride), (0, stride)], 'constant')
    mask_resized = np.pad(mask_resized, [(0, stride), (0, stride)], 'constant')

    output_map = np.zeros_like(mask_resized, dtype=np.float32)

    ###
    # UNet3D
    print(f"UNet3D")
    model = UNet3D()
    weight = torch.load(f"{model_dir}/UNet3D/fold{fold_id}/val_dice.ckpt")
    model.load_state_dict(weight["state_dict"], strict=False)
    model = model.to(device).eval()

    pred = inference(
        img_stack, mask_resized,
        z_start, z_end, img_depth,
        num_instances, roi_size, stride,
        model, device,
        is3d=True,
        is_rot_tta=is_rot_tta,
        is_flip_tta=is_flip_tta,
    )
    output_map = pred

    del model, pred
    gc.collect()
    torch.cuda.empty_cache()
    # UNet3D
    ###

    ###
    # ResidualUNetSE3D
    print(f"ResidualUNetSE3D")
    model = ResidualUNetSE3D()
    weight = torch.load(f"{model_dir}/ResidualUNetSE3D/fold{fold_id}/val_dice.ckpt")
    model.load_state_dict(weight["state_dict"], strict=False)
    model = model.to(device).eval()

    pred = inference(
        img_stack, mask_resized,
        z_start, z_end, img_depth,
        num_instances, roi_size, stride,
        model, device,
        is3d=True,
        is_rot_tta=is_rot_tta,
        is_flip_tta=is_flip_tta,
    )
    output_map = (1 - weight0) * output_map + weight0 * pred

    del model, pred
    gc.collect()
    torch.cuda.empty_cache()
    # ResidualUNetSE3D
    ###

    # resize
    output_map = cv2.resize(output_map, dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC)
    output_map = np.clip(output_map, 0, 1)
    # masking
    output_map = output_map[:mask_org.shape[0], :mask_org.shape[1]] * (mask_org > 0)

    del img_stack, mask_resized
    gc.collect()

    # stage1: CNN based segmentation
    #-------------------------------

    #-------------------------------
    # stage2: Transformer based segmentation
    print(f"fold{fold_id}: stage2")
    resize_ratio = 1
    z_start = 10
    z_end = 55
    img_depth = 3
    num_instances = 20

    roi_size = 384
    stride = roi_size // 2

    stage1_thr = 0.57

    # loading
    img_stack = load_images(image_path, img_depth, z_start, z_end, resize_ratio)
    mask = np.zeros_like(output_map)
    mask[output_map>stage1_thr] = 1
    mask_resized = cv2.resize(mask, dsize=None, fx=1.0/resize_ratio, fy=1.0/resize_ratio, interpolation=cv2.INTER_NEAREST)

    # padding
    img_stack = np.pad(img_stack, [(0, 0), (0, stride), (0, stride)], 'constant')
    mask_resized = np.pad(mask_resized, [(0, stride), (0, stride)], 'constant')

    print(f"SegFormer")
    model = SegFormer()
    weight = torch.load(f"{model_dir}/mit_b2/fold{fold_id}/val_dice.ckpt")
    model.load_state_dict(weight["state_dict"], strict=False)
    model = model.to(device).eval()

    output_map = inference(
        img_stack, mask_resized,
        z_start, z_end, img_depth,
        num_instances, roi_size, stride,
        model, device,
        is3d=False,
        is_rot_tta=is_rot_tta,
        is_flip_tta=is_flip_tta,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # resize
    output_map = cv2.resize(output_map, dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC)
    output_map = np.clip(output_map, 0, 1)
    # masking
    output_map = output_map[:mask_org.shape[0], :mask_org.shape[1]] * (mask_org > 0)

    del img_stack, mask, mask_resized
    gc.collect()

    # stage2: Transformer based segmentation
    #-------------------------------

    del mask_org
    gc.collect()

    return output_map

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_id", type=int, default=0)
    args = parser.parse_args()

    fold_id = args.fold_id

    model_dir = "../input/vcid-trained-weight"
    result_dir = "../output/validation"

    fragment_path = Path(f"../input/vcid-validation/fold{fold_id}")

    os.makedirs(f"{result_dir}/fold{fold_id}", exist_ok=True)

    seed_everything(42)

    device = torch.device("cuda:0")

    output_map = inference_fold(
        fold_id=fold_id,
        image_path=str(fragment_path / "surface_volume"),
        mask_path=str(fragment_path / "mask.png"),
        model_dir=model_dir,
        device=device,
        is_rot_tta=IS_ROT_TTA,
        is_flip_tta=IS_FLIP_TTA,
    )

    np.save(f"{result_dir}/fold{fold_id}/val_dice", output_map)
    cv2.imwrite(f"{result_dir}/fold{fold_id}/val_dice.png", (output_map * 255).astype(np.uint8))
