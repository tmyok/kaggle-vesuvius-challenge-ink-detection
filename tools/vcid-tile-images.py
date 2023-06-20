import argparse
import cv2
import numpy as np
import pandas as pd
import PIL.Image as Image

from pathlib import Path
from tqdm import tqdm

def get_scanning_positions(mask, roi_size, stride, offset=0):

    mask = np.pad(mask, [(0, roi_size), (0, roi_size)], "constant")

    h, w = mask.shape
    x_pos_list = []
    y_pos_list = []
    for y in range(offset, h-roi_size+1, stride):
        for x in range(offset, w-roi_size+1, stride):
            if mask[y:y+roi_size, x:x+roi_size].max() > 0:
                x_pos_list.append(x)
                y_pos_list.append(y)

    return x_pos_list, y_pos_list

def draw_scanning_positions(input_dir, output_dir, fold_id, x_pos_list, y_pos_list, roi_size):

    inklabels = cv2.imread(str(input_dir / f"fold{fold_id}" / "inklabels.png"))
    ir = cv2.imread(str(input_dir / f"fold{fold_id}" / "ir.png"))
    for x, y in zip(x_pos_list, y_pos_list):
        inklabels = cv2.rectangle(
            inklabels,
            (x, y),
            (x+roi_size, y+roi_size),
            (0, 255, 0),
            thickness=10)
        ir = cv2.rectangle(
            ir,
            (x, y),
            (x+roi_size, y+roi_size),
            (0, 255, 0),
            thickness=10)

    cv2.imwrite(str(output_dir / f"crop_image-fold{fold_id}-inklabels.png"), inklabels)
    cv2.imwrite(str(output_dir / f"crop_image-fold{fold_id}-ir.png"), ir)

if __name__ == '__main__':

    # fixed parameters
    num_folds = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--type", type=str, choices=["mask", "inklabels"], default="inklabels")

    args = parser.parse_args()

    roi_size = args.roi_size
    stride = args.stride
    mask_or_ink = args.type

    input_dir = Path("../input/vcid-validation")
    output_dir = Path(f"../output/vcid-tile-images-{mask_or_ink}-{roi_size}-{stride}")

    output_dir.mkdir(parents=True, exist_ok=True)

    fragment_id_list_all = []
    x_pos_list_all = []
    y_pos_list_all = []
    fold_id_list_all = []
    for fold_id in tqdm(range(num_folds), dynamic_ncols=True):

        # get scanning positions
        if mask_or_ink == "mask":
            mask = np.array(Image.open(str(input_dir / f"fold{fold_id}" / "mask.png")))
            mask[mask!=0] = 1
        else:
            mask = np.array(Image.open(str(input_dir / f"fold{fold_id}" / "inklabels.png")))
            mask[mask!=0] = 1
        x_pos_list, y_pos_list = get_scanning_positions(mask, roi_size, stride)

        # Drawing to check the cropped position
        draw_scanning_positions(input_dir, output_dir, fold_id, x_pos_list, y_pos_list, roi_size)

        if fold_id == 0:
            fragment_id = 1
        elif fold_id == 4:
            fragment_id = 3
        else:
            fragment_id = 2

        fragment_id_list_all.extend([fragment_id] * len(x_pos_list))
        x_pos_list_all.extend(x_pos_list)
        y_pos_list_all.extend(y_pos_list)
        fold_id_list_all.extend([fold_id] * len(x_pos_list))

    df = pd.DataFrame({
        "fragment_id": fragment_id_list_all,
        "x_pos": x_pos_list_all,
        "y_pos": y_pos_list_all,
        "fold": fold_id_list_all,
    })
    df.to_csv(str(output_dir / "train.csv"), index=False)