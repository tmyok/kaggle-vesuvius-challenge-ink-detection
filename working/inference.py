import argparse
import cv2
import numpy as np
import torch

from pathlib import Path

from validation import seed_everything, inference_fold

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    mask_path = Path(args.mask_path)

    output_dir = Path(f"../output/inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(42)

    device = torch.device("cuda:0")
    thr = 0.57
    num_folds = 5

    predictions = []
    for fold_id in range(0, num_folds):

        pred = inference_fold(
            fold_id=fold_id,
            image_path=str(image_path),
            mask_path=str(mask_path),
            model_dir="../input/vcid-trained-weight",
            device=device,
            is_rot_tta=True,
            is_flip_tta=False,
        )
        predictions.append(pred)

    predictions = np.array(predictions)
    output = np.mean(predictions, axis=0)

    pred_image = np.zeros(output.shape, dtype=np.uint8)
    pred_image[output > thr] = 1

    job_type = __file__.split("/")[-1].split(".")[0].split("_")[0]
    cv2.imwrite(str(output_dir / f"result.png"), (output * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / f"result_thr.png"), (pred_image * 255).astype(np.uint8))
