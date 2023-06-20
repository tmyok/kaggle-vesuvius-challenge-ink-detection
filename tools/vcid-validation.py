import gc
import numpy as np
import PIL.Image as Image
import tifffile as tiff

from pathlib import Path
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 10000000000  # Ignore PIL warnings about large images

INPUT_DIR = Path("../input/vesuvius-challenge-ink-detection/train/")
OUTPUT_DIR = Path("../output/vcid-validation/")

def crop_image(input_dir, output_dir, desc):

    output_data_dir_2a = Path(output_dir / "fold1")
    output_data_dir_2a.mkdir(parents=True, exist_ok=True)

    output_data_dir_2b = Path(output_dir / "fold2")
    output_data_dir_2b.mkdir(parents=True, exist_ok=True)

    output_data_dir_2c = Path(output_dir / "fold3")
    output_data_dir_2c.mkdir(parents=True, exist_ok=True)

    y_start_2b = 6144
    y_end_2b = 10624

    # mask
    mask = np.array(Image.open(str(input_dir / "mask.png")))
    mask[mask>0] = 255
    Image.fromarray(mask[:y_start_2b, :]).save(str(output_data_dir_2a / "mask.png"))
    Image.fromarray(mask[y_start_2b:y_end_2b, :]).save(str(output_data_dir_2b / "mask.png"))
    Image.fromarray(mask[y_end_2b:, :]).save(str(output_data_dir_2c / "mask.png"))

    # inklabels
    inklabels_img = np.array(Image.open(str(input_dir / "inklabels.png")))
    inklabels_img[inklabels_img>0] = 255
    Image.fromarray(inklabels_img[:y_start_2b, :]).save(str(output_data_dir_2a / "inklabels.png"))
    Image.fromarray(inklabels_img[y_start_2b:y_end_2b, :]).save(str(output_data_dir_2b / "inklabels.png"))
    Image.fromarray(inklabels_img[y_end_2b:, :]).save(str(output_data_dir_2c / "inklabels.png"))

    # ir
    ir = np.array(Image.open(str(input_dir / "ir.png")))
    Image.fromarray(ir[:y_start_2b, :]).save(str(output_data_dir_2a / "ir.png"))
    Image.fromarray(ir[y_start_2b:y_end_2b, :]).save(str(output_data_dir_2b / "ir.png"))
    Image.fromarray(ir[y_end_2b:, :]).save(str(output_data_dir_2c / "ir.png"))

    # Crop the image
    output_volume_dir_2a = Path(output_data_dir_2a / "surface_volume")
    output_volume_dir_2a.mkdir(parents=True, exist_ok=True)
    output_volume_dir_2b = Path(output_data_dir_2b / "surface_volume")
    output_volume_dir_2b.mkdir(parents=True, exist_ok=True)
    output_volume_dir_2c = Path(output_data_dir_2c / "surface_volume")
    output_volume_dir_2c.mkdir(parents=True, exist_ok=True)

    image_path_list = sorted(list(Path(input_dir / "surface_volume").glob('*.tif')))
    for image_path in tqdm(image_path_list, total=len(image_path_list), desc=desc, dynamic_ncols=True):

        img = tiff.imread(str(image_path))
        tiff.imwrite(str(output_volume_dir_2a / image_path.name), img[:y_start_2b, :])
        tiff.imwrite(str(output_volume_dir_2b / image_path.name), img[y_start_2b:y_end_2b, :])
        tiff.imwrite(str(output_volume_dir_2c / image_path.name), img[y_end_2b:, :])
        del img
        gc.collect()

def copy_image(input_dir, output_dir, desc):

    output_dir.mkdir(parents=True, exist_ok=True)

    # mask
    mask = np.array(Image.open(str(input_dir / "mask.png")))
    mask[mask>0] = 255
    Image.fromarray(mask).save(str(output_dir / "mask.png"))

    # inklabels
    inklabels = np.array(Image.open(str(input_dir / "inklabels.png")))
    inklabels[inklabels>0] = 255
    Image.fromarray(inklabels).save(str(output_dir / "inklabels.png"))

    # ir
    ir = np.array(Image.open(str(input_dir / "ir.png")))
    Image.fromarray(ir).save(str(output_dir / "ir.png"))

    # surface_volume
    image_path_list = sorted(list(Path(input_dir / "surface_volume").glob('*.tif')))
    output_volume_dir = Path(output_dir / "surface_volume")
    output_volume_dir.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(image_path_list, total=len(image_path_list), desc=desc, dynamic_ncols=True):

        img = tiff.imread(str(image_path))
        tiff.imwrite(str(output_volume_dir / image_path.name), img)
        del img
        gc.collect()

# main
if __name__ == '__main__':

    # fragment1 -> fold0
    copy_image(
        input_dir=INPUT_DIR / "1",
        output_dir=OUTPUT_DIR / "fold0",
        desc="fragment1",
    )

    # fragment2 -> fold1, fold2, fold3
    crop_image(
        input_dir=INPUT_DIR / "2",
        output_dir=OUTPUT_DIR,
        desc="fragment2",
    )

    # fragment3 -> fold4
    copy_image(
        input_dir=INPUT_DIR / "3",
        output_dir=OUTPUT_DIR / "fold4",
        desc="fragment3",
    )