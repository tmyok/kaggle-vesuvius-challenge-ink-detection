import albumentations as albu
import cv2
import numpy as np
import PIL.Image as Image
import polars as pl
import psutil
import torch

from lightning import LightningDataModule
from pathlib import Path
from tqdm import tqdm
from typing import Optional

def get_transform_unet3d():

    transform_param = []
    transform_param.append(albu.HorizontalFlip(p=0.5))
    transform_param.append(albu.VerticalFlip(p=0.5))
    transform_param.append(albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=180, p=1.0))
    transform_param.append(albu.CoarseDropout(max_holes=50, max_height=16, max_width=16, p=1.0))

    return albu.Compose(transform_param)

def get_transform_mit():

    transform_param = []
    transform_param.append(albu.HorizontalFlip(p=0.5))
    transform_param.append(albu.VerticalFlip(p=0.5))
    transform_param.append(albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=180, p=1.0))
    transform_param.append(albu.Affine(p=1.0))
    transform_param.append(albu.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0))
    transform_param.append(albu.CoarseDropout(max_holes=50, max_height=8, max_width=8, p=1.0))

    return albu.Compose(transform_param)

def normalize_img(img):

    img = img.astype(np.float32)
    if img.sum() != 0:
        img[img > 0] = (img[img > 0] - img[img > 0].mean()) / (img[img > 0].std() + 1e-8)

    return img

class MyDataset(object):
    def __init__(
        self,
        datalist,
        z_start,
        z_end,
        img_depth,
        num_instances,
        inklabels_ignore_edge,
        mask_ignore_edge,
        is_train,
        is3d,
        transforms=None,
    ):
        self.datalist = datalist
        self.z_start = z_start
        self.z_end = z_end
        self.img_depth = img_depth
        self.num_instances = num_instances
        self.inklabels_ignore_edge = inklabels_ignore_edge
        self.mask_ignore_edge = mask_ignore_edge
        self.is_train = is_train
        self.is3d = is3d
        self.transforms = transforms

    def __getitem__(self, index):

        data = self.datalist[index]

        img = (data["image"] / 255.0).astype(np.float32)
        mask = data["mask"]
        inklabels = data["inklabels"]
        ignore_inklabels = data["ignore_inklabels"]

        # preprocess
        if self.is_train:
            z_start_list = np.random.choice(np.arange(0, self.z_end-self.z_start+1), size=self.num_instances, replace=False)
        else:
            z_start_list = np.linspace(0, self.z_end-self.z_start, num=self.num_instances, dtype=np.int32)
        img_bag = self.preprocess_img(img, z_start_list)

        # transform
        if self.transforms:
            N, d, h, w = img_bag.shape
            img_bag = img_bag.reshape(N*d, h, w) # NDHW -> (ND)HW
            img_bag = img_bag.transpose(1, 2, 0) # (ND)HW -> HW(ND)

            # temp_mask へ mask, inklabels, ignore_inklabelsを合成
            temp_mask = np.zeros((h, w, 3), dtype=np.uint8)
            temp_mask[:, :, 0] = mask
            temp_mask[:, :, 1] = inklabels
            temp_mask[:, :, 2] = ignore_inklabels

            transformed = self.transforms(image=img_bag, mask=temp_mask)
            img_bag = transformed["image"]
            mask = transformed["mask"][..., 0]
            inklabels = transformed["mask"][..., 1]
            ignore_inklabels = transformed["mask"][..., 2]

            img_bag = img_bag.transpose(2, 0, 1) # HW(ND) -> (ND)HW
            nd, h, w = img_bag.shape
            assert nd == N*d
            img_bag = img_bag.reshape(N, d, h, w) # (ND)HW -> NDHW

        # normalize
        for i in range(img_bag.shape[0]):
            img_bag[i] = normalize_img(img_bag[i])

        # ignore region
        ignore_mask = np.ones_like(mask)

        # apply mask region
        ignore_mask[mask.astype(np.uint8) == 0] = 0

        # calcualte inklabels edge
        if self.is_train:
            kernel = np.ones((3, 3), np.uint8)
            dilated_inklabels = cv2.dilate(inklabels.astype(np.uint8), kernel, iterations=self.inklabels_ignore_edge)
            eroded_inklabels = cv2.erode(inklabels.astype(np.uint8), kernel, iterations=self.inklabels_ignore_edge)
            edge_inklabels = dilated_inklabels - eroded_inklabels
            ignore_mask[edge_inklabels > 0] = 0

            # apply ignore_inklabels
            ignore_mask[ignore_inklabels > 0] = 0

        # calculate mask edge
        if self.is_train:
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=self.mask_ignore_edge)
            ignore_mask[eroded_mask == 0] = 0

        if self.is3d:
            img_bag = img_bag[np.newaxis, ...]# NDHW -> CNDHW
            img_bag = img_bag.transpose(1, 0, 2, 3, 4) # CNDHW -> NCDHW

        inklabels = inklabels[np.newaxis, ...] # HW -> CHW

        return {
            "input": torch.as_tensor(img_bag, dtype=torch.float32),
            "target": torch.as_tensor(inklabels, dtype=torch.float32),
            "ignore_mask": ignore_mask,
        }

    def __len__(self):
        return len(self.datalist)

    def crop_instance(self, image, z_start):

        output_list = []
        for i in range(z_start, z_start + self.img_depth):
            output_list.append(image[i, ...])
        output = np.stack(output_list)

        return output

    def preprocess_img(self, image, z_start_list):

        output_list = []
        for z_start in z_start_list:
            output_list.append(
                self.crop_instance(image, z_start),
            )
        output = np.stack(output_list)

        return output

def read_image_mask(
    image_path, mask_path, inklabels_path, ignore_inklabels_path,
    resize_ratio, roi_size,
    z_min, z_max,
):

    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, dsize=None, fx=1.0/resize_ratio, fy=1.0/resize_ratio, interpolation=cv2.INTER_AREA)

    inklabels = cv2.imread(inklabels_path, 0)
    inklabels = cv2.resize(inklabels, dsize=None, fx=1.0/resize_ratio, fy=1.0/resize_ratio, interpolation=cv2.INTER_AREA)

    ignore_inklabels = cv2.imread(ignore_inklabels_path, 0)
    ignore_inklabels = cv2.resize(ignore_inklabels, dsize=None, fx=1.0/resize_ratio, fy=1.0/resize_ratio, interpolation=cv2.INTER_AREA)

    mask = np.pad(mask, [(0, roi_size), (0, roi_size)], constant_values=0)
    mask = np.where(mask > 0, 1, 0).astype(np.int8)

    inklabels = np.pad(inklabels, [(0, roi_size), (0, roi_size)], constant_values=0)
    inklabels = np.where(inklabels > 0, 1, 0).astype(np.int8)

    ignore_inklabels = np.pad(ignore_inklabels, [(0, roi_size), (0, roi_size)], constant_values=0)
    ignore_inklabels = np.where(ignore_inklabels > 0, 1, 0).astype(np.int8)

    images = []
    for i in range(z_min, z_max):
        image = np.array(Image.open(f"{image_path}/{i:02}.tif"), dtype=np.uint16)
        image = (image >> 8).astype(np.uint8)
        image = cv2.resize(image, dsize=None, fx=1.0/resize_ratio, fy=1.0/resize_ratio, interpolation=cv2.INTER_AREA)
        image = np.pad(image, [(0, roi_size), (0, roi_size)], constant_values=0)
        images.append(image)
    images = np.stack(images)

    return images, mask, inklabels, ignore_inklabels

def crop_images(
    image, mask, inklabels, ignore_inklabels,
    x_pos_list, y_pos_list,
    resize_ratio,
    roi_size,
):

    dataset_list = []
    for x_pos, y_pos in zip(x_pos_list, y_pos_list):

        y_start = y_pos // resize_ratio
        y_end = y_pos // resize_ratio + roi_size
        x_start = x_pos // resize_ratio
        x_end = x_pos // resize_ratio + roi_size

        dataset_list.append({
            "image": image[:, y_start:y_end, x_start:x_end],
            "mask": mask[y_start:y_end, x_start:x_end],
            "inklabels": inklabels[y_start:y_end, x_start:x_end],
            "ignore_inklabels": ignore_inklabels[y_start:y_end, x_start:x_end],
        })

    return dataset_list

def get_train_val_dataset(
    df,
    image_path,
    ignore_inklabels_path,
    num_folds, target_fold,
    resize_ratio, roi_size,
    z_min, z_max,
):
    train_datalist = []
    val_datalist = []
    for fold in tqdm(range(num_folds), desc="crop_image", dynamic_ncols=True):

        records = df.filter(pl.col("fold") == fold)

        image, mask, inklabels, ignore_inklabels = read_image_mask(
            image_path=str(Path(image_path) / f"fold{fold}" / "surface_volume"),
            mask_path=str(Path(image_path) / f"fold{fold}" / "mask.png"),
            inklabels_path=str(Path(image_path) / f"fold{fold}" / "inklabels.png"),
            ignore_inklabels_path=str(Path(ignore_inklabels_path) / f"fold{fold}" / "removed_area.png"),
            resize_ratio=resize_ratio, roi_size=roi_size,
            z_min=z_min, z_max=z_max,
        )

        dataset_list = crop_images(
            image=image, mask=mask, inklabels=inklabels, ignore_inklabels=ignore_inklabels,
            x_pos_list = records["x_pos"].to_list(),
            y_pos_list = records["y_pos"].to_list(),
            resize_ratio=resize_ratio,
            roi_size=roi_size,
        )

        if fold == target_fold:
            val_datalist.extend(dataset_list)
        else:
            train_datalist.extend(dataset_list)

    return train_datalist, val_datalist

class MyDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg

        self.train_datalist = None
        self.val_datalist = None

    def setup(
        self,
        stage: Optional[str] = None
    ):
        df = pl.read_csv(self.cfg.train_csv_path)

        # load train/val dataset
        if self.train_datalist is None and self.val_datalist is None:
            self.train_datalist, self.val_datalist = get_train_val_dataset(
                df=df,
                image_path=self.cfg.input_dir,
                ignore_inklabels_path=self.cfg.ignore_inklabel_dir,
                num_folds=self.cfg.num_folds,
                target_fold=self.cfg.fold,
                resize_ratio=self.cfg.model.resize_ratio,
                roi_size=self.cfg.model.image_size // self.cfg.model.resize_ratio,
                z_min=self.cfg.model.z_start,
                z_max=self.cfg.model.z_end+self.cfg.model.img_depth+1,
            )
            print("train:", len(self.train_datalist))
            print("val:", len(self.val_datalist))

    def train_dataloader(self):

        return torch.utils.data.DataLoader(
            dataset = MyDataset(
                datalist=self.train_datalist,
                z_start=self.cfg.model.z_start,
                z_end=self.cfg.model.z_end,
                img_depth=self.cfg.model.img_depth,
                num_instances=self.cfg.model.num_instances,
                inklabels_ignore_edge=self.cfg.inklabels_ignore_edge,
                mask_ignore_edge=self.cfg.mask_ignore_edge,
                is_train=True,
                is3d=True if self.cfg.model.is3d else False,
                transforms=get_transform_unet3d() if self.cfg.model.is3d else get_transform_mit(),
            ),
            batch_size=self.cfg.model.batch_size,
            num_workers=min([
                self.cfg.num_workers,
                psutil.cpu_count(logical=False)
            ]),
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):

        return torch.utils.data.DataLoader(
            dataset = MyDataset(
                datalist=self.val_datalist,
                z_start=self.cfg.model.z_start,
                z_end=self.cfg.model.z_end,
                img_depth=self.cfg.model.img_depth,
                num_instances=self.cfg.model.num_instances,
                inklabels_ignore_edge=self.cfg.inklabels_ignore_edge,
                mask_ignore_edge=self.cfg.mask_ignore_edge,
                is_train=False,
                is3d=True if self.cfg.model.is3d else False,
            ),
            batch_size=self.cfg.model.batch_size,
            num_workers=min([
                self.cfg.num_workers,
                psutil.cpu_count(logical=False)
            ]),
            shuffle=False,
            pin_memory=True,
        )
