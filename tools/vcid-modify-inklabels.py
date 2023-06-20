import cv2
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy.stats import norm
from tqdm import tqdm


def remove_small_inklabels(inklabels, openning_iterations=20, small_area_threshold=50000):
    """
    Remove small ink labels based on given threshold.
    """
    kernel = np.ones((3, 3), np.uint8)
    inklabels_new = cv2.morphologyEx(inklabels, cv2.MORPH_OPEN, kernel, iterations=openning_iterations)

    removed_area = inklabels - inklabels_new
    inklabels = inklabels_new

    label_num, label_img, label_stats, _ = cv2.connectedComponentsWithStats(inklabels)
    label_areas = label_stats[:, cv2.CC_STAT_AREA]

    small_label_list = [i for i in range(1, label_num) if label_areas[i] < small_area_threshold]

    for label in small_label_list:
        removed_area[label_img == label] = 255

    inklabels[removed_area != 0] = 0

    return inklabels, removed_area


def extract_roi(image, labels, stats, i, margin):
    """
    Extract region of interest (ROI) for the given label.
    """
    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    x1 = max(0, x-margin)
    y1 = max(0, y-margin)
    x2 = min(image.shape[1], x+w+margin)
    y2 = min(image.shape[0], y+h+margin)

    image_roi = image[y1:y2, x1:x2]
    labels_roi = labels[y1:y2, x1:x2]

    image_roi = cv2.GaussianBlur(image_roi, (15, 15), 0)
    return image_roi, labels_roi, x1, y1, x2, y2


def compute_histograms(image, labels):
    """Compute histograms for foreground and background regions."""
    fg = image[labels > 0]
    bg = image[labels == 0]
    hist_fg, bins_fg = np.histogram(fg, bins=256, density=True)
    hist_bg, bins_bg = np.histogram(bg, bins=256, density=True)
    return hist_fg, bins_fg, hist_bg, bins_bg

def find_threshold(image, img_thr=180):

    image = image[image < img_thr]
    retval, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return retval

def create_mask(image, labels, max_val):
    """Create mask of pixels that are greater than the max value."""
    mask = image > max_val
    mask[labels == 0] = 0
    return mask

def modify_mask(mask, removed_area):
    mask = mask.copy()
    mask[removed_area > 0] = 0
    return mask


def visualize_results(image, labels, mask, modified_labels, hist_fg, bins_fg, hist_bg, bins_bg, threshold, output_path):
    """Visualize the results in a subplot and save the figure."""
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_color[mask > 0] = [255, 0, 0]

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    ax[0].imshow(image, cmap='gray', vmin = 0, vmax = 255)
    ax[0].set_title("ir")

    ax[1].imshow(image_color)
    ax[1].set_title("ir mask")

    ax[2].imshow(labels, cmap='gray')
    ax[2].set_title("inklabels")

    ax[3].imshow(modified_labels, cmap='gray')
    ax[3].set_title("modified inklabels")

    ax[4].bar(bins_fg[:-1], hist_fg, width=1, color='r')
    ax[4].bar(bins_bg[:-1], hist_bg, width=1, color='b')
    ax[4].axvline(x=threshold, color='k', linestyle='--')
    ax[4].set_xlim(0, 255)
    ax[4].set_title("histogram")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def remove_background(inklabels, output_dir, margin, desc):
    """
    Remove background-like inklabels.
    """
    removed_area = np.zeros_like(inklabels)
    label_num, _, label_stats, _ = cv2.connectedComponentsWithStats(inklabels)

    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True, parents=True)

    for i in tqdm(range(1, label_num), desc=desc):
        roi_image, roi_labels, x1, y1, x2, y2 = extract_roi(ir, inklabels, label_stats, i, margin)
        fg_hist, fg_bins, bg_hist, bg_bins = compute_histograms(roi_image, roi_labels)
        threshold = find_threshold(roi_image)
        mask = create_mask(roi_image, roi_labels, threshold)

        removed_area[y1:y2, x1:x2][mask > 0] = 255

        modified_labels_roi = modify_mask(roi_labels, mask)
        visualize_results(roi_image, roi_labels, mask, modified_labels_roi, fg_hist, fg_bins, bg_hist, bg_bins, threshold, str(debug_dir / f"label{i}.png"))

    inklabels[removed_area != 0] = 0

    return inklabels, removed_area


if __name__ == '__main__':

    margin = 50

    output_dir = Path(f"../output/vcid-modify-inklabels")
    output_dir.mkdir(exist_ok=True, parents=True)

    for fold_id in range(5):

        input_dir = Path(f"../input/vcid-validation/fold{fold_id}")

        output_fold_dir = output_dir / f"fold{fold_id}"
        output_fold_dir.mkdir(exist_ok=True, parents=True)

        # load images
        inklabels = cv2.imread(str(input_dir / "inklabels.png"), 0)
        ir = cv2.imread(str(input_dir / "ir.png"), 0)

        inklabels_color = cv2.cvtColor(inklabels, cv2.COLOR_GRAY2BGR) # for debug

        # step1: remove small inklabels area
        inklabels_step1, removed_area1 = remove_small_inklabels(inklabels=inklabels)

        # debug
        inklabels_color[removed_area1 != 0] = [0, 0, 255]
        cv2.imwrite(str(output_fold_dir / "inklabels_step1.png"), inklabels_color)
        cv2.imwrite(str(output_fold_dir / "removed_area1.png"), removed_area1)

        # step2: Obtain removal regions based on the luminance distribution within small areas.
        inklabels_step2, removed_area2 = remove_background(inklabels=inklabels_step1, output_dir=output_fold_dir, margin=margin, desc=f"fold{fold_id}")

        # debug
        inklabels_color[removed_area2 != 0] = [0, 255, 0]
        cv2.imwrite(str(output_fold_dir / "inklabels_step2.png"), inklabels_color)
        cv2.imwrite(str(output_fold_dir / "removed_area2.png"), removed_area2)

        # whole removed area
        removed_area = removed_area1 + removed_area2
        removed_area[removed_area > 0] = 255

        # inklabels without removed area
        inklabels[removed_area != 0] = 0

        cv2.imwrite(str(output_fold_dir / "inklabels.png"), inklabels)
        cv2.imwrite(str(output_fold_dir / "removed_area.png"), removed_area)