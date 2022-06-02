import json
import sys
from cv2 import GaussianBlur, merge
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os




def extract(source):
    image = source.copy()
    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    th = 20  # defines the value below which a pixel is considered "black"
    black_pixels = np.where(
        (image[:, :, 0] < th) &
        (image[:, :, 1] < th) &
        (image[:, :, 2] < th)
    )

    # set those pixels to white
    image[black_pixels] = [255, 255, 255]

    not_white_pixels = np.where(
        (image[:, :, 0] != 255) |
        (image[:, :, 1] != 255) |
        (image[:, :, 2] != 255)
    )

    image[not_white_pixels] = [0, 0, 0]
    image = ~image

    output = cv2.connectedComponentsWithStats(image[:, :, 0], 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    components = []
    # loop over the number of unique connected component labels
    for i in range(0, numLabels):
        # skip background
        if i == 0:
            continue

        # get stats
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        components.append(source[y:y+h, x:x+w])

    return components


def remove_computed(path):
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            os.remove('{}/{}'.format(path, filename))


def compute_merged(path):
    images = []
    # extracted_images = []
    for filename in os.listdir(path):
        if filename != "merged.png" and filename != "best.png":
            images.append(
                [cv2.imread('{}/{}'.format(path, filename)), filename])

    # for image, filename in raw_images:
    #     extracted = extract(image)
    #     for image in extracted:
    #         extracted_images.append([image, filename])

    # maxW = max(map(lambda item: np.shape(item[0])[0], images))
    # maxH = max(map(lambda item: np.shape(item[0])[1], images))

    maxH = int(np.mean([np.shape(item[0])[1] for item in images]))
    maxW = int(np.mean([np.shape(item[0])[0] for item in images]))


    merged = np.zeros((maxW, maxH, 3))
    del_index = []
    for index, tup in enumerate(images):
        img, filename = tup
        if img.shape[0] < maxW/2 or img.shape[1] < maxH/2:
            del_index.append(index)
        else:
            padded = cv2.resize(img, dsize=(
                np.shape(merged)[1], np.shape(merged)[0]))
            merged = np.add(padded, merged)

    for index in sorted(del_index, reverse=True):
        del images[index]

    # grayscale
    merged = np.mean(merged, axis=2)

    # normalization
    merged_norm = np.zeros(np.shape(merged))
    merged = cv2.normalize(merged, merged_norm, 0, 255, cv2.NORM_MINMAX)
    merged_norm = np.round(merged_norm, 0).astype(np.uint8)

    thresh = np.max(merged_norm) - 30
    mask = np.zeros_like(merged_norm)
    mask[merged_norm >= thresh] = 255

    # get most similar image
    best_image = None
    best_score = 0
    best_file = None
    for img, filename in images:
        img = cv2.resize(img, dsize=(np.shape(mask)[1], np.shape(mask)[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        score = np.sum(mask * img)
        if score > best_score:
            best_score = score
            best_image = img
            best_file = filename

    # add gaussian blur
    gauss_size = int(min(np.shape(merged)[:2]) * 0.05)
    if gauss_size % 2 == 0:
        gauss_size += 1
    img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 0)


    merged = cv2.GaussianBlur(merged_norm, (9, 9), 0)

    # closing
    merged = cv2.dilate(merged, np.ones((15, 15), np.uint8))
    merged = cv2.erode(merged, np.ones((15, 15), np.uint8), iterations=2)

    # plt.imshow(best_image)
    best_image_cmap = cv2.applyColorMap(best_image, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite('{}/best.png'.format(path), best_image_cmap)
    # plt.imshow(merged_norm)
    merged_norm_cmap = cv2.applyColorMap(merged_norm, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite('{}/merged.png'.format(path), merged_norm_cmap)

    return best_file




def main():
    for period in sorted(os.listdir('results-raw')):
        for p in sorted(os.listdir('results-raw/{}'.format(period))):
            # if os.path.isdir('results-raw/{}/{}'.format(period, p)) and any(f.endswith(".jpg") or f.endswith(".png") for f in os.listdir('results-raw/{}/{}'.format(period, p))):
            if os.path.isdir('results-raw/{}/{}'.format(period, p)) and os.listdir('results-raw/{}/{}'.format(period, p)):
                print("merging {} | {}".format(period, p))
                best_file: str = compute_merged('results-raw/{}/{}'.format(period, p))
                if not best_file.startswith('BEST'):
                    os.rename('results-raw/{}/{}/{}'.format(period, p, best_file),
                              'results-raw/{}/{}/BEST-{}'.format(period, p, best_file))
                    os.rename('results/{}/{}/{}'.format(period, p, best_file),
                              'results/{}/{}/BEST-{}'.format(period, p, best_file))

if __name__ == '__main__':
    main()
