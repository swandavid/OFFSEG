# -*- coding: utf-8 -*-
"""
Pipeline for semantic segmentation and traversability classification.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import torch
from fast_pytorch_kmeans import KMeans
import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import glob

# Force TensorFlow to use CPU only (disable Metal/MPS)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')
tf.config.set_visible_devices([], 'TPU')

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Failed to load image Python extension.*')
warnings.filterwarnings('ignore', message='.*No training configuration found.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(123)
pal = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """Custom DepthwiseConv2D to ignore 'groups' param for legacy models."""
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

def img_seg(im, net, to_tensor, pal):
    """Run segmentation model and return colorized prediction and mask."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).to(device)
    out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
    pred = pal[out]
    out = cv2.cvtColor(out.astype('uint8'), cv2.COLOR_GRAY2BGR)
    return pred, out

def palette_lst(masked_img, n_classes=4):
    """Cluster image colors using KMeans and return list of masked images."""
    h, w = masked_img.shape[:2]
    print(f"Processing image of size {h}x{w}")
    
    # Check if image has any non-zero pixels
    if np.all(masked_img == 0):
        print("Warning: Input image is all zeros")
        return []
    
    data = pd.DataFrame(masked_img.reshape(-1, 3), columns=['R', 'G', 'B'], dtype=np.float32)
    torch_kmeans = KMeans(n_clusters=n_classes, mode='euclidean', verbose=0)
    input_data = data[['R', 'G', 'B']].values.astype(np.float32)
    input = torch.tensor(input_data)
    data['Cluster'] = torch_kmeans.fit_predict(input).cpu().numpy()
    palette = torch_kmeans.centroids.cpu().numpy()
    palette_list = [tuple(map(int, color)) for color in palette]
    img_c = np.array([palette_list[x] for x in data['Cluster']]).reshape(h, w, 3)
    img_lst = []
    for i in range(1, n_classes):
        mask = (img_c == palette_list[i]).all(axis=2)
        j = np.zeros_like(img_c)
        j[mask] = palette_list[i]
        # Only add non-empty masks
        if np.any(mask):
            print(f"Added mask {i} with {np.sum(mask)} pixels")
            img_lst.append(j)
        else:
            print(f"Skipping empty mask {i}")
    
    print(f"Returning {len(img_lst)} valid masks")
    return img_lst

def trav_cut(img, lpool):
    """Extract traversable section from RGB image using mask."""
    lpool = np.where(lpool != 1, 255, lpool)
    lpool = cv2.resize(lpool, (img.shape[1], img.shape[0]))
    dst = cv2.addWeighted(lpool, 1, img, 1, 0)
    h, w, c = img.shape
    image_bgra = np.concatenate([dst, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    white = np.all(dst == [0, 0, 0], axis=-1)
    image_bgra[white, -1] = 0
    masked_img = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
    return masked_img

def mask_pred(img_lst, model):
    """Predict mask class for each image using the classification model, batched and parallelized."""
    import gc
    gc.collect()
    mask_class = []
    valid_imgs = []
    for idx, i in enumerate(img_lst):
        if i is None or i.size == 0 or np.all(i == 0):
            continue
        if i.dtype != np.uint8:
            i = i.astype(np.uint8)
        if len(i.shape) != 3 or i.shape[2] != 3:
            continue
        valid_imgs.append(i)
    if not valid_imgs:
        return mask_class
    # Batch resize and preprocess
    batch = np.stack([cv2.resize(i, (224, 224)) for i in valid_imgs]).astype(np.float32)
    batch = (batch / 127.0) - 1
    # Run prediction in parallel (if model supports it)
    preds = model.predict(batch, verbose=0)
    mask_class = list(np.argmax(preds, axis=1) + 4)
    return mask_class

def mask_comb(newpool, img_lst, mask_class):
    """Combine mask predictions into a single mask."""
    newpool = cv2.cvtColor(newpool.astype('float32'), cv2.COLOR_BGR2GRAY)
    if not mask_class:
        print("Warning: No valid mask predictions found, returning original pool")
        return newpool.astype(np.uint8)
    
    for i, img in enumerate(img_lst):
        if i >= len(mask_class):
            break
        ima = cv2.resize(img, (newpool.shape[1], newpool.shape[0]))
        ima = cv2.cvtColor(ima.astype('float32'), cv2.COLOR_BGR2GRAY)
        ima[ima > 0] = mask_class[i]
        bm = (mask_class[i] - ima) / mask_class[i]
        fp = np.multiply(newpool, bm)
        newpool = np.add(fp, ima)
        newpool = np.asarray(newpool, dtype=np.uint8)
    return newpool

def col_seg(image, pool, model):
    """Run color segmentation and mask prediction pipeline."""
    travcut = trav_cut(image, pool)
    msk_img = palette_lst(travcut)
    predicts = mask_pred(msk_img, model)
    return msk_img, predicts

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, default='bisenetv2')
    parser.add_argument('--weight-path', type=str, default='/Users/davidswan/Documents/personal-projects/costmap-gen/OFFSEG/offseg-pretrained-weights/Pre-trained_RUGD/model_final.pth')
    parser.add_argument('--img-path', type=str, default='/Users/davidswan/Documents/personal-projects/costmap-gen/OFFSEG/data/Rellis_3D_image_example/pylon_camera_node/frame000002-1581624652_949.jpg')
    parser.add_argument('--img-dir', type=str, default=None, help='Directory of images to process')
    parser.add_argument('--output-path', type=str, default='/Users/davidswan/Documents/personal-projects/costmap-gen/OFFSEG/data/Rellis_3D_image_example/pylon_camera_node/frame000002-1581624652_949_output.jpg')
    args = parser.parse_args()

    # Suppress warnings for clean benchmarking
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    timings = {}
    t0 = time.time()
    cfg = cfg_factory[args.model]
    timings['config'] = time.time() - t0

    t1 = time.time()
    model = tf.keras.models.load_model(
        '/Users/davidswan/Documents/personal-projects/costmap-gen/OFFSEG/offseg-pretrained-weights/Classification_keras/keras_model.h5',
        custom_objects={'DepthwiseConv2D': CompatibleDepthwiseConv2D}
    )
    timings['load_classification_model'] = time.time() - t1

    t2 = time.time()
    net = model_factory[cfg.model_type](4)
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    net.eval()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    net.to(device)
    timings['load_segmentation_model'] = time.time() - t2

    t3 = time.time()
    to_tensor = T.ToTensor(
        mean=(0.3257, 0.3690, 0.3223),
        std=(0.2112, 0.2148, 0.2115),
    )
    timings['init_misc'] = time.time() - t3

    # Directory mode
    if args.img_dir:
        img_files = sorted(glob.glob(os.path.join(args.img_dir, '*.jpg')) + glob.glob(os.path.join(args.img_dir, '*.png')))
        if not img_files:
            print(f"No images found in directory: {args.img_dir}")
            return
        output_dir = args.output_path if os.path.isdir(args.output_path) else os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        total_times = []
        for img_path in img_files:
            per_img_t0 = time.time()
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Could not load image from {img_path}")
                continue
            max_dim = 640
            if max(image.shape[:2]) > max_dim:
                scale = max_dim / max(image.shape[:2])
                image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
            im = image.copy()[:, :, ::-1]
            pred, pool = img_seg(im, net, to_tensor, pal)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_pred_path = os.path.join(output_dir, f"{base}_seg_pred.jpg")
            cv2.imwrite(out_pred_path, pred)
            pool1 = pool.copy()
            msk_img, predicts = col_seg(image, pool, model)
            final_pool = mask_comb(pool1, msk_img, predicts)
            out_final_path = os.path.join(output_dir, f"{base}_output.jpg")
            cv2.imwrite(out_final_path, pal[final_pool])
            per_img_time = time.time() - per_img_t0
            total_times.append(per_img_time)
            print(f"Processed {img_path} in {per_img_time:.3f}s. Output: {out_final_path}")
        print(f"\nProcessed {len(img_files)} images in {sum(total_times):.2f}s. Avg: {np.mean(total_times):.3f}s/image")
        return

    # Single image mode (default)
    image = cv2.imread(args.img_path)
    if image is None:
        print(f"Error: Could not load image from {args.img_path}")
        exit(1)
    max_dim = 640
    if max(image.shape[:2]) > max_dim:
        scale = max_dim / max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
    im = image.copy()[:, :, ::-1]
    pred, pool = img_seg(im, net, to_tensor, pal)
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(args.output_path, pred)
    pool1 = pool.copy()
    msk_img, predicts = col_seg(image, pool, model)
    final_pool = mask_comb(pool1, msk_img, predicts)
    output_filename = os.path.basename(args.img_path).replace('.jpg', '_output.jpg')
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, pal[final_pool])
    print(f"\nProcessing complete. Output saved to: {output_path}")
    # Print timings for single image
    total_time = time.time() - t0
    print("\n--- Pipeline Timing (seconds) ---")
    for k, v in timings.items():
        print(f"{k:40s}: {v:.4f}")
    print(f"{'total':40s}: {total_time:.4f}")

if __name__ == '__main__':
    main()
