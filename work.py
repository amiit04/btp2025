"""
Batch-process images named 001.jpg .. 100.jpg (or configurable range) in a folder.
Produces two output folders:
 - <out_dir_without> containing 001_without_smoothing.jpg, ...
 - <out_dir_with>    containing 001_with_smoothing.jpg, ...
"""

import os
import argparse
import hashlib
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image, ImageFilter
import cv2
import torch
from facenet_pytorch import MTCNN

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return arr[:, :, ::-1].copy()  # RGB->BGR

def expand_bbox(bbox: Tuple[int,int,int,int], img_w: int, img_h: int, pad_factor: float = 0.15) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad_x = int(w * pad_factor)
    pad_y = int(h * pad_factor)
    nx1 = max(0, int(x1 - pad_x))
    ny1 = max(0, int(y1 - pad_y))
    nx2 = min(int(img_w), int(x2 + pad_x))
    ny2 = min(int(img_h), int(y2 + pad_y))
    return (nx1, ny1, nx2, ny2)

from anonymize_face import anonymize_face 
# from https://github.com/hanweikung/nullface

def deidentify(face_img: Image.Image) -> Image.Image:
    """
    Must return a PIL.Image (RGB) of same or resizable size.
    """
    
    folder_path = "temp"
    file_name = "img.jpg"
    full_path = os.path.join(folder_path, file_name)
    os.makedirs(folder_path, exist_ok=True)
    face_img.save(full_path)
    
    # model from nullface
    output_img = anonymize_face(
        image_path=full_path,
        mask_image_path="",
        sd_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
        insightface_model_path="~/.insightface",
        device_num=0,
        guidance_scale=10.0,
        num_diffusion_steps=100,
        eta=1.0,
        skip=70,
        ip_adapter_scale=1.0,
        id_emb_scale=1.0,
        output_log_file="log.txt",
        det_thresh=0.1,
        det_size=640,
        seed=0,
        mask_delay_steps=10,
    )

    if output_img:
        return output_img
    else:
        print("Face could not be detected. Please check the output log file for more details.")
        return face_img

# MTCNN helpers
def make_mtcnn(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)
    return mtcnn

def detect_faces_mtcnn(mtcnn: MTCNN, img_pil: Image.Image) -> List[Dict]:
    boxes, probs = mtcnn.detect(img_pil)  # boxes is (N,4) or None
    dets = []
    if boxes is None:
        return dets
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        dets.append({"bbox": (int(x1), int(y1), int(x2), int(y2)), "score": float(probs[i]) if probs is not None else None})
    return dets

# -------------------------
# Merging helpers
def paste_direct(target_pil: Image.Image, face_img: Image.Image, bbox: Tuple[int,int,int,int]) -> Image.Image:
    out = target_pil.copy()
    x1, y1, x2, y2 = map(int, bbox)
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    face_resized = face_img.resize((w, h), Image.LANCZOS)
    mask = Image.new("L", (w, h), 255)
    out.paste(face_resized, (x1, y1), mask)
    return out

def seamless_clone_merge(target_bgr: np.ndarray, face_img: Image.Image, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    src = pil_to_cv2(face_img.resize((w, h), Image.LANCZOS))  # BGR
    mask = 255 * np.ones((h, w), dtype=np.uint8)
    center = (x1 + w//2, y1 + h//2)
    try:
        cloned = cv2.seamlessClone(src, target_bgr, mask, center, cv2.NORMAL_CLONE)
        return cloned
    except Exception as e:
        # fallback to direct paste on BGR
        out = target_bgr.copy()
        out[y1:y2, x1:x2] = src
        return out

# -------------------------
# Process single image
def process_single_image(img_path: str, mtcnn: MTCNN, deidentify_func, expand_pad: float = 0.15, min_face_area: int = 1000):
    """
    Returns tuple (PIL_image_without_smoothing, numpy_bgr_with_smoothing)
    """
    image_pil = Image.open(img_path).convert("RGB")
    img_w, img_h = image_pil.size

    detections = detect_faces_mtcnn(mtcnn, image_pil)
    print(f"  Detected {len(detections)} faces in {os.path.basename(img_path)}")

    out_direct = image_pil.copy()
    out_smooth_bgr = pil_to_cv2(image_pil.copy())

    processed = 0
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        x1n, y1n, x2n, y2n = expand_bbox((x1, y1, x2, y2), img_w, img_h, pad_factor=expand_pad)
        area = (x2n - x1n) * (y2n - y1n)
        if area < min_face_area:
            print(f"Skipping tiny face bbox {x1n,y1n,x2n,y2n}")
            continue
        face_crop = image_pil.crop((x1n, y1n, x2n, y2n))
        try:
            deid_face = deidentify_func(face_crop)
        except Exception as e:
            print(f"deidentify failed for face #{i}: {e}")
            continue
        # resize/convert if needed
        if deid_face.size != face_crop.size:
            deid_face = deid_face.resize(face_crop.size, Image.LANCZOS)
        if deid_face.mode != "RGB":
            deid_face = deid_face.convert("RGB")

        out_direct = paste_direct(out_direct, deid_face, (x1n, y1n, x2n, y2n))
        out_smooth_bgr = seamless_clone_merge(out_smooth_bgr, deid_face, (x1n, y1n, x2n, y2n))
        processed += 1

    return out_direct, out_smooth_bgr, processed


# Batch process
def batch_process(src_dir: str, start_idx: int, end_idx: int, padding: int, out_without: str, out_with: str, mtcnn: MTCNN, deidentify_func, expand_pad: float = 0.15, min_face_area: int = 1000):
    os.makedirs(out_without, exist_ok=True)
    os.makedirs(out_with, exist_ok=True)

    for i in range(start_idx, end_idx + 1):
        name = str(i).zfill(padding)
        filename = f"{name}.jpg"
        src_path = os.path.join(src_dir, filename)
        if not os.path.isfile(src_path):
            print(f"[WARN] File not found: {src_path} — skipping.")
            continue
        try:
            out_direct_pil, out_smooth_bgr, processed_count = process_single_image(src_path, mtcnn, deidentify_func, expand_pad, min_face_area)
            without_name = f"{name}_without_smoothing.jpg"
            with_name = f"{name}_with_smoothing.jpg"
            out_direct_pil.save(os.path.join(out_without, without_name), quality=95)
            cv2.imwrite(os.path.join(out_with, with_name), out_smooth_bgr)
            print(f"[OK] {filename} -> {without_name} ({processed_count} faces), {with_name}")
        except Exception as e:
            print(f"[ERR] processing {src_path} failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch deidentify images named 001..100 style using MTCNN.")
    parser.add_argument("--src_dir", required=True, help="Folder containing input images (001.jpg ...).")
    parser.add_argument("--out_without", default="outputs_without_smoothing", help="Output folder for without-smoothing images.")
    parser.add_argument("--out_with", default="outputs_with_smoothing", help="Output folder for with-smoothing images.")
    parser.add_argument("--start", type=int, default=1, help="Start index (default 1).")
    parser.add_argument("--end", type=int, default=100, help="End index (default 100).")
    parser.add_argument("--padding", type=int, default=3, help="Zero-padding for filenames (default 3 -> 001).")
    parser.add_argument("--min_area", type=int, default=1000, help="Minimum face bbox area to process.")
    args = parser.parse_args()

    mtcnn = make_mtcnn()
    print("MTCNN device:", mtcnn.device)
    batch_process(args.src_dir, args.start, args.end, args.padding, args.out_without, args.out_with, mtcnn, deidentify, expand_pad=0.15, min_face_area=args.min_area)

if __name__ == "__main__":
    main()