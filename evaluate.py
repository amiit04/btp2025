#!/usr/bin/env python3
"""
evaluate_deid.py  (debug-friendly, recursive file discovery)

Usage example:
    python evaluate_deid.py --orig_dir ./input --deid_dirs ./no_smooth,./with_smooth --out my_eval
"""
import os, sys, argparse
from pathlib import Path
import math
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd

from facenet_pytorch import MTCNN, InceptionResnetV1
from skimage.metrics import structural_similarity as ssim
from torchvision.models import inception_v3
import torchvision.transforms as T
from scipy import linalg

# ---------------------------
# helpers: IoU
# ---------------------------
def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter_w = max(0, x2 - x1); inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0, (a[2] - a[0])) * max(0, (a[3] - a[1]))
    area_b = max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# ---------------------------
# File discovery helpers (recursive, case-insensitive)
# ---------------------------
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

def find_image_files_recursive(folder: Path):
    """Return sorted list of Path objects for images found recursively."""
    folder = Path(folder)
    if not folder.exists():
        return []
    found = []
    for p in folder.rglob('*'):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in IMAGE_EXTS:
            found.append(p.resolve())
    # Also include files without extension if the name is numeric (rare)
    for p in folder.rglob('*'):
        if not p.is_file(): continue
        if p.suffix == '' and p.stem.isdigit():
            found.append(p.resolve())
    # deduplicate and sort
    found = sorted(list(dict.fromkeys(found)))
    return found

def find_deid_candidates_for_orig(deid_dir: Path, orig_path: Path):
    """Prefer exact stem match (any extension) in deid_dir (recursive); fallback to prefix match."""
    stem = orig_path.stem
    if not deid_dir.exists():
        return []
    # search recursively
    candidates = []
    for p in deid_dir.rglob('*'):
        if not p.is_file(): continue
        if p.stem == stem:
            candidates.append(p.resolve())
    if candidates:
        return sorted(candidates)
    # fallback: name startswith stem
    for p in deid_dir.rglob('*'):
        if not p.is_file(): continue
        if p.name.startswith(stem):
            candidates.append(p.resolve())
    return sorted(candidates)

# ---------------------------
# FID utilities (unchanged)
# ---------------------------
# ---------------------------
# Robust InceptionFeatureExtractor (works across torchvision versions)
# ---------------------------
import warnings
try:
    # torchvision >= 0.13
    from torchvision.models import Inception_V3_Weights
    _HAS_INCEPTION_WEIGHTS = True
except Exception:
    _HAS_INCEPTION_WEIGHTS = False

# ---------- robust Inception feature extractor (use create_feature_extractor) ----------
from torchvision.models.feature_extraction import create_feature_extractor

# Robust Inception feature extractor (must call inception_v3 with aux_logits=True)
from torchvision.models.feature_extraction import create_feature_extractor
import warnings

class InceptionFeatureExtractor(torch.nn.Module):
    """
    Loads torchvision.inception_v3 with weights in a way compatible across versions,
    forcing aux_logits=True when weights are provided, and extracts the avgpool features
    (2048-d).
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        # Load inception_v3 using the modern weights API if available; ensure aux_logits=True
        try:
            # Newer torchvision: use Weights enum
            from torchvision.models import Inception_V3_Weights
            weights = Inception_V3_Weights.DEFAULT
            base = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        except Exception:
            # Fallback older API: pretrained arg (still require aux_logits=True)
            try:
                base = inception_v3(pretrained=True, aux_logits=True, transform_input=False)
            except Exception as e:
                # Last-resort fallback: try without weights (will be uninitialized)
                warnings.warn(f"Failed to load inception with pretrained weights: {e}. "
                              "Falling back to uninitialized inception (results will be meaningless).")
                base = inception_v3(weights=None, aux_logits=True, transform_input=False)

        base.to(device).eval()

        # Create a feature extractor that returns the avgpool output.
        # avgpool node name is usually "avgpool", but can vary across versions.
        # We'll try common node names, else search for an AdaptiveAvgPool2d module.
        candidate_nodes = ["avgpool", "AuxLogits", "Mixed_7c", None]
        return_nodes = None
        # Prefer the canonical 'avgpool' if present
        try:
            return_nodes = {"avgpool": "feat"}
            self.feat_extractor = create_feature_extractor(base, return_nodes=return_nodes).to(device)
        except Exception:
            # search for a module which is an AdaptiveAvgPool2d and use its name
            found_name = None
            for name, module in base.named_modules():
                from torch.nn import AdaptiveAvgPool2d
                if isinstance(module, AdaptiveAvgPool2d):
                    found_name = name
                    break
            if found_name is not None:
                return_nodes = {found_name: "feat"}
                self.feat_extractor = create_feature_extractor(base, return_nodes=return_nodes).to(device)
            else:
                # As a last fallback, use children() slicing (less reliable)
                warnings.warn("Could not find avgpool node using create_feature_extractor; "
                              "falling back to slicing children() (fragile).")
                self.features = torch.nn.Sequential(*list(base.children())[:-1]).to(device)
                self.features.eval()
                self.feat_extractor = None

        if self.feat_extractor is not None:
            self.feat_extractor.eval()

    def forward(self, x):
        """
        Input: x (B,3,299,299) tensor (normalized).
        Returns: (B,2048) numpy-ready tensor
        """
        with torch.no_grad():
            if self.feat_extractor is not None:
                out = self.feat_extractor(x)
                # out is a dict: pick the only value (mapped to "feat")
                feats = list(out.values())[0]
            else:
                # fallback path — use sliced features
                feats = self.features(x)
            # feats expected shape (B,2048,1,1) -> flatten
            if feats.ndim == 4:
                feats = feats.reshape(feats.size(0), -1)
            elif feats.ndim == 2:
                # already flattened (B,2048)
                pass
            else:
                raise RuntimeError(f"Unexpected feats shape from Inception: {tuple(feats.shape)}")
        return feats


# ---------------------------
# Inception preprocessing transform (needed by compute_activations)
# ---------------------------
def get_inception_transform():
    """
    InceptionV3 expects 299x299 input and ImageNet normalization.
    This transform returns a tensor ready for the InceptionFeatureExtractor.
    """
    return T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
    ])


def compute_activations(images_pil, model, batch_size=32, device='cpu'):
    """
    images_pil: list of PIL.Image
    model: InceptionFeatureExtractor instance (on device)
    returns: numpy array (N,2048)
    """
    transform = get_inception_transform()
    acts = []
    model.eval()

    n = len(images_pil)
    if n == 0:
        return np.zeros((0,2048), dtype=np.float32)

    for i in range(0, n, batch_size):
        batch_imgs = images_pil[i:i+batch_size]
        # build tensor batch
        try:
            tensors = torch.stack([transform(im) for im in batch_imgs], dim=0).to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to transform images to tensor: {e}")

        # SANITY CHECK: ensure shape is (B,3,H,W)
        if tensors.ndim != 4 or tensors.size(1) != 3:
            raise RuntimeError(f"Expected image tensor of shape (B,3,H,W) but got {tuple(tensors.shape)}. "
                               "This usually means one of the inputs is not an image (or transform produced the wrong shape).")

        with torch.no_grad():
            feats = model(tensors).cpu().numpy()   # should be (B,2048)
        acts.append(feats)

    return np.concatenate(acts, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)

def compute_fid_for_crops(orig_crops, deid_crops, device='cpu', batch_size=32):
    if len(orig_crops) == 0 or len(deid_crops) == 0:
        return float('nan')
    extractor = InceptionFeatureExtractor(device=device)
    acts_orig = compute_activations(orig_crops, extractor, batch_size=batch_size, device=device)
    acts_deid = compute_activations(deid_crops, extractor, batch_size=batch_size, device=device)
    mu1, sigma1 = acts_orig.mean(axis=0), np.cov(acts_orig, rowvar=False)
    mu2, sigma2 = acts_deid.mean(axis=0), np.cov(acts_deid, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

# ---------------------------
# Main evaluation (mostly same, uses Path lists)
# ---------------------------
def evaluate(orig_dir, deid_dirs, iou_thresh=0.5, device=None, out_prefix="eval_results"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    mtcnn = MTCNN(keep_all=True, device=device)
    embed_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    orig_dir = Path(orig_dir).resolve()
    deid_dirs = [Path(d).resolve() for d in deid_dirs]

    print("Working directory:", Path.cwd().resolve())
    print("Orig dir (abs):", orig_dir)
    for d in deid_dirs:
        print("Deid dir (abs):", d)

    orig_paths = find_image_files_recursive(orig_dir)
    print(f"Found {len(orig_paths)} image files under orig_dir. Examples (up to 10):")
    for p in orig_paths[:10]:
        print("  ", p)

    # Prepare storage
    per_folder_rows = {str(d): [] for d in deid_dirs}
    all_orig_crops = []
    all_deid_crops_by_folder = {str(d): [] for d in deid_dirs}

    # Process each original file path
    for orig_path in tqdm(orig_paths, desc="Images"):
        img_name = orig_path.name

        try:
            orig_img = Image.open(orig_path).convert('RGB')
        except Exception as e:
            print("Skip", orig_path, "error:", e)
            continue

        boxes_orig, _ = mtcnn.detect(orig_img)
        orig_crops = []
        if boxes_orig is not None:
            for b in boxes_orig:
                x1,y1,x2,y2 = [int(round(x)) for x in b]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(orig_img.width, x2); y2 = min(orig_img.height, y2)
                if x2 <= x1 or y2 <= y1: continue
                crop = orig_img.crop((x1,y1,x2,y2))
                orig_crops.append(((x1,y1,x2,y2), crop))
                all_orig_crops.append(crop)

        for dpath in deid_dirs:
            candidates = find_deid_candidates_for_orig(dpath, orig_path)
            if len(candidates) == 0:
                per_folder_rows[str(dpath)].append({
                    "image": orig_path.name,
                    "n_faces_orig": len(orig_crops),
                    "n_faces_deid": 0,
                    "n_matched": 0,
                    "mean_cosine": float('nan'),
                    "mean_ssim": float('nan'),
                    "matched_file": None
                })
                continue

            deid_path = candidates[0]
            try:
                deid_img = Image.open(deid_path).convert('RGB')
            except Exception as e:
                print("Skip deid", deid_path, "error:", e)
                per_folder_rows[str(dpath)].append({
                    "image": orig_path.name,
                    "n_faces_orig": len(orig_crops),
                    "n_faces_deid": 0,
                    "n_matched": 0,
                    "mean_cosine": float('nan'),
                    "mean_ssim": float('nan'),
                    "matched_file": str(deid_path)
                })
                continue

            boxes_deid, _ = mtcnn.detect(deid_img)
            deid_crops = []
            if boxes_deid is not None:
                for b in boxes_deid:
                    x1,y1,x2,y2 = [int(round(x)) for x in b]
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(deid_img.width, x2); y2 = min(deid_img.height, y2)
                    if x2 <= x1 or y2 <= y1: continue
                    crop = deid_img.crop((x1,y1,x2,y2))
                    deid_crops.append(((x1,y1,x2,y2), crop))
                    all_deid_crops_by_folder[str(dpath)].append(crop)

            matched_pairs = []
            if len(orig_crops) > 0 and len(deid_crops) > 0:
                used = set()
                for i, (box_o, crop_o) in enumerate(orig_crops):
                    best_j = -1; best_iou = 0.0
                    for j, (box_d, crop_d) in enumerate(deid_crops):
                        if j in used: continue
                        iouscore = iou_xyxy(box_o, box_d)
                        if iouscore > best_iou:
                            best_iou = iouscore; best_j = j
                    if best_j >= 0 and best_iou >= iou_thresh:
                        used.add(best_j)
                        matched_pairs.append((crop_o, deid_crops[best_j][1]))

            cosines = []; ssims = []
            emb_transform = T.Compose([T.Resize((160,160)), T.ToTensor()])
            for (crop_o, crop_d) in matched_pairs:
                t_o = emb_transform(crop_o).unsqueeze(0).to(device)
                t_d = emb_transform(crop_d).unsqueeze(0).to(device)
                with torch.no_grad():
                    v_o = embed_model(t_o).cpu().numpy().flatten()
                    v_d = embed_model(t_d).cpu().numpy().flatten()
                num = float(np.dot(v_o, v_d))
                den = float(np.linalg.norm(v_o) * np.linalg.norm(v_d) + 1e-10)
                cosines.append(num / den)

                size = (256,256)
                go = crop_o.convert('L').resize(size, Image.BILINEAR)
                gd = crop_d.convert('L').resize(size, Image.BILINEAR)
                a = np.array(go, dtype=np.float32) / 255.0
                b = np.array(gd, dtype=np.float32) / 255.0
                try:
                    s = ssim(a, b, data_range=1.0)
                except Exception:
                    s = float('nan')
                ssims.append(s)

            mean_cosine = float(np.nanmean(cosines)) if len(cosines) > 0 else float('nan')
            mean_ssim = float(np.nanmean(ssims)) if len(ssims) > 0 else float('nan')

            per_folder_rows[str(dpath)].append({
                "image": orig_path.name,
                "n_faces_orig": len(orig_crops),
                "n_faces_deid": len(deid_crops),
                "n_matched": len(matched_pairs),
                "mean_cosine": mean_cosine,
                "mean_ssim": mean_ssim,
                "matched_file": str(deid_path.name)
            })

    # Summaries + CSVs
    summaries = []
    for dpath in deid_dirs:
        rows = pd.DataFrame(per_folder_rows[str(dpath)])
        mean_cos = float(rows['mean_cosine'].dropna().mean()) if 'mean_cosine' in rows and not rows['mean_cosine'].dropna().empty else float('nan')
        mean_ssim = float(rows['mean_ssim'].dropna().mean()) if 'mean_ssim' in rows and not rows['mean_ssim'].dropna().empty else float('nan')
        fid_val = compute_fid_for_crops(all_orig_crops, all_deid_crops_by_folder[str(dpath)], device=device, batch_size=32)
        summary = {
            "deid_folder": str(dpath),
            "mean_image_level_cosine": mean_cos,
            "mean_image_level_ssim": mean_ssim,
            "fid_faces": fid_val,
            "n_images": len(rows),
            "n_total_orig_faces": len(all_orig_crops),
            "n_total_deid_faces": len(all_deid_crops_by_folder[str(dpath)])
        }
        summaries.append(summary)
        out_csv = f"{out_prefix}_{dpath.name}_per_image.csv"
        rows.to_csv(out_csv, index=False)
        print("Wrote per-image CSV:", out_csv)

    summary_df = pd.DataFrame(summaries)
    summary_csv = f"{out_prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("Wrote summary CSV:", summary_csv)
    print("\n=== Summary (per deid folder) ===")
    print(summary_df.to_string(index=False))
    return summary_df

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_dir", required=True, help="Folder with original images")
    parser.add_argument("--deid_dirs", required=True, help="Comma-separated list of deid folders (e.g. ./no_smooth,./with_smooth)")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for face matching")
    parser.add_argument("--device", default=None, help="cuda or cpu (default auto)")
    parser.add_argument("--out", default="eval_results", dest="out_prefix", help="Output prefix for CSVs")
    args = parser.parse_args()

    deid_list = [p.strip() for p in args.deid_dirs.split(",") if p.strip()]
    evaluate(args.orig_dir, deid_list, iou_thresh=args.iou_thresh, device=args.device, out_prefix=args.out_prefix)
