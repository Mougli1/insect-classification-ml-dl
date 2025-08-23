#!/usr/bin/env python3
"""
predict_aligned.py - Prédiction sur les images de test avec features flexibles
==============================================================================

Script pour traiter les images de test (251-347), extraire les features
et faire des prédictions avec le modèle entraîné.

Features extraites (configurables):
- 1 ratio de pixels (ratio_area)
- 15 features de couleur (min/max/mean/median/std pour R/G/B)
- 2 features shape/symmetry (inscribed_circle_radius_norm, symmetry_loss_min)
- 2 features morphologiques (hu_1, eccentricity)
- N features texture/couleur configurables (GLCM, histogramme, HSV)

Usage:
    python predict_aligned.py --start 251 --end 347
    python predict_aligned.py --start 251 --end 347 --model_dir output_models --output submission.csv
"""

import numpy as np
import pandas as pd
from skimage import io, measure, feature, morphology
from scipy import optimize, ndimage as ndi, stats
import cv2
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import joblib
import argparse
import warnings
import time
warnings.filterwarnings('ignore')

# Reproductibilité
np.random.seed(42)

# ===============================================
# CONFIGURATION DES FEATURES - MODIFIER ICI !
# =====================================================


# Features GLCM (Gray Level Co-occurrence Matrix)
GLCM_FEATURES = {
    'homogeneity': False,     # Homogénéité (inverse du contraste)
    'contrast': False,       # Contraste local
    'dissimilarity': False,  # Dissimilarité
    'energy': True,         # Energie (ASM)
    'correlation': False,    # Corrélation
    'ASM': False,           # Angular Second Moment (= energy)
}

# Features d'histogramme (sur l'image en niveaux de gris)
HISTOGRAM_FEATURES = {
    'hist_mean': False,      # Moyenne
    'hist_std': False,       # Écart-type
    'hist_variance': True,  # Variance
    'hist_skewness': False,  # Asymétrie (skewness)
    'hist_kurtosis': False,  # Aplatissement (kurtosis)
    'hist_entropy': False,   # Entropie
    'hist_energy': False,    # Énergie
    'hist_mode': False,      # Mode (valeur la plus fréquente)
}

# Moments statistiques d'histogramme
HISTOGRAM_MOMENTS = {
    'moment1': False,        # Premier moment (= moyenne)
    'moment2': False,        # Deuxième moment central (= variance)
    'moment3': False,        # Troisième moment central
    'moment4': False,        # Quatrième moment central
}

# Features de couleur par canal HSV
HSV_FEATURES = {
    'hue_mean': False,       # Moyenne de teinte
    'hue_std': False,        # Écart-type de teinte
    'saturation_mean': False, # Moyenne de saturation
    'saturation_std': False,  # Écart-type de saturation
    'value_mean': False,     # Moyenne de valeur (luminosité)
    'value_std': False,      # Écart-type de valeur
}

# Garde toujours hue_skewness actif (feature originale)
KEEP_ORIGINAL_HUE_SKEWNESS = True

# ==========
# ===============================================

# ====== FONCTIONS DE FEATURES (identiques à features.py) ======

def rotate_image(theta_degree, xc, yc, arr):
    """Rotation d'image (depuis lab01)"""
    theta = theta_degree * np.pi / 180
    s1, s2 = arr.shape[:2]

    c_rot_grid, r_rot_grid = np.meshgrid(np.arange(s2), np.arange(s1))
    r_rot_flat = r_rot_grid.flatten()
    c_rot_flat = c_rot_grid.flatten()

    r_src = np.cos(theta) * (r_rot_flat - xc) + np.sin(theta) * (c_rot_flat - yc) + xc
    c_src = -np.sin(theta) * (r_rot_flat - xc) + np.cos(theta) * (c_rot_flat - yc) + yc

    r_src_idx = (np.round(r_src)).astype(int)
    c_src_idx = (np.round(c_src)).astype(int)

    valid_mask = (r_src_idx >= 0) & (r_src_idx < s1) & \
                 (c_src_idx >= 0) & (c_src_idx < s2)

    if arr.ndim == 3:
        rot_arr = np.zeros_like(arr)
        for i in range(arr.shape[2]):
            channel_arr = np.zeros((s1,s2))
            channel_arr[r_rot_flat[valid_mask], c_rot_flat[valid_mask]] = arr[r_src_idx[valid_mask], c_src_idx[valid_mask], i]
            rot_arr[:,:,i] = channel_arr.reshape(s1,s2)
    else:
        rot_arr = np.zeros_like(arr)
        rot_arr[r_rot_flat[valid_mask], c_rot_flat[valid_mask]] = arr[r_src_idx[valid_mask], c_src_idx[valid_mask]]
        rot_arr = rot_arr.reshape(s1,s2)

    return rot_arr

def load_and_clean_mask(mask_p, img_p, img_id):
    """Version améliorée avec nettoyage morphologique pour réduire le bruit"""
    try:
        img_arr_bgr = cv2.imread(img_p)
        if img_arr_bgr is None: 
            return None, None, None, 0, None
        img_arr_rgb = cv2.cvtColor(img_arr_bgr, cv2.COLOR_BGR2RGB)

        mask_arr = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask_arr is None: 
            return None, None, None, 0, None

        with Image.open(img_p) as temp_pil_img:
            original_dims = temp_pil_img.size

        bin_mask = (mask_arr >= 1).astype(np.uint8)
        orig_bug_px_count = np.sum(bin_mask)
        
        # Nettoyage morphologique pour réduire le bruit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remplissage des trous
        bin_mask_fill = ndi.binary_fill_holes(bin_mask).astype(np.uint8)
        lbl_mask, n_lbl = ndi.label(bin_mask_fill)

        if n_lbl == 0:
            return None, None, None, orig_bug_px_count, original_dims

        props = measure.regionprops(lbl_mask)
        if not props:
            return None, None, None, orig_bug_px_count, original_dims

        mask_h, mask_w = bin_mask_fill.shape
        non_border_props = []
        for p_item in props:
            min_r_p, min_c_p, max_r_p, max_c_p = p_item.bbox
            touches_border = (min_r_p == 0 or min_c_p == 0 or max_r_p == mask_h or max_c_p == mask_w)
            if not touches_border: 
                non_border_props.append(p_item)

        if non_border_props:
            largest_comp = max(non_border_props, key=lambda p: p.area)
        else:
            largest_comp = max(props, key=lambda p: p.area)

        cleaned_mask = (lbl_mask == largest_comp.label).astype(np.uint8)
        min_r, min_c, max_r, max_c = largest_comp.bbox

        if not (min_r < max_r and min_c < max_c):
            return None, None, None, orig_bug_px_count, original_dims

        crop_img = img_arr_rgb[min_r:max_r, min_c:max_c, :]
        crop_mask = cleaned_mask[min_r:max_r, min_c:max_c]

        if crop_img.size == 0 or crop_mask.size == 0:
            return None, None, None, orig_bug_px_count, original_dims

        return crop_img, crop_mask, (min_r, min_c, max_r, max_c), orig_bug_px_count, original_dims

    except Exception as e:
        return None, None, None, 0, None

def centroid(mask_arr):
    """Calcule le centroïde du masque"""
    if mask_arr is None or not np.any(mask_arr): 
        return None
    mask_2d = mask_arr.astype(np.uint8)
    if mask_2d.ndim == 3 and mask_2d.shape[2] == 1: 
        mask_2d = mask_2d[:,:,0]
    if mask_2d.ndim != 2: 
        return None
    com = ndi.center_of_mass(mask_2d)
    return com[0], com[1]

def compute_best_inscribed_circle(mask_arr, image_id_for_log="N/A"):
    """Cercle inscrit robuste avec scipy.optimize.minimize"""
    if mask_arr is None or not np.any(mask_arr) or mask_arr.ndim != 2:
        return np.nan, np.nan, np.nan, np.nan

    dt = cv2.distanceTransform(mask_arr.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    if not np.any(dt) or dt.max() < 0.5:
        return np.nan, np.nan, np.nan, np.nan

    initial_center_yx = centroid(mask_arr)
    dt_flat_argmax = np.argmax(dt)
    best_y_dt, best_x_dt = np.unravel_index(dt_flat_argmax, dt.shape)
    best_radius_dt = dt[best_y_dt, best_x_dt]

    center_y_row, center_x_col, radius = np.nan, np.nan, np.nan

    if initial_center_yx is not None:
        cy_init_row, cx_init_col = initial_center_yx
        cy_init_row = np.clip(cy_init_row, 0, dt.shape[0] - 1)
        cx_init_col = np.clip(cx_init_col, 0, dt.shape[1] - 1)
        initial_guess = [cy_init_row, cx_init_col]

        def objective_func_circle(params_rc):
            r_param, c_param = params_rc
            r_idx = int(np.clip(round(r_param), 0, dt.shape[0] - 1))
            c_idx = int(np.clip(round(c_param), 0, dt.shape[1] - 1))
            return -dt[r_idx, c_idx]

        res = optimize.minimize(objective_func_circle, initial_guess, method='Nelder-Mead',
                       options={'maxiter': 100, 'xatol': 0.5, 'fatol': 0.1, 'adaptive': True})

        if res.success:
            opt_r_f, opt_c_f = res.x
            opt_r = np.clip(int(round(opt_r_f)), 0, dt.shape[0] - 1)
            opt_c = np.clip(int(round(opt_c_f)), 0, dt.shape[1] - 1)
            optimized_radius = dt[opt_r, opt_c]

            if optimized_radius > 0.1:
                center_y_row, center_x_col, radius = float(opt_r), float(opt_c), float(optimized_radius)
            else:
                r_idx_init = int(round(cy_init_row))
                c_idx_init = int(round(cx_init_col))
                radius_at_centroid = dt[r_idx_init, c_idx_init]
                if radius_at_centroid > 0.1:
                    center_y_row, center_x_col, radius = float(r_idx_init), float(c_idx_init), float(radius_at_centroid)
                else:
                    center_y_row, center_x_col, radius = float(best_y_dt), float(best_x_dt), float(best_radius_dt)
        else:
            r_idx_init = int(round(cy_init_row))
            c_idx_init = int(round(cx_init_col))
            radius_at_centroid = dt[r_idx_init, c_idx_init]
            if radius_at_centroid > 0.1:
                 center_y_row, center_x_col, radius = float(r_idx_init), float(c_idx_init), float(radius_at_centroid)
            else:
                center_y_row, center_x_col, radius = float(best_y_dt), float(best_x_dt), float(best_radius_dt)
    else:
        center_y_row, center_x_col, radius = float(best_y_dt), float(best_x_dt), float(best_radius_dt)

    ratio_circle_mask = np.nan
    if not np.isnan(radius) and radius > 0.1:
        area_mask_val = np.sum(mask_arr > 0)
        area_circle = np.pi * (radius**2)
        if area_mask_val > 0:
            ratio_circle_mask = area_circle / area_mask_val

    return center_y_row, center_x_col, radius, ratio_circle_mask

def create_symmetric_image_vectorized(img, sym_axis_col):
    """Crée l'image symétrique"""
    h, w = img.shape[:2]
    sym_img_out = np.zeros_like(img)
    for c_idx in range(w):
        reflected_c = int(round(2 * sym_axis_col - c_idx))
        if 0 <= reflected_c < w:
            if img.ndim == 3:
                sym_img_out[:, c_idx, :] = img[:, reflected_c, :]
            else:
                sym_img_out[:, c_idx] = img[:, reflected_c]
    return sym_img_out

def symmetry_loss(angle_degrees_param, image_arr_gray, center_of_rotation_rc, radius_roi):
    """Fonction de loss pour la symétrie"""
    current_angle = float(angle_degrees_param)
    img_gray_2d = image_arr_gray
    if img_gray_2d.ndim == 3 and img_gray_2d.shape[2] == 1: 
        img_gray_2d = img_gray_2d[:,:,0]

    rotated_img = rotate_image(current_angle, center_of_rotation_rc[0], center_of_rotation_rc[1], img_gray_2d)

    h_r, w_r = rotated_img.shape[:2]
    sym_axis_col_in_rotated_img = center_of_rotation_rc[1]
    sym_rotated = create_symmetric_image_vectorized(rotated_img, sym_axis_col_in_rotated_img)

    roi_center_r = center_of_rotation_rc[0]
    roi_center_c = center_of_rotation_rc[1]

    Y_r_grid, X_r_grid = np.ogrid[:h_r, :w_r]
    dist_roi = np.sqrt((Y_r_grid - roi_center_r)**2 + (X_r_grid - roi_center_c)**2)
    mask_roi = (dist_roi <= radius_roi).astype(np.uint8)

    if np.sum(mask_roi) < 10: 
        return 1e9

    diff_op = rotated_img.astype(np.float32) - sym_rotated.astype(np.float32)
    masked_diff_sq = (diff_op**2) * mask_roi
    sum_sq_diff_roi = np.sum(masked_diff_sq)
    norm_factor_roi = np.sum(mask_roi) * (255.0**2) + 1e-6

    return sum_sq_diff_roi / norm_factor_roi

def compute_best_symmetry_plane(image_arr, inscribed_circle_params, image_id_for_log="N/A"):
    """Plan de symétrie robuste avec scipy.optimize.minimize_scalar"""
    if image_arr is None or inscribed_circle_params[0] is None or np.isnan(inscribed_circle_params[0]):
        return np.nan, np.nan

    cic_center_row, cic_center_col, cic_radius, _ = inscribed_circle_params
    if cic_radius is None or np.isnan(cic_radius) or cic_radius <= 1:
        return np.nan, np.nan

    if image_arr.ndim == 3 and image_arr.shape[2] == 3:
        img_gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    elif image_arr.ndim == 2:
        img_gray = image_arr
    else:
        return np.nan, np.nan

    center_for_rotation_rc = (cic_center_row, cic_center_col)

    def loss_for_scalar_opt(angle):
        return symmetry_loss(angle, img_gray, center_for_rotation_rc, cic_radius)

    angles_to_try = np.arange(0, 180, 15)
    coarse_losses = []
    for angle_scan in angles_to_try:
        try: 
            coarse_losses.append(loss_for_scalar_opt(angle_scan))
        except Exception:
            coarse_losses.append(np.inf)

    if not coarse_losses or np.all(np.isinf(coarse_losses)) or np.all(np.isnan(coarse_losses)):
        return np.nan, np.nan

    best_initial_angle_idx = np.nanargmin(coarse_losses)
    best_initial_angle = angles_to_try[best_initial_angle_idx]

    bound_width = 20
    lower_bound = max(0.0, best_initial_angle - bound_width)
    upper_bound = min(179.9, best_initial_angle + bound_width)
    if lower_bound >= upper_bound:
        lower_bound = 0.0
        upper_bound = 179.9

    try:
        res = optimize.minimize_scalar(loss_for_scalar_opt, method='bounded', bounds=(lower_bound, upper_bound),
                              options={'maxiter': 30, 'xatol': 0.5})

        if res.success:
            min_loss_val = res.fun
            symmetry_score = max(0.0, 1.0 - min_loss_val)
            return res.x, symmetry_score
        else:
            min_loss_val_fb = coarse_losses[best_initial_angle_idx] if not np.isinf(coarse_losses[best_initial_angle_idx]) else 1.0
            symmetry_score_fb = max(0.0, 1.0 - min_loss_val_fb)
            return best_initial_angle, symmetry_score_fb
    except Exception:
        min_loss_val_fb = coarse_losses[best_initial_angle_idx] if not np.isinf(coarse_losses[best_initial_angle_idx]) else 1.0
        symmetry_score_fb = max(0.0, 1.0 - min_loss_val_fb)
        return best_initial_angle, symmetry_score_fb

def compute_color_features(image, mask):
    """Calcule les stats de couleur"""
    features = {}
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = image[:, :, i]
        masked_pixels = channel_data[mask == 1]
        
        if len(masked_pixels) > 0:
            features[f'color_min_{channel}'] = np.min(masked_pixels)
            features[f'color_max_{channel}'] = np.max(masked_pixels)
            features[f'color_mean_{channel}'] = np.mean(masked_pixels)
            features[f'color_median_{channel}'] = np.median(masked_pixels)
            features[f'color_std_{channel}'] = np.std(masked_pixels)
        else:
            for stat in ['min', 'max', 'mean', 'median', 'std']:
                features[f'color_{stat}_{channel}'] = 0
    return features

def compute_texture_color_features(img_rgb, mask):
    """
    Calcule les features de texture et couleur selon la configuration
    """
    features = {}
    
    try:
        # -------- GLCM Features --------
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # Appliquer le masque pour ne considérer que l'insecte
        masked_gray = img_gray.copy()
        masked_gray[mask == 0] = 0
        
        # Réduire l'échelle pour la matrice de co-occurrence
        img_low = (masked_gray / 16).astype(np.uint8)
        
        # Calculer GLCM
        glcm = feature.graycomatrix(img_low, distances=[2], angles=[0], 
                                   levels=32, symmetric=True, normed=True)
        
        # Extraire les features GLCM demandées
        for feat_name, is_active in GLCM_FEATURES.items():
            if is_active:
                # ASM est un alias pour energy
                prop_name = 'ASM' if feat_name in ['energy', 'ASM'] else feat_name
                value = feature.graycoprops(glcm, prop=prop_name)[0, 0]
                features[f'glcm_{feat_name}'] = float(value)
        
        # -------- Histogram Features --------
        # Pixels de l'insecte seulement
        gray_pixels = img_gray[mask == 1]
        
        if len(gray_pixels) > 0 and any(HISTOGRAM_FEATURES.values()):
            # Histogramme normalisé
            hist, _ = np.histogram(gray_pixels, bins=256, range=(0, 256))
            hist_norm = hist.astype(float) / hist.sum()
            
            if HISTOGRAM_FEATURES['hist_mean']:
                features['hist_mean'] = float(np.mean(gray_pixels))
            
            if HISTOGRAM_FEATURES['hist_std']:
                features['hist_std'] = float(np.std(gray_pixels))
            
            if HISTOGRAM_FEATURES['hist_variance']:
                features['hist_variance'] = float(np.var(gray_pixels))
            
            if HISTOGRAM_FEATURES['hist_skewness']:
                features['hist_skewness'] = float(stats.skew(gray_pixels))
            
            if HISTOGRAM_FEATURES['hist_kurtosis']:
                features['hist_kurtosis'] = float(stats.kurtosis(gray_pixels))
            
            if HISTOGRAM_FEATURES['hist_entropy']:
                # Entropie de Shannon
                hist_nonzero = hist_norm[hist_norm > 0]
                entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
                features['hist_entropy'] = float(entropy)
            
            if HISTOGRAM_FEATURES['hist_energy']:
                # Énergie = somme des carrés
                features['hist_energy'] = float(np.sum(hist_norm ** 2))
            
            if HISTOGRAM_FEATURES['hist_mode']:
                # Mode = valeur la plus fréquente
                features['hist_mode'] = float(np.argmax(hist))
        
        # -------- Histogram Moments --------
        if any(HISTOGRAM_MOMENTS.values()) and len(gray_pixels) > 0:
            mean = np.mean(gray_pixels)
            
            if HISTOGRAM_MOMENTS['moment1']:
                features['hist_moment1'] = float(mean)
            
            if HISTOGRAM_MOMENTS['moment2']:
                features['hist_moment2'] = float(np.mean((gray_pixels - mean) ** 2))
            
            if HISTOGRAM_MOMENTS['moment3']:
                features['hist_moment3'] = float(np.mean((gray_pixels - mean) ** 3))
            
            if HISTOGRAM_MOMENTS['moment4']:
                features['hist_moment4'] = float(np.mean((gray_pixels - mean) ** 4))
        
        # -------- HSV Features --------
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        if any(HSV_FEATURES.values()):
            hue = img_hsv[..., 0][mask == 1].astype(np.float32)
            saturation = img_hsv[..., 1][mask == 1].astype(np.float32)
            value = img_hsv[..., 2][mask == 1].astype(np.float32)
            
            if HSV_FEATURES['hue_mean'] and hue.size > 0:
                features['hue_mean'] = float(np.mean(hue))
            
            if HSV_FEATURES['hue_std'] and hue.size > 0:
                features['hue_std'] = float(np.std(hue))
            
            if HSV_FEATURES['saturation_mean'] and saturation.size > 0:
                features['saturation_mean'] = float(np.mean(saturation))
            
            if HSV_FEATURES['saturation_std'] and saturation.size > 0:
                features['saturation_std'] = float(np.std(saturation))
            
            if HSV_FEATURES['value_mean'] and value.size > 0:
                features['value_mean'] = float(np.mean(value))
            
            if HSV_FEATURES['value_std'] and value.size > 0:
                features['value_std'] = float(np.std(value))
        
        # -------- Original hue_skewness (toujours inclus) --------
        if KEEP_ORIGINAL_HUE_SKEWNESS:
            hue = img_hsv[..., 0][mask == 1].astype(np.float32)
            hue_skew = float(stats.skew(hue)) if hue.size > 20 else 0.0
            features['hue_skewness'] = hue_skew
        
        return features
    
    except Exception as e:
        print(f"[WARN] texture/color features: {e}")
        # Retourner des valeurs par défaut pour toutes les features actives
        default_features = {}
        
        for feat_name, is_active in GLCM_FEATURES.items():
            if is_active:
                default_features[f'glcm_{feat_name}'] = 0.0
        
        for feat_name, is_active in HISTOGRAM_FEATURES.items():
            if is_active:
                default_features[feat_name] = 0.0
        
        for feat_name, is_active in HISTOGRAM_MOMENTS.items():
            if is_active:
                default_features[f'hist_{feat_name}'] = 0.0
        
        for feat_name, is_active in HSV_FEATURES.items():
            if is_active:
                default_features[feat_name] = 0.0
        
        if KEEP_ORIGINAL_HUE_SKEWNESS:
            default_features['hue_skewness'] = 0.0
        
        return default_features

def compute_morphological_features(mask):
    """
    Renvoie : hu_1 et eccentricity
    """
    try:
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

        # --- Hu moment #1 ---
        m = cv2.moments(mask_uint8)
        hu = cv2.HuMoments(m).flatten()
        hu1 = -np.sign(hu[0]) * np.log10(np.abs(hu[0])) if hu[0] != 0 else 0.0

        # --- Eccentricity ---
        props = measure.regionprops(mask_uint8)
        if props:
            eccentricity = float(props[0].eccentricity)  # [0,1]
        else:
            eccentricity = 0.0

        return {
            'hu_1': float(hu1),
            'eccentricity': eccentricity
        }

    except Exception as e:
        print(f"[WARN] morph features: {e}")
        return {'hu_1': 0.0, 'eccentricity': 0.0}

def extract_features_single_test(args):
    """Extrait features pour une image de test"""
    image_id, test_dir = args
    
    try:
        image_path = f"{test_dir}/{image_id}.JPG"
        mask_path = f"{test_dir}/masks/binary_{image_id}.tif"
        
        # Preprocessing robuste avec nettoyage morphologique
        cropped_img, cleaned_mask, bbox, orig_bug_px, original_dims = load_and_clean_mask(mask_path, image_path, image_id)
        
        if cropped_img is None or cleaned_mask is None:
            print(f" Image {image_id}: preprocessing failed, using default features")
            # Retourner des features par défaut
            features = {'image_id': image_id}
            features['ratio_area'] = 0
            for stat in ['min', 'max', 'mean', 'median', 'std']:
                for ch in ['R', 'G', 'B']:
                    features[f'color_{stat}_{ch}'] = 0
            features['inscribed_circle_radius_norm'] = 0
            features['symmetry_loss_min'] = 1.0
            features['hu_1'] = 0.0
            features['eccentricity'] = 0.0
            
            # Features texture/couleur par défaut
            texture_color_default = compute_texture_color_features(
                np.zeros((10, 10, 3), dtype=np.uint8), 
                np.zeros((10, 10), dtype=np.uint8)
            )
            features.update(texture_color_default)
            
            return features
        
        features = {'image_id': image_id}
        
        # 1. Ratio de pixels
        if original_dims:
            total_pixels = original_dims[0] * original_dims[1]
            features['ratio_area'] = orig_bug_px / total_pixels if total_pixels > 0 else 0
        else:
            features['ratio_area'] = 0
        
        # 2. Features de couleur (15 features)
        color_features = compute_color_features(cropped_img, cleaned_mask)
        features.update(color_features)
        
        # 3. Shape & Symmetry (2 features)
        circle_params = compute_best_inscribed_circle(cleaned_mask, image_id)
        features['inscribed_circle_radius_norm'] = circle_params[2] / min(cleaned_mask.shape) if not np.isnan(circle_params[2]) else 0
        
        sym_angle, sym_score = compute_best_symmetry_plane(cropped_img, circle_params, image_id)
        features['symmetry_loss_min'] = 1.0 - sym_score if not np.isnan(sym_score) else 1.0
        features['symmetry_angle']  = float(sym_angle) if not np.isnan(sym_angle) else np.nan
        features['symmetry_score']  = float(sym_score) if not np.isnan(sym_score) else 0.0
        # 4. Features morphologiques (2 features)
        morph_features = compute_morphological_features(cleaned_mask)
        features.update(morph_features)
        
        # 5. Features texture/couleur flexibles
        texture_color_features = compute_texture_color_features(cropped_img, cleaned_mask)
        features.update(texture_color_features)
        
        return features
        
    except Exception as e:
        print(f"Image {image_id}: {str(e)}")
        # Retourner des features par défaut en cas d'erreur
        features = {'image_id': image_id}
        features['ratio_area'] = 0
        for stat in ['min', 'max', 'mean', 'median', 'std']:
            for ch in ['R', 'G', 'B']:
                features[f'color_{stat}_{ch}'] = 0
        features['inscribed_circle_radius_norm'] = 0
        features['symmetry_loss_min'] = 1.0
        features['hu_1'] = 0.0
        features['eccentricity'] = 0.0
        
        # Features texture/couleur par défaut
        texture_color_default = compute_texture_color_features(
            np.zeros((10, 10, 3), dtype=np.uint8), 
            np.zeros((10, 10), dtype=np.uint8)
        )
        features.update(texture_color_default)
        
        return features

# ====== FONCTIONS DE PRÉDICTION ======

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Prédiction sur images de test')
    parser.add_argument('--start', type=int, default=251, help='ID de début (défaut: 251)')
    parser.add_argument('--end', type=int, default=347, help='ID de fin (défaut: 347)')
    parser.add_argument('--test_dir', default='test', help='Dossier des images de test')
    parser.add_argument('--model_dir', default='output_models', help='Dossier contenant le modèle')
    parser.add_argument('--output', default='submission_predict.csv', help='Fichier de sortie')
    parser.add_argument('--batch_size', type=int, default=10, help='Taille des batchs pour le multiprocessing')
    return parser.parse_args()

def load_model_and_encoder(model_dir):
    """Charge le modèle et l'encodeur sauvegardés"""
    model_dir = Path(model_dir)
    pipeline_path = model_dir / 'best_pipeline.joblib'
    encoder_path = model_dir / 'label_encoder.joblib'
    
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline non trouvé: {pipeline_path}")
    
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encodeur non trouvé: {encoder_path}")
    
    print(f"Chargement du modèle...")
    pipeline = joblib.load(pipeline_path)
    label_encoder = joblib.load(encoder_path)
    print(f"   Modèle et encodeur chargés")
    
    return pipeline, label_encoder

def extract_features_batch(image_ids, test_dir):
    """Extrait les features pour un batch d'images"""
    args = [(img_id, test_dir) for img_id in image_ids]
    
    with Pool(processes=cpu_count()) as pool:
        features_list = list(tqdm(
            pool.imap(extract_features_single_test, args),
            total=len(image_ids),
            desc="Extraction features",
            unit="image"
        ))
    
    return features_list

def predict_batch(features_df, pipeline, label_encoder):
    """Fait des prédictions sur un batch de features"""
    # Préparer les features (exclure ID)
    X = features_df.drop(columns=['image_id'])
    
    # Prédictions
    y_pred_encoded = pipeline.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    
    # Créer DataFrame résultats
    results = pd.DataFrame({
        'ID': features_df['image_id'],
        'bug type': y_pred_labels
    })
    
    return results

def count_active_features():
    """Compte le nombre de features actives"""
    count = 0
    count += sum(GLCM_FEATURES.values())
    count += sum(HISTOGRAM_FEATURES.values())
    count += sum(HISTOGRAM_MOMENTS.values())
    count += sum(HSV_FEATURES.values())
    if KEEP_ORIGINAL_HUE_SKEWNESS:
        count += 1
    return count

def main():
    """Fonction principale"""
    print(" PRÉDICTION SUR IMAGES DE TEST - FEATURES FLEXIBLES")
    print("=" * 60)
    
    # Afficher les features actives
    active_texture_color = count_active_features()
    base_features = 20  # 1 ratio + 15 couleur + 2 shape/symmetry + 2 morpho
    total_features = base_features + active_texture_color
    
    print(f"Features totales: {total_features}")
    print(f"   • Base: {base_features} features")
    print(f"     - Ratio: 1 (ratio_area)")
    print(f"     - Couleur: 15 (min/max/mean/median/std × RGB)")
    print(f"     - Shape/Symmetry: 2 (inscribed_circle_radius_norm, symmetry_loss_min)")
    print(f"     - Morphologiques: 2 (hu_1, eccentricity)")
    print(f"   • Texture/Couleur flexibles: {active_texture_color} features")
    
    print("\nFeatures texture/couleur actives:")
    print("   GLCM Features:")
    for feat, active in GLCM_FEATURES.items():
        if active:
            print(f"      glcm_{feat}")
    
    print("   Histogram Features:")
    for feat, active in HISTOGRAM_FEATURES.items():
        if active:
            print(f"      {feat}")
    
    print("   Histogram Moments:")
    for feat, active in HISTOGRAM_MOMENTS.items():
        if active:
            print(f"      hist_{feat}")
    
    print("   HSV Features:")
    for feat, active in HSV_FEATURES.items():
        if active:
            print(f"      {feat}")
    
    if KEEP_ORIGINAL_HUE_SKEWNESS:
        print("    hue_skewness (feature originale)")
    
    print("=" * 60)
    
    # Arguments
    args = parse_arguments()
    
    try:
        # 1. Charger le modèle
        pipeline, label_encoder = load_model_and_encoder(args.model_dir)
        
        # 2. Liste des IDs à traiter
        image_ids = list(range(args.start, args.end + 1))
        print(f"\n Images à traiter: {len(image_ids)} (IDs {args.start} à {args.end})")
        
        # 3. Traitement par batchs
        all_predictions = []
        
        for i in range(0, len(image_ids), args.batch_size):
            batch_ids = image_ids[i:i + args.batch_size]
            print(f"\nBatch {i//args.batch_size + 1}/{(len(image_ids)-1)//args.batch_size + 1}")
            
            # Extraction features
            features_list = extract_features_batch(batch_ids, args.test_dir)
            
            # Conversion en DataFrame
            features_df = pd.DataFrame(features_list)
            
            # Prédictions
            predictions = predict_batch(features_df, pipeline, label_encoder)
            all_predictions.append(predictions)
        
        # 4. Combiner toutes les prédictions
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        final_predictions = final_predictions.sort_values('ID').reset_index(drop=True)
        
        # 5. Sauvegarder
        final_predictions.to_csv(args.output, index=False)
        
        print(f"\n Prédictions terminées!")
        print(f" Fichier de sortie: {args.output}")
        print(f"  Nombre de prédictions: {len(final_predictions)}")
        
        # Afficher distribution
        print(f"\n Distribution des prédictions:")
        pred_counts = final_predictions['bug type'].value_counts().sort_index()
        for bug_type, count in pred_counts.items():
            pct = (count / len(final_predictions)) * 100
            print(f"   {bug_type:12s} : {count:3d} ({pct:5.1f}%)")
        
        # Vérifier la cohérence
        expected_ids = set(range(args.start, args.end + 1))
        actual_ids = set(final_predictions['ID'].values)
        missing_ids = expected_ids - actual_ids
        
        if missing_ids:
            print(f"\n  IDs manquants: {sorted(missing_ids)}")
        else:
            print(f"\n Tous les IDs sont présents")
        
    except Exception as e:
        print(f"\n ERREUR: {e}")
        raise

if __name__ == "__main__":
    main()