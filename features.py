#features minimal correct avec texture et couleur avancées - VERSION FLEXIBLE

import numpy as np
import pandas as pd
from skimage import io, measure, feature, morphology
from scipy import optimize, ndimage as ndi, stats
import cv2
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # backend "headless" pour juste sauvegarder
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
import time
warnings.filterwarnings('ignore')

# Reproductibilité
np.random.seed(42)

# ===============================================
# CONFIGURATION DES FEATURES - MODIFIER ICI !
# ===============================================

# Features GLCM (Gray Level Co-occurrence Matrix)
GLCM_FEATURES = {
    'homogeneity': True,     # Homogénéité (inverse du contraste)
    'contrast': False,       # Contraste local
    'dissimilarity': True,  # Dissimilarité
    'energy': True,         # Energie (ASM)
    'correlation': True,    # Corrélation
    'ASM': False,           # Angular Second Moment (= energy)
}

# Features d'histogramme (sur l'image en niveaux de gris)
HISTOGRAM_FEATURES = {
    'hist_mean': True,      # Moyenne
    'hist_std': True,       # Écart-type
    'hist_variance': True,  # Variance
    'hist_skewness': True,  # Asymétrie (skewness)
    'hist_kurtosis': True,  # Aplatissement (kurtosis)
    'hist_entropy': True,   # Entropie
    'hist_energy': True,    # Énergie
    'hist_mode': True,      # Mode (valeur la plus fréquente)
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

# Moments de Hu (invariants aux transformations)
HU_MOMENTS = {
    'hu_1': True,           # Premier moment de Hu (toujours actif par défaut)
    'hu_2': True,          # Deuxième moment de Hu
    'hu_3': True,          # Troisième moment de Hu
    'hu_4': True,          # Quatrième moment de Hu
    'hu_5': True,          # Cinquième moment de Hu
    'hu_6': True,          # Sixième moment de Hu
    'hu_7': True,          # Septième moment de Hu
}

# Garde toujours hue_skewness actif (feature originale)
KEEP_ORIGINAL_HUE_SKEWNESS = False

# ===============================================

# --- Rotation Function from Lab01 ---
def rotate_image(theta_degree, xc, yc, arr): # xc, yc are row, col for center
    """ Proper rotation of angle theta_degree and center (xc, yc) of an image.
        Based on lab01.py Q4 logic. xc, yc are 0-indexed (row, col) coords.
    """
    theta = theta_degree * np.pi / 180
    s1, s2 = arr.shape[:2] # Works for grayscale and color images

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
        for i in range(arr.shape[2]): # Handle color channels
            channel_arr = np.zeros((s1,s2))
            channel_arr[r_rot_flat[valid_mask], c_rot_flat[valid_mask]] = arr[r_src_idx[valid_mask], c_src_idx[valid_mask], i]
            rot_arr[:,:,i] = channel_arr.reshape(s1,s2)
    else: # Grayscale
        rot_arr = np.zeros_like(arr)
        rot_arr[r_rot_flat[valid_mask], c_rot_flat[valid_mask]] = arr[r_src_idx[valid_mask], c_src_idx[valid_mask]]
        rot_arr = rot_arr.reshape(s1,s2) # ensure shape

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

        # Dimensions originales pour le ratio
        with Image.open(img_p) as temp_pil_img:
            original_dims = temp_pil_img.size # (width, height)

        # Binarisation et nettoyage
        bin_mask = (mask_arr >= 1).astype(np.uint8)
        orig_bug_px_count = np.sum(bin_mask)
        
        # Nettoyage morphologique pour réduire le bruit
        # Closing léger pour éliminer les petits trous et unifier les régions
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

        # Préférer les composantes qui ne touchent pas les bords
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
    return com[0], com[1] # (center_row, center_col)

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
            return -dt[r_idx, c_idx] # Minimize negative distance (maximize distance)

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

def rotate_image_symmetry(image, theta_degree, xc, yc):
    """ Rotation of angle theta_degree around center (xc, yc) of a binary image 
    Adaptation exacte du code fourni """
    # Convert the angle into radian
    theta = np.radians(theta_degree)
    # Image size
    s1, s2 = image.shape  # s1 = height (rows), s2 = width (cols)
    # Get all the pixels in the input array
    y, x = np.indices((s1, s2))
    # Compute the coordinates obtained after rotation
    xr = np.cos(theta) * (x - xc) - np.sin(theta) * (y - yc) + xc
    yr = np.sin(theta) * (x - xc) + np.cos(theta) * (y - yc) + yc
    # Initialize the rotation array (binary image)
    rot_arr = np.zeros_like(image)
    # Round coordinates and filter valid ones
    xr = np.round(xr).astype(int)
    yr = np.round(yr).astype(int)
    valid_mask = (xr >= 0) & (xr < s2) & (yr >= 0) & (yr < s1)
    rot_arr[yr[valid_mask], xr[valid_mask]] = image[y[valid_mask], x[valid_mask]]
    return rot_arr

def find_centroid_symmetry(mask):
    """Trouve le centroïde du masque pour la symétrie"""
    # Find the coordinates of non-zero (white) pixels
    y, x = np.nonzero(mask)
    if len(x) == 0 or len(y) == 0:
        return None, None
    # Calculate the mean coordinates (centroid)
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    return int(centroid_x), int(centroid_y)

def calculate_symmetry_index(mask, centroid_x):
    """Calcule l'index de symétrie en comparant les deux moitiés du masque"""
    # Vérifier que centroid_x est valide
    if centroid_x <= 0 or centroid_x >= mask.shape[1]:
        return 0
    
    # Split the mask into left and right halves at the centroid_x
    left_half = mask[:, :centroid_x].astype(float)
    right_half = mask[:, centroid_x:].astype(float)
    
    # Vérifier que les deux moitiés ont du contenu
    if left_half.size == 0 or right_half.size == 0:
        return 0
    
    # If the halves have different widths, adjust
    if left_half.shape[1] > right_half.shape[1]:
        left_half = left_half[:, :right_half.shape[1]]
    elif right_half.shape[1] > left_half.shape[1]:
        right_half = right_half[:, :left_half.shape[1]]

    # Flip the right half horizontally
    right_half = np.fliplr(right_half)
    
    # Calculate the absolute difference between the left and right halves
    difference = np.abs(left_half - right_half)
    
    # Sum the differences to get total difference score
    symmetry_score = np.sum(difference)
    
    # Normalize the score by the total number of pixels in the halves
    total_pixels = np.prod(left_half.shape)
    if total_pixels == 0:
        return 0
    
    symmetry_index = symmetry_score / total_pixels
    
    # Return the symmetry index where 1 means perfectly symmetrical
    return 1 - symmetry_index

def find_best_symmetry_full(mask):
    """Version complète qui retourne le score ET l'angle (pour visualisation)"""
    # S'assurer que le masque est binaire (0 ou 1)
    mask = mask.astype(np.uint8)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
    
    centroid_x, centroid_y = find_centroid_symmetry(mask)
    if centroid_x is None or centroid_y is None:
        return 0, 0
    
    best_symmetry_index = 0
    best_angle = 0
    
    # Test angles from -90 to 90 degrees in 5-degree increments
    for angle in range(-90, 91, 5):
        rotated_mask = rotate_image_symmetry(mask, angle, centroid_x, centroid_y)
        symmetry_index = calculate_symmetry_index(rotated_mask, centroid_x)
        if symmetry_index > best_symmetry_index:
            best_symmetry_index = symmetry_index
            best_angle = angle

    return best_symmetry_index, best_angle

def find_best_symmetry(mask):
    """Trouve le meilleur score de symétrie pour un masque (interface originale)"""
    score, _ = find_best_symmetry_full(mask)
    return score

def compute_best_symmetry_plane(image_arr, mask_arr, inscribed_circle_params, image_id_for_log="N/A"):
    """Calcule le meilleur plan de symétrie en utilisant la méthode de division en deux moitiés"""
    if mask_arr is None:
        return np.nan, np.nan
    
    # Utiliser la nouvelle méthode
    best_symmetry_index, best_angle = find_best_symmetry_full(mask_arr)
    
    # Retourner l'angle et le score de symétrie
    return float(best_angle), float(best_symmetry_index)

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
    Renvoie : hu_1, hu_2, ..., hu_7 (selon configuration) et eccentricity
    """
    try:
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

        # --- Calculer tous les moments de Hu ---
        m = cv2.moments(mask_uint8)
        hu_raw = cv2.HuMoments(m).flatten()
        
        # Transformer les moments de Hu avec log pour meilleure échelle
        # (les valeurs peuvent être très petites)
        hu_transformed = []
        for hu_val in hu_raw:
            if hu_val != 0:
                hu_transformed.append(-np.sign(hu_val) * np.log10(np.abs(hu_val)))
            else:
                hu_transformed.append(0.0)
        
        # Ajouter seulement les moments de Hu demandés
        features = {}
        for i in range(7):
            moment_name = f'hu_{i+1}'
            if HU_MOMENTS.get(moment_name, False):
                features[moment_name] = float(hu_transformed[i])

        # --- Eccentricity ---
        props = measure.regionprops(mask_uint8)
        if props:
            eccentricity = float(props[0].eccentricity)  # [0,1]
        else:
            eccentricity = 0.0
        
        features['eccentricity'] = eccentricity

        return features

    except Exception as e:
        print(f"[WARN] morph features: {e}")
        # Retourner des valeurs par défaut pour toutes les features actives
        default_features = {'eccentricity': 0.0}
        for i in range(7):
            moment_name = f'hu_{i+1}'
            if HU_MOMENTS.get(moment_name, False):
                default_features[moment_name] = 0.0
        return default_features

def save_visualization(image, mask, circle_params, sym_angle, image_id, output_dir, features):
    """Sauvegarde plusieurs visualisations des features"""
    try:
        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Figure principale avec plusieurs subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Image originale
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(image)
        ax1.set_title('Image originale', fontsize=14, weight='bold')
        ax1.axis('off')
        
        # 2. Masque avec contours
        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(image, alpha=0.7)
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ax2.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=3)
        ax2.set_title('Masque et contours', fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 3. Cercle inscrit
        ax3 = plt.subplot(2, 4, 3)
        ax3.imshow(image)
        if not np.isnan(circle_params[2]):
            circle = patches.Circle((circle_params[1], circle_params[0]), circle_params[2], 
                                   linewidth=3, edgecolor='blue', facecolor='none')
            ax3.add_patch(circle)
            ax3.plot(circle_params[1], circle_params[0], 'bo', markersize=8)
            ax3.text(0.05, 0.95, f'Rayon: {circle_params[2]:.1f}px\nRatio: {circle_params[3]:.3f}', 
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax3.set_title('Cercle inscrit', fontsize=14, weight='bold')
        ax3.axis('off')
        
        # 4. Plan de symétrie
        ax4 = plt.subplot(2, 4, 4)
        ax4.imshow(image)
        if not np.isnan(sym_angle):
            h, w = mask.shape
            # Utiliser le centroïde pour l'affichage de la symétrie
            centroid_x, centroid_y = find_centroid_symmetry(mask)
            if centroid_x is not None and centroid_y is not None:
                # L'angle est maintenant entre -90 et 90, représentant la rotation du masque
                # Pour l'axe de symétrie vertical après rotation, on trace une ligne perpendiculaire
                angle_rad = np.radians(90 - sym_angle)  # Perpendiculaire à l'angle de rotation
                line_length = max(h, w) * 0.5
                
                dx = line_length * np.cos(angle_rad)
                dy = line_length * np.sin(angle_rad)
                
                x1, y1 = centroid_x - dx, centroid_y - dy
                x2, y2 = centroid_x + dx, centroid_y + dy
                
                ax4.plot([x1, x2], [y1, y2], 'g-', linewidth=3)
                # Ajouter un point au centroïde
                ax4.plot(centroid_x, centroid_y, 'go', markersize=8)
                sym_score = features.get('symmetry_score', 0)
                ax4.text(0.05, 0.95, f'Angle rotation: {sym_angle:.1f}°\nScore: {sym_score:.3f}', 
                        transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax4.set_title('Plan de symétrie', fontsize=14, weight='bold')
        ax4.axis('off')
        
        # 5. Features morphologiques (texte)
        ax5 = plt.subplot(2, 4, 5)
        ax5.axis('off')
        morph_text = f"Features Morphologiques\n\n"
        
        # Afficher tous les moments de Hu actifs
        for i in range(7):
            moment_name = f'hu_{i+1}'
            if moment_name in features:
                morph_text += f"Hu moment #{i+1}: {features[moment_name]:.4f}\n"
        
        morph_text += f"Eccentricité: {features.get('eccentricity', 0):.3f}\n"
        morph_text += f"Ratio area: {features.get('ratio_area', 0):.4f}\n"
        
        # Ajouter les features de texture si actives
        if any(GLCM_FEATURES.values()):
            morph_text += f"\nGLCM Features:\n"
            for feat, active in GLCM_FEATURES.items():
                if active:
                    val = features.get(f'glcm_{feat}', 0)
                    morph_text += f"  {feat}: {val:.4f}\n"
        
        ax5.text(0.1, 0.9, morph_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', family='monospace')
        ax5.set_title('Features numériques', fontsize=14, weight='bold')
        
        # 6. Masque binaire seul
        ax6 = plt.subplot(2, 4, 6)
        ax6.imshow(mask, cmap='gray')
        ax6.set_title('Masque binaire', fontsize=14, weight='bold')
        ax6.axis('off')
        
        # 7. Distance transform (pour visualiser le cercle inscrit)
        ax7 = plt.subplot(2, 4, 7)
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        im7 = ax7.imshow(dt, cmap='hot')
        if not np.isnan(circle_params[0]):
            ax7.plot(circle_params[1], circle_params[0], 'bo', markersize=8)
        ax7.set_title('Distance Transform', fontsize=14, weight='bold')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        # 8. Vue combinée
        ax8 = plt.subplot(2, 4, 8)
        ax8.imshow(image)
        # Contours
        for contour in contours:
            ax8.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2, alpha=0.7)
        # Cercle
        if not np.isnan(circle_params[2]):
            circle = patches.Circle((circle_params[1], circle_params[0]), circle_params[2], 
                                   linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7)
            ax8.add_patch(circle)
        # Symétrie
        if not np.isnan(sym_angle):
            centroid_x, centroid_y = find_centroid_symmetry(mask)
            if centroid_x is not None and centroid_y is not None:
                # L'angle représente la rotation, l'axe de symétrie est perpendiculaire
                angle_rad = np.radians(90 - sym_angle)
                line_length = max(mask.shape[0], mask.shape[1]) * 0.4
                dx = line_length * np.cos(angle_rad)
                dy = line_length * np.sin(angle_rad)
                x1, y1 = centroid_x - dx, centroid_y - dy
                x2, y2 = centroid_x + dx, centroid_y + dy
                ax8.plot([x1, x2], [y1, y2], 'g--', linewidth=2, alpha=0.7)
        ax8.set_title('Vue combinée', fontsize=14, weight='bold')
        ax8.axis('off')
        
        # Titre principal
        fig.suptitle(f'Analyse des Features - Image {image_id}', fontsize=18, weight='bold')
        
        # Ajuster l'espacement
        plt.tight_layout()
        
        # Sauvegarder
        output_path = viz_dir / f"features_{image_id:03d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Créer une seconde figure pour les histogrammes si des features histogram sont actives
        if any(HISTOGRAM_FEATURES.values()) or any(HISTOGRAM_MOMENTS.values()):
            fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogramme de l'image en niveaux de gris
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_pixels = img_gray[mask == 1]
            
            if len(gray_pixels) > 0:
                axes[0].hist(gray_pixels, bins=50, alpha=0.7, color='gray', edgecolor='black')
                axes[0].set_title('Histogramme des niveaux de gris', fontsize=12, weight='bold')
                axes[0].set_xlabel('Valeur de pixel')
                axes[0].set_ylabel('Fréquence')
                axes[0].grid(True, alpha=0.3)
                
                # Afficher les statistiques
                hist_text = "Statistiques:\n"
                if 'hist_mean' in features:
                    hist_text += f"Moyenne: {features['hist_mean']:.1f}\n"
                if 'hist_std' in features:
                    hist_text += f"Std: {features['hist_std']:.1f}\n"
                if 'hist_skewness' in features:
                    hist_text += f"Skewness: {features['hist_skewness']:.3f}\n"
                if 'hist_kurtosis' in features:
                    hist_text += f"Kurtosis: {features['hist_kurtosis']:.3f}\n"
                if 'hist_entropy' in features:
                    hist_text += f"Entropie: {features['hist_entropy']:.3f}\n"
                
                axes[0].text(0.65, 0.95, hist_text, transform=axes[0].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Histogramme HSV si actif
            if any(HSV_FEATURES.values()) or KEEP_ORIGINAL_HUE_SKEWNESS:
                img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                hue = img_hsv[..., 0][mask == 1]
                
                if len(hue) > 0:
                    axes[1].hist(hue, bins=50, alpha=0.7, color='orange', edgecolor='black')
                    axes[1].set_title('Histogramme de teinte (Hue)', fontsize=12, weight='bold')
                    axes[1].set_xlabel('Valeur de teinte')
                    axes[1].set_ylabel('Fréquence')
                    axes[1].grid(True, alpha=0.3)
                    
                    if 'hue_skewness' in features:
                        axes[1].text(0.65, 0.95, f"Skewness: {features['hue_skewness']:.3f}",
                                   transform=axes[1].transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            fig2.suptitle(f'Histogrammes - Image {image_id}', fontsize=14, weight='bold')
            plt.tight_layout()
            
            hist_path = viz_dir / f"histograms_{image_id:03d}.png"
            plt.savefig(hist_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        
        return True
        
    except Exception as e:
        print(f"Erreur visualisation image {image_id}: {str(e)}")
        if 'fig' in locals():
            plt.close(fig)
        if 'fig2' in locals():
            plt.close(fig2)
        return False

def extract_features_single_robust(args):
    """Extrait features pour une image (version robuste)"""
    image_id, save_visuals, output_dir = args
    
    try:
        image_path = f"train/{image_id}.JPG"
        mask_path = f"train/masks/binary_{image_id}.tif"
        
        # Preprocessing robuste avec nettoyage morphologique
        cropped_img, cleaned_mask, bbox, orig_bug_px, original_dims = load_and_clean_mask(mask_path, image_path, image_id)
        
        if cropped_img is None or cleaned_mask is None:
            return {
                'image_id': image_id,
                'valid': False,
                'error': 'Preprocessing failed',
                'features': None
            }
        
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
        
        # 3. Shape & Symmetry (2 features) - AVEC SCIPY.OPTIMIZE
        circle_params = compute_best_inscribed_circle(cleaned_mask, image_id)
        features['inscribed_circle_radius_norm'] = circle_params[2] / min(cleaned_mask.shape) if not np.isnan(circle_params[2]) else 0
        
        sym_angle, sym_score = compute_best_symmetry_plane(cropped_img, cleaned_mask, circle_params, image_id)
        features['symmetry_loss_min'] = 1.0 - sym_score if not np.isnan(sym_score) else 1.0
        features['symmetry_angle']  = float(sym_angle) if not np.isnan(sym_angle) else np.nan
        features['symmetry_score']  = float(sym_score) if not np.isnan(sym_score) else 0.0

        # 4. Features morphologiques - MAINTENANT AVEC MOMENTS DE HU CONFIGURABLES
        morph_features = compute_morphological_features(cleaned_mask)
        features.update(morph_features)
        
        # 5. Features texture/couleur flexibles
        texture_color_features = compute_texture_color_features(cropped_img, cleaned_mask)
        features.update(texture_color_features)
        
        # 6. Visualisation avec toutes les features
        viz_success = True
        if save_visuals:
            viz_success = save_visualization(cropped_img, cleaned_mask, circle_params, 
                                           sym_angle, image_id, output_dir, features)
        
        return {
            'image_id': image_id,
            'valid': True,
            'error': None,
            'features': features,
            'viz_success': viz_success
        }
        
    except Exception as e:
        return {
            'image_id': image_id,
            'valid': False,
            'error': f"Erreur: {str(e)}",
            'features': None
        }

def count_active_features():
    """Compte le nombre de features actives"""
    count = 0
    count += sum(GLCM_FEATURES.values())
    count += sum(HISTOGRAM_FEATURES.values())
    count += sum(HISTOGRAM_MOMENTS.values())
    count += sum(HSV_FEATURES.values())
    count += sum(HU_MOMENTS.values())  # Compter les moments de Hu actifs
    if KEEP_ORIGINAL_HUE_SKEWNESS:
        count += 1
    # Eccentricity est toujours inclus donc ne pas le compter ici
    return count

def create_min_max_feature_visualizations(df, output_dir):
    """Crée des visualisations montrant les images avec valeurs min/max pour chaque feature"""
    print("\nCréation des visualisations min/max des features...")
    
    minmax_dir = Path(output_dir) / "minmax_features"
    minmax_dir.mkdir(exist_ok=True)
    
    # Liste des features à visualiser (excluant les couleurs et image_id)
    features_to_viz = [col for col in df.columns 
                      if col != 'image_id' 
                      and not col.startswith('color_')
                      and col not in ['symmetry_angle']]  # Angle n'est pas une métrique de qualité
    
    for feature_name in tqdm(features_to_viz, desc="Visualisation min/max"):
        try:
            # Ignorer les NaN
            valid_df = df[df[feature_name].notna()]
            
            if len(valid_df) < 2:
                continue
                
            # Trouver les indices min et max
            min_idx = valid_df[feature_name].idxmin()
            max_idx = valid_df[feature_name].idxmax()
            
            min_id = int(valid_df.loc[min_idx, 'image_id'])
            max_id = int(valid_df.loc[max_idx, 'image_id'])
            
            min_val = valid_df.loc[min_idx, feature_name]
            max_val = valid_df.loc[max_idx, feature_name]
            
            # Charger les images et masques
            min_img_path = f"train/{min_id}.JPG"
            max_img_path = f"train/{max_id}.JPG"
            min_mask_path = f"train/masks/binary_{min_id}.tif"
            max_mask_path = f"train/masks/binary_{max_id}.tif"
            
            # Lire et préparer les images
            min_img, min_mask, _, _, _ = load_and_clean_mask(min_mask_path, min_img_path, min_id)
            max_img, max_mask, _, _, _ = load_and_clean_mask(max_mask_path, max_img_path, max_id)
            
            if min_img is None or max_img is None:
                continue
            
            # Créer la figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Min - Image
            axes[0, 0].imshow(min_img)
            axes[0, 0].set_title(f'MIN - Image {min_id}', fontsize=14, weight='bold')
            axes[0, 0].text(0.05, 0.95, f'{feature_name} = {min_val:.4f}', 
                           transform=axes[0, 0].transAxes, fontsize=12,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            axes[0, 0].axis('off')
            
            # Min - Masque
            axes[0, 1].imshow(min_mask, cmap='gray')
            # Ajouter contours sur le masque
            contours = measure.find_contours(min_mask, 0.5)
            for contour in contours:
                axes[0, 1].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
            axes[0, 1].set_title(f'MIN - Masque {min_id}', fontsize=14, weight='bold')
            axes[0, 1].axis('off')
            
            # Max - Image
            axes[1, 0].imshow(max_img)
            axes[1, 0].set_title(f'MAX - Image {max_id}', fontsize=14, weight='bold')
            axes[1, 0].text(0.05, 0.95, f'{feature_name} = {max_val:.4f}', 
                           transform=axes[1, 0].transAxes, fontsize=12,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            axes[1, 0].axis('off')
            
            # Max - Masque
            axes[1, 1].imshow(max_mask, cmap='gray')
            # Ajouter contours sur le masque
            contours = measure.find_contours(max_mask, 0.5)
            for contour in contours:
                axes[1, 1].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
            axes[1, 1].set_title(f'MAX - Masque {max_id}', fontsize=14, weight='bold')
            axes[1, 1].axis('off')
            
            # Ajouter des annotations spécifiques selon la feature
            if feature_name == 'inscribed_circle_radius_norm':
                # Calculer et afficher les cercles
                for idx, (img, mask, ax_img, ax_mask) in enumerate([
                    (min_img, min_mask, axes[0, 0], axes[0, 1]),
                    (max_img, max_mask, axes[1, 0], axes[1, 1])
                ]):
                    circle_params = compute_best_inscribed_circle(mask)
                    if not np.isnan(circle_params[2]):
                        circle = patches.Circle((circle_params[1], circle_params[0]), 
                                              circle_params[2], linewidth=2, 
                                              edgecolor='blue', facecolor='none')
                        circle2 = patches.Circle((circle_params[1], circle_params[0]), 
                                               circle_params[2], linewidth=2, 
                                               edgecolor='blue', facecolor='none')
                        ax_img.add_patch(circle)
                        ax_mask.add_patch(circle2)
                        ax_img.plot(circle_params[1], circle_params[0], 'bo', markersize=6)
                        ax_mask.plot(circle_params[1], circle_params[0], 'bo', markersize=6)
            
            elif feature_name in ['symmetry_score', 'symmetry_loss_min']:
                # Afficher l'axe de symétrie
                for idx, (img, mask, ax_img, ax_mask, img_id) in enumerate([
                    (min_img, min_mask, axes[0, 0], axes[0, 1], min_id),
                    (max_img, max_mask, axes[1, 0], axes[1, 1], max_id)
                ]):
                    if mask is not None:
                        # Recalculer l'angle de symétrie pour chaque image
                        sym_score, sym_angle = find_best_symmetry_full(mask)
                        if not np.isnan(sym_angle):
                            h, w = mask.shape
                            # Utiliser le centroïde
                            centroid_x, centroid_y = find_centroid_symmetry(mask)
                            if centroid_x is not None and centroid_y is not None:
                                # L'axe de symétrie est perpendiculaire à l'angle de rotation
                                angle_rad = np.radians(90 - sym_angle)
                                line_length = max(h, w) * 0.4
                                
                                dx = line_length * np.cos(angle_rad)
                                dy = line_length * np.sin(angle_rad)
                                
                                x1, y1 = centroid_x - dx, centroid_y - dy
                                x2, y2 = centroid_x + dx, centroid_y + dy
                                
                                # Tracer sur l'image et le masque
                                ax_img.plot([x1, x2], [y1, y2], 'g-', linewidth=3)
                                ax_mask.plot([x1, x2], [y1, y2], 'g-', linewidth=3)
                                # Marquer le centroïde
                                ax_img.plot(centroid_x, centroid_y, 'go', markersize=6)
                                ax_mask.plot(centroid_x, centroid_y, 'go', markersize=6)
            
            # Titre principal
            fig.suptitle(f'Feature: {feature_name}\nRange: [{min_val:.4f}, {max_val:.4f}]', 
                        fontsize=16, weight='bold')
            
            plt.tight_layout()
            
            # Sauvegarder
            output_path = minmax_dir / f"minmax_{feature_name}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            print(f"Erreur pour feature {feature_name}: {str(e)}")
            if 'fig' in locals():
                plt.close()

def main():
    """Version robuste avec features texture et couleur flexibles"""
    print(" Extraction robuste avec features flexibles")
    print(" Cercle inscrit: distance transform + scipy.optimize.minimize")
    print(" Plan de symétrie: rotation custom + scipy.optimize.minimize_scalar")
    print(" Features morphologiques: moments de Hu configurables + eccentricity")
    
    # Afficher les features actives
    active_texture_color = count_active_features()
    base_morph_features = 1  # eccentricity toujours inclus
    print(f"\n Features texture/couleur actives ({active_texture_color} au total):")
    
    print("\n   Moments de Hu:")
    active_hu_count = 0
    for moment, active in HU_MOMENTS.items():
        status = "✓" if active else "✗"
        print(f"     [{status}] {moment}")
        if active:
            active_hu_count += 1
    
    print("\n   GLCM Features:")
    for feat, active in GLCM_FEATURES.items():
        status = "✓" if active else "✗"
        print(f"     [{status}] glcm_{feat}")
    
    print("\n   Histogram Features:")
    for feat, active in HISTOGRAM_FEATURES.items():
        status = "✓" if active else "✗"
        print(f"     [{status}] {feat}")
    
    print("\n   Histogram Moments:")
    for feat, active in HISTOGRAM_MOMENTS.items():
        status = "✓" if active else "✗"
        print(f"     [{status}] hist_{feat}")
    
    print("\n   HSV Features:")
    for feat, active in HSV_FEATURES.items():
        status = "✓" if active else "✗"
        print(f"     [{status}] {feat}")
    
    if KEEP_ORIGINAL_HUE_SKEWNESS:
        print("\n   [✓] hue_skewness (feature originale)")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    save_visuals = True
    image_ids = list(range(1, 251))  # Toutes les images
    
    print(f"\nImages à traiter: {len(image_ids)}")
    print(f"Processeurs: {cpu_count()}")
    
    args = [(img_id, save_visuals, output_dir) for img_id in image_ids]
    
    start_time = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(extract_features_single_robust, args),
            total=len(image_ids),
            desc="Extraction robuste",
            unit="image"
        ))
    
    valid_results = [r for r in results if r['valid']]
    invalid_results = [r for r in results if not r['valid']]
    valid_features = [r['features'] for r in valid_results]
    
    processing_time = time.time() - start_time
    
    print(f"\n Terminé en {processing_time:.1f}s")
    print(f"Images réussies: {len(valid_features)}/{len(image_ids)}")
    print(f"Images échouées: {len(invalid_results)}")
    
    if invalid_results:
        print(f"\n Images échouées:")
        for result in invalid_results[:10]:  # Montrer les 10 premières
            print(f"   Image {result['image_id']}: {result['error']}")
        if len(invalid_results) > 10:
            print(f"   ... et {len(invalid_results)-10} autres")
    
    # Sauvegarde
    if valid_features:
        df = pd.DataFrame(valid_features)
        
        csv_path = output_dir / "features_robust_final.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n Features sauvegardées: {csv_path}")
        
        summary_path = output_dir / "features_summary_robust.csv"
        df.describe().to_csv(summary_path)
        print(f"Résumé: {summary_path}")
        
        if save_visuals:
            viz_success = sum(1 for r in valid_results if r.get('viz_success', False))
            print(f" Visualisations: {viz_success}/{len(valid_results)}")
            
            # Créer les visualisations min/max
            create_min_max_feature_visualizations(df, output_dir)
            print(f" Visualisations min/max créées dans: {output_dir}/minmax_features/")
        
        # Compter les features finales
        base_features = 1 + 15 + 2 + base_morph_features  # ratio + couleur + shape/symmetry + eccentricity
        total_features = base_features + active_texture_color
        
        print(f"\nFEATURES EXTRAITES:")
        print(f"   • Total: {len(df.columns)-1} features (calculé: {total_features})")
        print(f"   • Base: {base_features} features")
        print(f"     - Ratio: 1 feature - ratio_area")
        print(f"     - Couleur: 15 features - min/max/mean/median/std pour R/G/B")
        print(f"     - Shape & Symmetry: 2 features - cercle inscrit + symétrie")
        print(f"     - Morphologiques: {base_morph_features + active_hu_count} features - {active_hu_count} moments de Hu + eccentricity")
        print(f"   • Texture/Couleur flexibles: {active_texture_color - active_hu_count} features")
        
        # Afficher toutes les colonnes pour vérification
        print(f"\n Colonnes extraites:")
        for i, col in enumerate(df.columns):
            if col != 'image_id':
                print(f"   {i:2d}. {col}")
                
    else:
        print("Aucune feature extraite")

if __name__ == "__main__":
    main()