import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy import stats, signal
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Reproductibilité
np.random.seed(42)

def load_image_and_mask(image_id):
    """Charge l'image et le masque"""
    try:
        # Chemins
        img_path = f"train/{image_id}.JPG"
        mask_path = f"train/masks/binary_{image_id}.tif"
        
        # Charger l'image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None, None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Charger le masque
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None
            
        # Binariser le masque
        mask = (mask >= 1).astype(np.uint8)
        
        return img_rgb, mask
        
    except Exception as e:
        print(f"Erreur chargement image {image_id}: {e}")
        return None, None

def compute_histogram_features(image_id):
    """Calcule toutes les features d'histogramme pour une image"""
    try:
        # Charger l'image et le masque
        img_rgb, mask = load_image_and_mask(image_id)
        
        if img_rgb is None or mask is None:
            return None
        
        # Convertir en niveaux de gris
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Extraire uniquement les pixels de l'insecte
        bug_pixels = img_gray[mask == 1]
        
        if len(bug_pixels) == 0:
            return None
        
        features = {'image_id': image_id}
        
        # 1. STATISTIQUES DE BASE
        features['hist_mean'] = np.mean(bug_pixels)
        features['hist_std'] = np.std(bug_pixels)
        features['hist_variance'] = np.var(bug_pixels)
        features['hist_median'] = np.median(bug_pixels)
        features['hist_mad'] = np.median(np.abs(bug_pixels - features['hist_median']))  # Median Absolute Deviation
        
        # 2. MOMENTS STATISTIQUES
        features['hist_skewness'] = stats.skew(bug_pixels)
        features['hist_kurtosis'] = stats.kurtosis(bug_pixels)
        
        # Moments centraux normalisés
        mean = features['hist_mean']
        for moment_order in range(2, 6):
            moment = np.mean((bug_pixels - mean) ** moment_order)
            features[f'hist_moment_{moment_order}'] = moment
        
        # 3. PERCENTILES ET QUANTILES
        percentiles = [5, 10, 25, 75, 90, 95]
        for p in percentiles:
            features[f'hist_p{p}'] = np.percentile(bug_pixels, p)
        
        # Inter-quartile range
        features['hist_iqr'] = features['hist_p75'] - features['hist_p25']
        features['hist_range'] = np.max(bug_pixels) - np.min(bug_pixels)
        
        # Coefficient de variation
        features['hist_cv'] = features['hist_std'] / features['hist_mean'] if features['hist_mean'] != 0 else 0
        
        # 4. ANALYSE DE L'HISTOGRAMME
        hist_counts, bin_edges = np.histogram(bug_pixels, bins=256, range=(0, 256))
        hist_norm = hist_counts / hist_counts.sum()
        
        # Entropie
        hist_nonzero = hist_norm[hist_norm > 0]
        features['hist_entropy'] = -np.sum(hist_nonzero * np.log2(hist_nonzero))
        
        # Énergie
        features['hist_energy'] = np.sum(hist_norm ** 2)
        
        # Mode (valeur la plus fréquente)
        mode_idx = np.argmax(hist_counts)
        features['hist_mode'] = mode_idx
        features['hist_mode_count'] = hist_counts[mode_idx]
        features['hist_mode_ratio'] = hist_counts[mode_idx] / len(bug_pixels)
        
        # Nombre de pics dans l'histogramme
        peaks, properties = signal.find_peaks(hist_counts, height=len(bug_pixels)*0.01)
        features['hist_num_peaks'] = len(peaks)
        
        # 5. MESURES DE CONTRASTE
        # Contraste de Michelson
        L_max = np.max(bug_pixels)
        L_min = np.min(bug_pixels)
        features['hist_michelson_contrast'] = (L_max - L_min) / (L_max + L_min) if (L_max + L_min) > 0 else 0
        
        # RMS contrast
        features['hist_rms_contrast'] = features['hist_std'] / features['hist_mean'] if features['hist_mean'] > 0 else 0
        
        # 6. ANALYSE DE LA DISTRIBUTION
        # Test de normalité (Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(bug_pixels)
        features['hist_jarque_bera_stat'] = jb_stat
        features['hist_is_normal'] = 1 if jb_pvalue > 0.05 else 0
        
        # Uniformité (variance de l'histogramme normalisé)
        features['hist_uniformity'] = np.var(hist_norm)
        
        # Smoothness (différences entre bins consécutifs)
        hist_diff = np.diff(hist_counts)
        features['hist_smoothness'] = np.std(hist_diff)
        
        # 7. CARACTÉRISTIQUES SPECTRALES DE L'HISTOGRAMME
        # FFT de l'histogramme pour capturer la périodicité
        hist_fft = np.fft.fft(hist_counts)
        hist_fft_mag = np.abs(hist_fft[:128])  # Première moitié
        
        # Énergie dans différentes bandes de fréquence
        features['hist_fft_low_energy'] = np.sum(hist_fft_mag[:32])
        features['hist_fft_mid_energy'] = np.sum(hist_fft_mag[32:64])
        features['hist_fft_high_energy'] = np.sum(hist_fft_mag[64:128])
        
        # 8. MESURES BASÉES SUR L'HISTOGRAMME CUMULÉ
        hist_cumsum = np.cumsum(hist_norm)
        
        # Médiane depuis l'histogramme cumulé
        median_idx = np.argmax(hist_cumsum >= 0.5)
        features['hist_median_from_cumsum'] = median_idx
        
        # Spread autour de la médiane
        idx_25 = np.argmax(hist_cumsum >= 0.25)
        idx_75 = np.argmax(hist_cumsum >= 0.75)
        features['hist_quartile_spread'] = idx_75 - idx_25
        
        # 9. MESURES DE TEXTURE BASÉES SUR L'HISTOGRAMME
        # Local Binary Pattern simplifiée
        lbp_simple = np.zeros_like(img_gray)
        for i in range(1, img_gray.shape[0]-1):
            for j in range(1, img_gray.shape[1]-1):
                if mask[i, j] == 1:
                    center = img_gray[i, j]
                    code = 0
                    code |= (img_gray[i-1, j] > center) << 0
                    code |= (img_gray[i, j+1] > center) << 1
                    code |= (img_gray[i+1, j] > center) << 2
                    code |= (img_gray[i, j-1] > center) << 3
                    lbp_simple[i, j] = code
        
        lbp_hist, _ = np.histogram(lbp_simple[mask == 1], bins=16, range=(0, 16))
        lbp_hist_norm = lbp_hist / lbp_hist.sum() if lbp_hist.sum() > 0 else lbp_hist
        
        # Entropie LBP
        lbp_nonzero = lbp_hist_norm[lbp_hist_norm > 0]
        features['hist_lbp_entropy'] = -np.sum(lbp_nonzero * np.log2(lbp_nonzero)) if len(lbp_nonzero) > 0 else 0
        
        # 10. RATIOS ET MESURES DÉRIVÉES
        # Ratio entre différents percentiles
        if features['hist_p90'] > 0:
            features['hist_p10_p90_ratio'] = features['hist_p10'] / features['hist_p90']
        else:
            features['hist_p10_p90_ratio'] = 0
            
        # Asymétrie basée sur la médiane
        features['hist_median_skewness'] = (features['hist_mean'] - features['hist_median']) / features['hist_std'] if features['hist_std'] > 0 else 0
        
        # Concentration autour de la moyenne
        within_1std = np.sum(np.abs(bug_pixels - mean) <= features['hist_std'])
        features['hist_concentration_1std'] = within_1std / len(bug_pixels)
        
        return features
        
    except Exception as e:
        print(f"Erreur calcul histogramme pour image {image_id}: {e}")
        return None

def perform_anova_analysis(df):
    """Effectue une analyse ANOVA pour chaque feature d'histogramme"""
    # Créer des groupes basés sur une caractéristique de référence
    # Utilisons la moyenne de l'histogramme pour créer des groupes
    df['group'] = pd.qcut(df['hist_mean'], q=5, labels=['Very Dark', 'Dark', 'Medium', 'Light', 'Very Light'])
    
    feature_cols = [col for col in df.columns if col.startswith('hist_') and col != 'hist_mean']
    
    anova_results = []
    
    for col in feature_cols:
        try:
            # Préparer les données pour l'ANOVA
            groups = []
            for group_name in df['group'].unique():
                group_data = df[df['group'] == group_name][col].dropna().values
                if len(group_data) > 0:
                    groups.append(group_data)
            
            if len(groups) < 2:
                continue
                
            # Effectuer l'ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Calculer eta-squared (taille d'effet)
            grand_mean = df[col].mean()
            ssb = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            sst = np.sum((df[col].dropna() - grand_mean)**2)
            eta_squared = ssb / sst if sst > 0 else 0
            
            # Coefficient de variation
            cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
            
            anova_results.append({
                'feature': col,
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'cv': abs(cv),
                'mean': df[col].mean(),
                'std': df[col].std()
            })
            
        except Exception as e:
            print(f"Erreur ANOVA pour {col}: {e}")
    
    anova_df = pd.DataFrame(anova_results)
    anova_df = anova_df.sort_values('f_statistic', ascending=False)
    
    return anova_df, df

def compute_mutual_information(df):
    """Calcule l'information mutuelle pour chaque feature d'histogramme"""
    
    # Préparer les features
    feature_cols = [col for col in df.columns if col.startswith('hist_')]
    X = df[feature_cols].values
    
    # Remplacer les NaN
    for i in range(X.shape[1]):
        col_data = X[:, i]
        nan_mask = np.isnan(col_data)
        if np.any(nan_mask):
            median_val = np.nanmedian(col_data)
            X[nan_mask, i] = median_val
    
    # Normaliser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Créer des clusters
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calculer l'information mutuelle
    mi_scores = mutual_info_classif(X, clusters, random_state=42)
    
    # Information mutuelle entre features
    mi_between_features = []
    
    for i in range(len(feature_cols)):
        mi_with_others = []
        for j in range(len(feature_cols)):
            if i != j:
                mi_score = mutual_info_classif(
                    X[:, j].reshape(-1, 1), 
                    (X[:, i] > np.median(X[:, i])).astype(int),
                    random_state=42
                )[0]
                mi_with_others.append(mi_score)
        
        avg_mi_with_others = np.mean(mi_with_others) if mi_with_others else 0
        mi_between_features.append(avg_mi_with_others)
    
    # Scores
    uniqueness_scores = 1 / (1 + np.array(mi_between_features))
    composite_scores = mi_scores * uniqueness_scores
    
    # Créer le DataFrame
    mi_results = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores,
        'mi_avg_with_others': mi_between_features,
        'uniqueness_score': uniqueness_scores,
        'composite_score': composite_scores
    })
    
    # Ajouter des stats
    for i, feature in enumerate(feature_cols):
        mi_results.loc[i, 'mean'] = np.mean(X[:, i])
        mi_results.loc[i, 'std'] = np.std(X[:, i])
        mi_results.loc[i, 'cv'] = np.std(X[:, i]) / (np.mean(X[:, i]) + 1e-10)
    
    mi_results = mi_results.sort_values('composite_score', ascending=False)
    
    return mi_results, clusters

def create_comparison_visualizations(df, anova_results, mi_results, clusters, output_dir):
    """Crée des visualisations comparant ANOVA et MI"""
    viz_dir = Path(output_dir) / "histogram_analysis"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Comparaison directe ANOVA vs MI
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Fusionner les résultats
    comparison_df = pd.merge(
        anova_results[['feature', 'f_statistic', 'p_value', 'eta_squared']],
        mi_results[['feature', 'mi_score', 'composite_score']],
        on='feature'
    )
    
    # Normaliser les scores pour la comparaison
    comparison_df['f_stat_norm'] = (comparison_df['f_statistic'] - comparison_df['f_statistic'].min()) / (comparison_df['f_statistic'].max() - comparison_df['f_statistic'].min())
    comparison_df['mi_score_norm'] = (comparison_df['mi_score'] - comparison_df['mi_score'].min()) / (comparison_df['mi_score'].max() - comparison_df['mi_score'].min())
    
    # Scatter plot
    scatter = ax1.scatter(comparison_df['f_stat_norm'], comparison_df['mi_score_norm'],
                         s=100, alpha=0.6, c=comparison_df['eta_squared'],
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Ligne diagonale
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Accord parfait')
    
    # Annoter les points intéressants
    for idx, row in comparison_df.head(10).iterrows():
        ax1.annotate(row['feature'].replace('hist_', ''), 
                    (row['f_stat_norm'], row['mi_score_norm']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Score ANOVA normalisé', fontsize=12)
    ax1.set_ylabel('Score MI normalisé', fontsize=12)
    ax1.set_title('Comparaison ANOVA vs Information Mutuelle', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    plt.colorbar(scatter, ax=ax1, label='Eta-squared')
    
    # Corrélation entre les méthodes
    corr = comparison_df['f_stat_norm'].corr(comparison_df['mi_score_norm'])
    ax1.text(0.05, 0.95, f'Corrélation: {corr:.3f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Top features par méthode
    top_n = 15
    top_anova = anova_results.head(top_n)['feature'].tolist()
    top_mi = mi_results.head(top_n)['feature'].tolist()
    
    # Diagramme de Venn conceptuel
    common_features = set(top_anova) & set(top_mi)
    only_anova = set(top_anova) - set(top_mi)
    only_mi = set(top_mi) - set(top_anova)
    
    ax2.text(0.5, 0.9, f'Top {top_n} Features', ha='center', fontsize=14, weight='bold',
             transform=ax2.transAxes)
    
    ax2.text(0.25, 0.7, 'ANOVA seul', ha='center', fontsize=12, weight='bold',
             transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax2.text(0.25, 0.5, '\n'.join([f.replace('hist_', '') for f in list(only_anova)[:5]]),
             ha='center', fontsize=10, transform=ax2.transAxes)
    
    ax2.text(0.75, 0.7, 'MI seul', ha='center', fontsize=12, weight='bold',
             transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax2.text(0.75, 0.5, '\n'.join([f.replace('hist_', '') for f in list(only_mi)[:5]]),
             ha='center', fontsize=10, transform=ax2.transAxes)
    
    ax2.text(0.5, 0.3, 'Commun', ha='center', fontsize=12, weight='bold',
             transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax2.text(0.5, 0.1, '\n'.join([f.replace('hist_', '') for f in list(common_features)[:5]]),
             ha='center', fontsize=10, transform=ax2.transAxes)
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'anova_vs_mi_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Ranking côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # ANOVA ranking
    top_20_anova = anova_results.head(20)
    bars1 = ax1.barh(range(len(top_20_anova)), top_20_anova['f_statistic'])
    ax1.set_yticks(range(len(top_20_anova)))
    ax1.set_yticklabels(top_20_anova['feature'].str.replace('hist_', ''))
    ax1.set_xlabel('F-statistique', fontsize=12)
    ax1.set_title('Top 20 Features par ANOVA', fontsize=14, weight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Colorier selon la p-value
    colors_anova = ['darkgreen' if p < 0.001 else 'green' if p < 0.01 else 'orange' 
                    for p in top_20_anova['p_value']]
    for bar, color in zip(bars1, colors_anova):
        bar.set_color(color)
    
    # MI ranking
    top_20_mi = mi_results.head(20)
    bars2 = ax2.barh(range(len(top_20_mi)), top_20_mi['composite_score'])
    ax2.set_yticks(range(len(top_20_mi)))
    ax2.set_yticklabels(top_20_mi['feature'].str.replace('hist_', ''))
    ax2.set_xlabel('Score Composite MI', fontsize=12)
    ax2.set_title('Top 20 Features par Information Mutuelle', fontsize=14, weight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Colorier selon l'unicité
    colors_mi = plt.cm.viridis(top_20_mi['uniqueness_score'])
    for bar, color in zip(bars2, colors_mi):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'features_ranking_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distributions des meilleures features selon les deux méthodes
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top 3 ANOVA
    for i in range(3):
        if i < len(anova_results):
            feature = anova_results.iloc[i]['feature']
            ax = axes[0, i]
            
            # Histogramme par groupe
            for group in df['group'].unique():
                group_data = df[df['group'] == group][feature].dropna()
                ax.hist(group_data, bins=20, alpha=0.5, label=group)
            
            ax.set_title(f"ANOVA #{i+1}: {feature.replace('hist_', '')}\nF={anova_results.iloc[i]['f_statistic']:.2f}",
                        fontsize=10, weight='bold')
            ax.set_xlabel('Valeur')
            ax.set_ylabel('Fréquence')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    # Top 3 MI
    for i in range(3):
        if i < len(mi_results):
            feature = mi_results.iloc[i]['feature']
            ax = axes[1, i]
            
            # Histogramme par cluster
            for cluster_id in range(max(clusters) + 1):
                cluster_data = df.iloc[clusters == cluster_id][feature].dropna()
                ax.hist(cluster_data, bins=20, alpha=0.5, label=f'Cluster {cluster_id}')
            
            ax.set_title(f"MI #{i+1}: {feature.replace('hist_', '')}\nScore={mi_results.iloc[i]['composite_score']:.3f}",
                        fontsize=10, weight='bold')
            ax.set_xlabel('Valeur')
            ax.set_ylabel('Fréquence')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    plt.suptitle('Distribution des Top Features: ANOVA (haut) vs MI (bas)', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / 'top_features_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap des corrélations pour les features sélectionnées
    plt.figure(figsize=(14, 12))
    
    # Sélectionner les top features des deux méthodes
    selected_features = list(set(
        anova_results.head(10)['feature'].tolist() + 
        mi_results.head(10)['feature'].tolist()
    ))
    
    if len(selected_features) > 3:
        corr_matrix = df[selected_features].corr()
        
        # Annoter avec la méthode qui a sélectionné chaque feature
        labels = []
        for feat in selected_features:
            in_anova = feat in anova_results.head(10)['feature'].values
            in_mi = feat in mi_results.head(10)['feature'].values
            
            if in_anova and in_mi:
                labels.append(f"{feat.replace('hist_', '')} (Both)")
            elif in_anova:
                labels.append(f"{feat.replace('hist_', '')} (ANOVA)")
            else:
                labels.append(f"{feat.replace('hist_', '')} (MI)")
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5,
                   xticklabels=labels, yticklabels=labels)
        
        plt.title('Corrélation entre les Top Features (ANOVA + MI)', 
                 fontsize=16, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(viz_dir / 'selected_features_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualisations comparatives sauvegardées dans: {viz_dir}")

def main():
    """Fonction principale"""
    print(" Analyse des Features d'Histogramme avec ANOVA et Information Mutuelle")
    print("=" * 70)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Liste des images
    image_ids = list(range(1, 251))
    
    print(f"\n Calcul des features d'histogramme pour {len(image_ids)} images...")
    print(f" Utilisation de {cpu_count()} processeurs")
    
    # Calcul parallèle
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(compute_histogram_features, image_ids),
            total=len(image_ids),
            desc="Extraction histogramme",
            unit="image"
        ))
    
    # Filtrer les résultats valides
    valid_results = [r for r in results if r is not None]
    
    print(f"\n Images traitées avec succès: {len(valid_results)}/{len(image_ids)}")
    
    if len(valid_results) < 10:
        print(" Pas assez de données pour l'analyse")
        return
    
    # Créer le DataFrame
    df = pd.DataFrame(valid_results)
    
    # Sauvegarder les features
    csv_path = output_dir / "histogram_features_complete.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n Features d'histogramme sauvegardées: {csv_path}")
    
    # Nombre de features
    n_features = len([col for col in df.columns if col.startswith('hist_')])
    print(f" Nombre de features d'histogramme extraites: {n_features}")
    
    # ANALYSE ANOVA
    print("\n Analyse ANOVA...")
    anova_results, df_with_groups = perform_anova_analysis(df)
    
    anova_path = output_dir / "histogram_anova_results.csv"
    anova_results.to_csv(anova_path, index=False)
    print(f"💾 Résultats ANOVA sauvegardés: {anova_path}")
    
    # ANALYSE INFORMATION MUTUELLE
    print("\nAnalyse par Information Mutuelle...")
    mi_results, clusters = compute_mutual_information(df)
    
    mi_path = output_dir / "histogram_mi_results.csv"
    mi_results.to_csv(mi_path, index=False)
    print(f" Résultats MI sauvegardés: {mi_path}")
    
    # COMPARAISON DES RÉSULTATS
    print("\nComparaison ANOVA vs Information Mutuelle:")
    print("=" * 80)
    
    # Top 10 par méthode
    print("\n TOP 10 ANOVA:")
    print(f"{'Rang':<5} {'Feature':<30} {'F-stat':<10} {'p-value':<12} {'η²':<8}")
    print("-" * 75)
    
    for idx, row in anova_results.head(10).iterrows():
        print(f"{idx+1:<5} {row['feature'].replace('hist_', ''):<30} "
              f"{row['f_statistic']:<10.2f} {row['p_value']:<12.4e} "
              f"{row['eta_squared']:<8.3f}")
    
    print("\n TOP 10 Information Mutuelle:")
    print(f"{'Rang':<5} {'Feature':<30} {'MI Score':<10} {'Unicité':<10} {'Composite':<10}")
    print("-" * 75)
    
    for idx, row in mi_results.head(10).iterrows():
        print(f"{idx+1:<5} {row['feature'].replace('hist_', ''):<30} "
              f"{row['mi_score']:<10.4f} {row['uniqueness_score']:<10.4f} "
              f"{row['composite_score']:<10.4f}")
    
    # Features communes dans le top 10
    top_10_anova = set(anova_results.head(10)['feature'])
    top_10_mi = set(mi_results.head(10)['feature'])
    common_top_10 = top_10_anova & top_10_mi
    
    print(f"\n Features communes dans le TOP 10: {len(common_top_10)}")
    for feat in common_top_10:
        anova_rank = anova_results[anova_results['feature'] == feat].index[0] + 1
        mi_rank = mi_results[mi_results['feature'] == feat].index[0] + 1
        print(f"   - {feat.replace('hist_', '')}: ANOVA #{anova_rank}, MI #{mi_rank}")
    
    # Créer les visualisations comparatives
    print("\n Création des visualisations comparatives...")
    create_comparison_visualizations(df_with_groups, anova_results, mi_results, clusters, output_dir)
    
    # RECOMMANDATIONS FINALES
    print("\n Recommandations basées sur l'analyse combinée:")
    
    # Features robustes (bien classées dans les deux méthodes)
    print("\n Features robustes (top 20 dans les deux méthodes):")
    top_20_anova = set(anova_results.head(20)['feature'])
    top_20_mi = set(mi_results.head(20)['feature'])
    robust_features = top_20_anova & top_20_mi
    
    for feat in list(robust_features)[:8]:
        anova_row = anova_results[anova_results['feature'] == feat].iloc[0]
        mi_row = mi_results[mi_results['feature'] == feat].iloc[0]
        print(f"   - {feat.replace('hist_', '')}: "
              f"ANOVA F={anova_row['f_statistic']:.2f}, "
              f"MI={mi_row['mi_score']:.3f}")
    
    # Features complémentaires
    print("\nEnsemble optimal de features (diversité maximale):")
    
    # Sélectionner un mix des deux méthodes
    selected_features = []
    
    # Prendre les top 3 de chaque méthode
    for feat in anova_results.head(3)['feature']:
        if feat not in selected_features:
            selected_features.append(feat)
    
    for feat in mi_results.head(3)['feature']:
        if feat not in selected_features:
            selected_features.append(feat)
    
    # Ajouter des features uniques à chaque méthode
    unique_anova = top_10_anova - top_10_mi
    unique_mi = top_10_mi - top_10_anova
    
    if unique_anova:
        selected_features.append(list(unique_anova)[0])
    if unique_mi:
        selected_features.append(list(unique_mi)[0])
    
    print("\nFeatures sélectionnées pour un ensemble optimal:")
    for i, feat in enumerate(selected_features[:8]):
        method = "Both" if feat in common_top_10 else "ANOVA" if feat in top_10_anova else "MI"
        print(f"   {i+1}. {feat.replace('hist_', '')} (sélectionnée par: {method})")
    
    # Analyse des features originales demandées
    print("\n Statut des features originales demandées:")
    original_features = ['hist_mean', 'hist_std', 'hist_variance', 'hist_skewness',
                        'hist_kurtosis', 'hist_entropy', 'hist_energy', 'hist_mode']
    
    for feat in original_features:
        if feat in df.columns:
            if feat in anova_results['feature'].values:
                anova_rank = anova_results[anova_results['feature'] == feat].index[0] + 1
                anova_f = anova_results[anova_results['feature'] == feat].iloc[0]['f_statistic']
            else:
                anova_rank = 'N/A'
                anova_f = 'N/A'
                
            if feat in mi_results['feature'].values:
                mi_rank = mi_results[mi_results['feature'] == feat].index[0] + 1
                mi_score = mi_results[mi_results['feature'] == feat].iloc[0]['mi_score']
            else:
                mi_rank = 'N/A'
                mi_score = 'N/A'
                
            print(f"   - {feat}: ANOVA rang #{anova_rank} (F={anova_f}), "
                  f"MI rang #{mi_rank} (score={mi_score})")
    
    print("\n Analyse terminée!")
    print(f" Tous les résultats sont dans: {output_dir}")

if __name__ == "__main__":
    main()