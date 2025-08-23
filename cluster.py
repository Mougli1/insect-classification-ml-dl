#!/usr/bin/env python3
"""
clustering_analysis.py - Analyse de clustering non supervisé
============================================================

Script d'analyse de clustering avec KMeans et clustering hiérarchique.

Usage:
    python clustering_analysis.py --csv features.csv --xlsx classif.xlsx
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyse de clustering non supervisé')
    parser.add_argument('--csv', default='/Users/mouloudmerbouche/Library/Mobile Documents/com~apple~CloudDocs/Documents/ia+ml2/output/selected_feature_sets_rfe_subset/features_rfe_top_15_subset.csv', help='Chemin vers le CSV de features')
    parser.add_argument('--xlsx', default='train/classif.xlsx', help='Chemin vers classif.xlsx')
    parser.add_argument('--outdir', default='clustering_results', help='Dossier de sortie')
    return parser.parse_args()


def load_and_merge_data(csv_path, xlsx_path):
    print("Chargement des données...")
    
    # Lire CSV features
    df_feat = pd.read_csv(csv_path)
    if 'image_id' in df_feat.columns:
        df_feat = df_feat.rename(columns={'image_id': 'ID'})
    
    # Lire XLSX labels
    df_lbl = pd.read_excel(xlsx_path, sheet_name=0)
    
    # Fusion
    df_merged = df_feat.merge(df_lbl[['ID', 'bug type']], on='ID', how='inner')
    
    # Filtrer classes avec peu d'échantillons
    min_samples = 5
    class_counts = df_merged['bug type'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df_filtered = df_merged[df_merged['bug type'].isin(valid_classes)]
    
    print(f"   Images fusionnées: {len(df_filtered)} / {len(df_feat)}")
    print(f"   Classes retenues: {len(valid_classes)}")
    
    # Afficher distribution
    print("\nDistribution bug types:")
    for bug_type, count in df_filtered['bug type'].value_counts().sort_index().items():
        print(f"   {bug_type:12s} ... {count:3d}")
    
    return df_filtered, len(valid_classes)


def prepare_data(df_merged):
    print("\nPréparation des données...")
    
    # Séparation features/labels
    feature_cols = [col for col in df_merged.columns if col not in ['ID', 'bug type']]
    X = df_merged[feature_cols].values
    y_true = df_merged['bug type'].values
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA pour visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"   Features: {X.shape}")
    print(f"   Variance expliquée PCA: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")
    
    return X_scaled, X_pca, y_true


def kmeans_clustering(X_scaled, X_pca, y_true, n_clusters, outdir):
    print(f"\n=== KMeans (k={n_clusters}) ===")
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    
    # Métriques
    silhouette = silhouette_score(X_scaled, labels_kmeans)
    dbi = davies_bouldin_score(X_scaled, labels_kmeans)
    ari = adjusted_rand_score(y_true, labels_kmeans)
    
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Index: {dbi:.3f}")
    print(f"Adjusted Rand Index: {ari:.3f}")
    
    # Visualisation PCA
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, 
                         cmap='tab10', alpha=0.7, s=50)
    
    # Centres des clusters (projeter en PCA)
    pca_temp = PCA(n_components=2)
    pca_temp.fit(X_scaled)
    centers_pca = pca_temp.transform(kmeans.cluster_centers_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
               marker='*', s=300, c='black', edgecolor='white', linewidth=2)
    
    plt.title(f'KMeans Clustering (k={n_clusters})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(Path(outdir) / 'kmeans_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Matrice de contingence
    contingency = pd.crosstab(labels_kmeans, y_true)
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
    plt.title('KMeans: Clusters vs True Classes')
    plt.xlabel('True Bug Type')
    plt.ylabel('Cluster')
    plt.savefig(Path(outdir) / 'kmeans_contingency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Elbow method
    print("\nCalcul elbow method...")
    inertias = []
    silhouettes = []
    k_range = range(2, min(15, n_clusters*2))
    
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(k_range, inertias, 'bo-')
    ax1.axvline(x=n_clusters, color='red', linestyle='--')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(k_range, silhouettes, 'go-')
    ax2.axvline(x=n_clusters, color='red', linestyle='--')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs k')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'kmeans_elbow.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return labels_kmeans


def hierarchical_clustering(X_scaled, X_pca, y_true, n_clusters, outdir):
    print(f"\n=== Clustering Hiérarchique (k={n_clusters}) ===")
    
    # Clustering hiérarchique
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_hier = agg.fit_predict(X_scaled)
    
    # Métriques
    silhouette = silhouette_score(X_scaled, labels_hier)
    dbi = davies_bouldin_score(X_scaled, labels_hier)
    ari = adjusted_rand_score(y_true, labels_hier)
    
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Index: {dbi:.3f}")
    print(f"Adjusted Rand Index: {ari:.3f}")
    
    # Visualisation PCA
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_hier,
                         cmap='viridis', alpha=0.7, s=50)
    plt.title(f'Hierarchical Clustering (k={n_clusters})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(Path(outdir) / 'hierarchical_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Dendrogramme (sur échantillon pour lisibilité)
    n_samples = min(100, len(X_scaled))
    indices = np.random.choice(len(X_scaled), n_samples, replace=False)
    linkage_matrix = linkage(X_scaled[indices], method='ward')
    
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.savefig(Path(outdir) / 'hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Matrice de contingence
    contingency = pd.crosstab(labels_hier, y_true)
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Greens')
    plt.title('Hierarchical: Clusters vs True Classes')
    plt.xlabel('True Bug Type')
    plt.ylabel('Cluster')
    plt.savefig(Path(outdir) / 'hierarchical_contingency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return labels_hier


def compare_clusterings(labels_kmeans, labels_hier, y_true, X_pca, X_scaled, outdir):
    print("\n=== Comparaison des méthodes ===")
    
    # Calcul des métriques pour comparaison
    metrics_kmeans = {
        'Silhouette': silhouette_score(X_scaled, labels_kmeans),
        'Davies-Bouldin': davies_bouldin_score(X_scaled, labels_kmeans),
        'ARI': adjusted_rand_score(y_true, labels_kmeans)
    }
    
    metrics_hier = {
        'Silhouette': silhouette_score(X_scaled, labels_hier),
        'Davies-Bouldin': davies_bouldin_score(X_scaled, labels_hier),
        'ARI': adjusted_rand_score(y_true, labels_hier)
    }
    
    # Tableau récapitulatif
    metrics_df = pd.DataFrame({
        'KMeans': metrics_kmeans,
        'Hierarchical': metrics_hier
    }).T
    
    print("\nTableau récapitulatif des métriques:")
    print(metrics_df.round(3))
    
    # Visualisation des métriques
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Comparaison des métriques de clustering')
    ax.legend(loc='best')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Comparaison visuelle
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # KMeans
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans,
                   cmap='tab10', alpha=0.7, s=50)
    axes[0].set_title('KMeans')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    
    # Hierarchical
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_hier,
                   cmap='viridis', alpha=0.7, s=50)
    axes[1].set_title('Hierarchical')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    
    # True labels
    unique_labels = np.unique(y_true)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y_true_numeric = np.array([label_to_int[label] for label in y_true])
    
    axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true_numeric,
                   cmap='Paired', alpha=0.7, s=50)
    axes[2].set_title('True Classes')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Agreement entre méthodes
    agreement = adjusted_rand_score(labels_kmeans, labels_hier)
    print(f"\nAgreement between methods (ARI): {agreement:.3f}")
    
    # Sauvegarder les métriques
    metrics_df.to_csv(outdir / 'clustering_metrics.csv')
    
    return metrics_df


def main():
    print("ANALYSE DE CLUSTERING NON SUPERVISÉ")
    print("=" * 40)
    
    args = parse_arguments()
    
    # Création dossier sortie
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    
    # Chargement données
    df_merged, n_clusters = load_and_merge_data(args.csv, args.xlsx)
    
    # Préparation
    X_scaled, X_pca, y_true = prepare_data(df_merged)
    
    # KMeans
    labels_kmeans = kmeans_clustering(X_scaled, X_pca, y_true, n_clusters, outdir)
    
    # Hierarchical
    labels_hier = hierarchical_clustering(X_scaled, X_pca, y_true, n_clusters, outdir)
    
    # Comparaison
    metrics_df = compare_clusterings(labels_kmeans, labels_hier, y_true, X_pca, X_scaled, outdir)
    
    # Sauvegarde résultats
    results_df = pd.DataFrame({
        'ID': df_merged['ID'].values,
        'true_class': y_true,
        'kmeans_cluster': labels_kmeans,
        'hierarchical_cluster': labels_hier
    })
    results_df.to_csv(outdir / 'clustering_results.csv', index=False)
    
    print(f"\nTerminé. Résultats dans: {outdir}")


if __name__ == "__main__":
    main()