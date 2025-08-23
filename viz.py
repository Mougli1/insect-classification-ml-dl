import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_data():
    features_df = pd.read_csv('output/features_robust_final.csv')
    labels_df = pd.read_excel('train/classif.xlsx')
    
    labels_df.columns = ['ID', 'bug type', 'species']
    labels_df['image_id'] = labels_df['ID']
    
    merged_df = features_df.merge(labels_df[['image_id', 'bug type', 'species']], on='image_id')
    
    feature_cols = [col for col in features_df.columns if col != 'image_id']
    X = merged_df[feature_cols].values
    
    return merged_df, X, feature_cols

def plot_pca_circle(X, feature_names):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(16, 16))
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Définir des couleurs par groupe de features
    feature_colors = {}
    for feat in feature_names:
        if 'color' in feat and '_R' in feat:
            feature_colors[feat] = 'red'
        elif 'color' in feat and '_G' in feat:
            feature_colors[feat] = 'green'
        elif 'color' in feat and '_B' in feat:
            feature_colors[feat] = 'blue'
        elif 'hu_' in feat:
            feature_colors[feat] = 'purple'
        elif 'glcm' in feat:
            feature_colors[feat] = 'orange'
        elif 'hist_' in feat:
            feature_colors[feat] = 'brown'
        elif 'circle' in feat or 'symmetry' in feat:
            feature_colors[feat] = 'darkgreen'
        else:
            feature_colors[feat] = 'gray'
    
    # Afficher toutes les features
    for i, feature in enumerate(feature_names):
        # Flèche
        arrow_scale = 3
        plt.arrow(0, 0, loadings[i, 0]*arrow_scale, loadings[i, 1]*arrow_scale, 
                 head_width=0.02, head_length=0.02, 
                 fc=feature_colors[feature], ec=feature_colors[feature], 
                 alpha=0.7, linewidth=1)
        
        # Calculer la position du texte
        text_scale = 3.4
        text_x = loadings[i, 0]*text_scale
        text_y = loadings[i, 1]*text_scale
        
        # Ajuster la taille de police selon l'importance
        loading_magnitude = np.sqrt(loadings[i, 0]**2 + loadings[i, 1]**2)
        fontsize = max(6, min(10, 6 + loading_magnitude * 4))
        
        plt.text(text_x, text_y, feature, 
                fontsize=fontsize, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='white', alpha=0.7, 
                         edgecolor=feature_colors[feature]))
    
    # Cercle unitaire
    circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=3, linestyle='--')
    plt.gca().add_patch(circle)
    
    # Légende pour les groupes de features
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Color R features'),
        Patch(facecolor='green', label='Color G features'),
        Patch(facecolor='blue', label='Color B features'),
        Patch(facecolor='purple', label='Hu moments'),
        Patch(facecolor='orange', label='GLCM features'),
        Patch(facecolor='brown', label='Histogram features'),
        Patch(facecolor='darkgreen', label='Shape/Symmetry features'),
        Patch(facecolor='gray', label='Other features')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    plt.title('PCA Loading Plot - Circle of Correlations (All Features)', fontsize=18, weight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=1)
    plt.axvline(x=0, color='k', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('output/pca_circle_correlations_all.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Créer une version zoomée pour les features importantes
    plt.figure(figsize=(14, 14))
    
    # Sélectionner les features importantes
    loading_scores = np.abs(loadings).sum(axis=1)
    top_features_idx = np.argsort(loading_scores)[-30:]
    
    for i in top_features_idx:
        arrow_scale = 3
        plt.arrow(0, 0, loadings[i, 0]*arrow_scale, loadings[i, 1]*arrow_scale, 
                 head_width=0.05, head_length=0.05, 
                 fc=feature_colors[feature_names[i]], ec=feature_colors[feature_names[i]], 
                 alpha=0.8, linewidth=2)
        
        text_scale = 3.3
        text_x = loadings[i, 0]*text_scale
        text_y = loadings[i, 1]*text_scale
        
        plt.text(text_x, text_y, feature_names[i], 
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='yellow', alpha=0.8,
                         edgecolor=feature_colors[feature_names[i]]))
    
    circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=3, linestyle='--')
    plt.gca().add_patch(circle)
    
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    plt.title('PCA Loading Plot - Circle of Correlations (Top 30 Features)', fontsize=18, weight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=1)
    plt.axvline(x=0, color='k', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('output/pca_circle_correlations_top30.png', bbox_inches='tight')
    plt.close()
    
    return pca

def plot_pca_projection(X, df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 10))
    
    bug_types = df['bug type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(bug_types)))
    
    for i, bug_type in enumerate(bug_types):
        mask = df['bug type'] == bug_type
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], label=bug_type, 
                   alpha=0.8, s=100, edgecolors='black', linewidth=0.5)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    plt.title('PCA Projection by Bug Type', fontsize=18, weight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/pca_projection_bug_type.png', bbox_inches='tight')
    plt.close()
    
    return X_pca

def plot_pca_variance(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca_all = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    
    explained_var = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)
    
    n_components_to_show = min(len(explained_var), len(explained_var))
    
    plt.bar(range(1, n_components_to_show+1), explained_var[:n_components_to_show], 
            alpha=0.7, color='steelblue', label='Individual variance', width=0.8)
    plt.plot(range(1, n_components_to_show+1), cumsum_var[:n_components_to_show], 
             'ro-', linewidth=3, markersize=8, label='Cumulative variance')
    
    n_95 = np.argmax(cumsum_var >= 0.95) + 1
    plt.axhline(y=0.95, color='g', linestyle='--', linewidth=2, label='95% variance')
    plt.axvline(x=n_95, color='g', linestyle='--', linewidth=1, alpha=0.7)
    plt.text(n_95+1, 0.93, f'{n_95} components\nfor 95% variance', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.xlabel('Principal Component', fontsize=14)
    plt.ylabel('Explained Variance Ratio', fontsize=14)
    plt.title('PCA - Variance Explained by Component', fontsize=18, weight='bold', pad=20)
    plt.legend(fontsize=12, loc='center right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/pca_variance_explained.png', bbox_inches='tight')
    plt.close()

def plot_pca_centroids(X, df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 10))
    
    bug_types = df['bug type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(bug_types)))
    
    for i, bug_type in enumerate(bug_types):
        mask = df['bug type'] == bug_type
        points = X_pca[mask]
        center = points.mean(axis=0)
        
        plt.scatter(points[:, 0], points[:, 1], c=[colors[i]], alpha=0.3, s=50)
        plt.scatter(center[0], center[1], c=[colors[i]], s=400, marker='*', 
                   edgecolors='black', linewidth=2, label=f'{bug_type} centroid')
        
        cov = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues)
        
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                         facecolor=colors[i], alpha=0.2, edgecolor=colors[i], linewidth=2)
        plt.gca().add_patch(ellipse)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    plt.title('PCA Projection with Centroids and Confidence Ellipses', fontsize=18, weight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/pca_centroids_ellipses.png', bbox_inches='tight')
    plt.close()

def plot_feature_importance(X, feature_names):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, max(8, len(feature_names)*0.25)))
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    feature_importance = np.abs(loadings).sum(axis=1)
    sorted_idx = np.argsort(feature_importance)
    
    y_pos = np.arange(len(feature_importance))
    bars = plt.barh(y_pos, feature_importance[sorted_idx], 
                    color=plt.cm.plasma(feature_importance[sorted_idx]/feature_importance.max()))
    
    plt.yticks(y_pos, [feature_names[i] for i in sorted_idx], fontsize=9)
    plt.xlabel('Feature Importance (Sum of Absolute Loadings)', fontsize=14)
    plt.title('All Features Ranked by PCA Importance', fontsize=18, weight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    
    for i, (idx, importance) in enumerate(zip(sorted_idx, feature_importance[sorted_idx])):
        plt.text(importance + 0.01, i, f'{importance:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('output/pca_feature_importance.png', bbox_inches='tight')
    plt.close()

def plot_species_by_bug_type(df):
    plt.figure(figsize=(16, 10))
    
    species_bug_type = df.groupby('species')['bug type'].agg(lambda x: x.mode()[0] if len(x) > 0 else 'Unknown')
    species_counts = df['species'].value_counts()
    
    species_data = pd.DataFrame({
        'species': species_counts.index,
        'count': species_counts.values,
        'bug_type': [species_bug_type[sp] for sp in species_counts.index]
    })
    
    bug_types = df['bug type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(bug_types)))
    color_map = {bt: colors[i] for i, bt in enumerate(bug_types)}
    
    species_data = species_data.sort_values('count', ascending=False)
    
    bars = plt.bar(range(len(species_data)), species_data['count'], 
                    color=[color_map[bt] for bt in species_data['bug_type']])
    
    plt.xticks(range(len(species_data)), species_data['species'], rotation=90, ha='right', fontsize=8)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Species by Their Unique Bug Type', fontsize=14, weight='bold')
    
    legend_elements = [plt.Rectangle((0,0),1,1, fc=color_map[bt], label=bt) for bt in bug_types]
    plt.legend(handles=legend_elements, title='Bug Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/species_by_bug_type_distribution.png', bbox_inches='tight')
    plt.close()

def plot_bug_distribution(df):
    fig = plt.figure(figsize=(20, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    bug_counts = df['bug type'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(bug_counts)))
    
    wedges, texts, autotexts = ax1.pie(bug_counts.values, autopct='%1.1f%%',
                                        colors=colors, startangle=90, pctdistance=0.85)
    
    for i, (wedge, text, autotext) in enumerate(zip(wedges, texts, autotexts)):
        autotext.set_color('black')
        autotext.set_weight('bold')
        autotext.set_fontsize(11)
    
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax1.add_artist(centre_circle)
    ax1.legend(wedges, bug_counts.index, title="Bug Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax1.set_title('Bug Type Distribution (Donut Chart)', fontsize=14, weight='bold')
    
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(bug_counts.index, bug_counts.values, color=colors)
    for bar, count in zip(bars, bug_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontsize=10, weight='bold')
    ax2.set_xlabel('Bug Type', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Count by Bug Type', fontsize=14, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    ax3 = plt.subplot(2, 3, 3)
    bug_type_species = df.groupby('bug type')['species'].nunique().sort_values(ascending=False)
    bars = ax3.bar(bug_type_species.index, bug_type_species.values, color=colors)
    for bar, count in zip(bars, bug_type_species.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontsize=10, weight='bold')
    ax3.set_xlabel('Bug Type', fontsize=12)
    ax3.set_ylabel('Number of Unique Species', fontsize=12)
    ax3.set_title('Species Diversity by Bug Type', fontsize=14, weight='bold')
    ax3.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    ax4 = plt.subplot(2, 3, 4)
    species_bug_type_unique = df.groupby('species')['bug type'].apply(lambda x: x.iloc[0] if len(x.unique()) == 1 else 'Multiple').reset_index()
    species_bug_type_unique.columns = ['species', 'unique_bug_type']
    
    unique_bug_type_counts = species_bug_type_unique['unique_bug_type'].value_counts()
    
    colors_unique = []
    for bt in unique_bug_type_counts.index:
        if bt == 'Multiple':
            colors_unique.append('gray')
        else:
            colors_unique.append(colors[list(bug_counts.index).index(bt)])
    
    wedges, texts, autotexts = ax4.pie(unique_bug_type_counts.values, labels=unique_bug_type_counts.index, 
                                        autopct='%1.1f%%', colors=colors_unique, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    ax4.set_title('Distribution of Species by Their Unique Bug Type', fontsize=14, weight='bold')
    
    ax5 = plt.subplot(2, 3, 5)
    species_count_by_type = df.groupby('species')['bug type'].nunique()
    species_single_type = (species_count_by_type == 1).sum()
    species_multiple_types = (species_count_by_type > 1).sum()
    
    labels = ['Single Bug Type', 'Multiple Bug Types']
    sizes = [species_single_type, species_multiple_types]
    colors_pie = ['lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = ax5.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                        colors=colors_pie, startangle=90)
    
    ax5.set_title('Species Classification: Single vs Multiple Bug Types', fontsize=14, weight='bold')
    
    ax6 = plt.subplot(2, 3, 6)
    species_counts = df['species'].value_counts().head(20)
    
    bars = ax6.barh(range(len(species_counts)), species_counts.values)
    ax6.set_yticks(range(len(species_counts)))
    ax6.set_yticklabels(species_counts.index, fontsize=9)
    
    for i, (count, species) in enumerate(zip(species_counts.values, species_counts.index)):
        bars[i].set_color(plt.cm.viridis(i/len(bars)))
        ax6.text(count + 0.5, i, str(count), va='center', fontsize=9)
    
    ax6.set_xlabel('Number of Samples', fontsize=12)
    ax6.set_title('Top 20 Species by Sample Count', fontsize=14, weight='bold')
    ax6.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/bug_distribution_complete.png', bbox_inches='tight')
    plt.close()

def plot_tsne_analysis(X, df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    perplexities = [5, 15, 30, 50]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    
    bug_types = df['bug type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(bug_types)))
    
    for idx, perp in enumerate(perplexities):
        print(f"  Computing t-SNE with perplexity={perp}...")
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        for j, bug_type in enumerate(bug_types):
            mask = df['bug type'] == bug_type
            axes[idx].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[j]], 
                            label=bug_type, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        axes[idx].set_title(f't-SNE (perplexity={perp})', fontsize=14, weight='bold')
        axes[idx].set_xlabel('t-SNE 1', fontsize=12)
        axes[idx].set_ylabel('t-SNE 2', fontsize=12)
        axes[idx].grid(alpha=0.3)
        
        if idx == 0:
            axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.suptitle('t-SNE Analysis with Different Perplexity Values', fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig('output/tsne_analysis_complete.png', bbox_inches='tight')
    plt.close()

def plot_isomap_analysis(X, df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_neighbors_list = [5, 15, 30, 50]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    
    bug_types = df['bug type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(bug_types)))
    
    for idx, n_neighbors in enumerate(n_neighbors_list):
        print(f"  Computing Isomap with n_neighbors={n_neighbors}...")
        isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
        X_isomap = isomap.fit_transform(X_scaled)
        
        for j, bug_type in enumerate(bug_types):
            mask = df['bug type'] == bug_type
            axes[idx].scatter(X_isomap[mask, 0], X_isomap[mask, 1], c=[colors[j]], 
                            label=bug_type, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        axes[idx].set_title(f'Isomap (n_neighbors={n_neighbors})', fontsize=14, weight='bold')
        axes[idx].set_xlabel('Isomap 1', fontsize=12)
        axes[idx].set_ylabel('Isomap 2', fontsize=12)
        axes[idx].grid(alpha=0.3)
        
        if idx == 0:
            axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.suptitle('Isomap Analysis with Different n_neighbors Values', fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig('output/isomap_analysis_complete.png', bbox_inches='tight')
    plt.close()

def plot_final_comparison(X, df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("  Computing final projections...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    isomap = Isomap(n_components=2, n_neighbors=15)
    X_isomap = isomap.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    projections = [X_pca, X_tsne, X_isomap]
    titles = [f'PCA\n(Explained variance: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%})', 
              't-SNE\n(perplexity=30)', 
              'Isomap\n(n_neighbors=15)']
    
    bug_types = df['bug type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(bug_types)))
    
    for ax, X_proj, title in zip(axes, projections, titles):
        for i, bug_type in enumerate(bug_types):
            mask = df['bug type'] == bug_type
            ax.scatter(X_proj[mask, 0], X_proj[mask, 1], c=[colors[i]], 
                     label=bug_type, alpha=0.8, s=80, edgecolors='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, weight='bold', pad=15)
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    plt.suptitle('Comparison of Dimensionality Reduction Methods', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('output/methods_comparison_final.png', bbox_inches='tight')
    plt.close()

def main():
    print("Loading data...")
    df, X, feature_names = load_data()
    
    print("\nCreating species by bug type distribution...")
    plot_species_by_bug_type(df)
    
    print("\nCreating distribution visualizations...")
    plot_bug_distribution(df)
    
    print("\nPerforming PCA analysis...")
    print("  - PCA variance explained...")
    plot_pca_variance(X)
    
    print("  - PCA circle of correlations...")
    plot_pca_circle(X, feature_names)
    
    print("  - PCA projection by bug type...")
    plot_pca_projection(X, df)
    
    print("  - PCA centroids and ellipses...")
    plot_pca_centroids(X, df)
    
    print("  - PCA feature importance...")
    plot_feature_importance(X, feature_names)
    
    print("\nPerforming t-SNE analysis...")
    plot_tsne_analysis(X, df)
    
    print("\nPerforming Isomap analysis...")
    plot_isomap_analysis(X, df)
    
    print("\nCreating final comparison...")
    plot_final_comparison(X, df)
    
    print("\nVisualization complete! Files saved in output/")
    print("Generated files:")
    print("  - species_by_bug_type_distribution.png")
    print("  - bug_distribution_complete.png")
    print("  - pca_variance_explained.png")
    print("  - pca_circle_correlations_all.png (with all features)")
    print("  - pca_circle_correlations_top30.png (with top 30 features)")
    print("  - pca_projection_bug_type.png")
    print("  - pca_centroids_ellipses.png")
    print("  - pca_feature_importance.png")
    print("  - tsne_analysis_complete.png")
    print("  - isomap_analysis_complete.png")
    print("  - methods_comparison_final.png")

if __name__ == "__main__":
    main()