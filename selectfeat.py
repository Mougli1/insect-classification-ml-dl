import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Configuration
ORIGINAL_FEATURES_CSV_PATH = Path("output/features_robust_final.csv")
TARGET_XLSX_PATH = Path("train/classif.xlsx")
XLSX_ID_COL_ORIGINAL = "ID"
XLSX_TARGET_COL_ORIGINAL = "bug type"
ID_COLUMN_NAME_IN_FEATURES_CSV = 'image_id'
OUTPUT_DIR = Path("output/rfe_analysis")

def load_data():
    """Load data for analysis."""
    print("Loading data...")
    
    # Load features
    features_df = pd.read_csv(ORIGINAL_FEATURES_CSV_PATH)
    print(f"Features loaded: {features_df.shape}")
    
    # Load labels
    target_df = pd.read_excel(TARGET_XLSX_PATH)
    print(f"Labels loaded: {target_df.shape}")
    
    # Rename columns
    target_df = target_df.rename(columns={
        XLSX_ID_COL_ORIGINAL: 'image_id',
        XLSX_TARGET_COL_ORIGINAL: 'target'
    })
    
    # Handle ID column in features_df if necessary
    if ID_COLUMN_NAME_IN_FEATURES_CSV != 'image_id':
        features_df = features_df.rename(columns={ID_COLUMN_NAME_IN_FEATURES_CSV: 'image_id'})
    
    # Merge data
    merged_df = pd.merge(features_df, target_df[['image_id', 'target']], 
                         on='image_id', how='inner')
    print(f"Merged data: {merged_df.shape}")
    
    # Prepare X and y
    feature_columns = [col for col in merged_df.columns 
                      if col not in ['image_id', 'target']]
    
    X = merged_df[feature_columns].values
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(merged_df['target'])
    
    print(f"\nFinal dimensions: X: {X.shape}, y: {y.shape}")
    print(f"Classes: {le.classes_}")
    
    return X, y, feature_columns

def create_justification_graph(X, y):
    """
    Create performance graph as a function of feature count.
    """
    print("\n" + "="*60)
    print("FEATURE SELECTION PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Préparer les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Configuration
    estimator = LogisticRegression(solver='liblinear', C=0.1, random_state=42, max_iter=300)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
    
    # Tester différents nombres de features
    n_range = [5, 10, 15, 20, 25, 30]  # Points à tester
    scores = []  # Pour stocker les accuracies moyennes
    stds = []    # Pour stocker les écarts-types
    
    print("\nComputing performance...")
    for n in n_range:
        if n <= X.shape[1]:
            # Select the n best features with RFE
            print(f"  Testing with {n} features...")
            rfe = RFE(estimator, n_features_to_select=n, step=1)
            rfe.fit(X_scaled, y)
            
            # Transform data with only these features
            X_selected = rfe.transform(X_scaled)
            
            # Cross-validation
            cv_scores = cross_val_score(estimator, X_selected, y, cv=cv, scoring='accuracy')
            scores.append(cv_scores.mean())
            stds.append(cv_scores.std())
            
            print(f"    → Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Créer le graphique
    plt.figure(figsize=(12, 8))
    
    # Courbe principale avec barres d'erreur
    plt.errorbar(n_range[:len(scores)], scores, yerr=stds, 
                marker='o', linewidth=2, markersize=8, capsize=5)
    
    # Mettre en évidence le point à 15 features
    idx_15 = n_range.index(15)
    plt.plot(15, scores[idx_15], 'ro', markersize=15, zorder=5)
    plt.annotate(f'15 features\n{scores[idx_15]:.4f}', 
                xy=(15, scores[idx_15]), 
                xytext=(17, scores[idx_15] - 0.01),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red')
    
    # Ajouter les zones colorées
    plt.axvspan(5, 10, alpha=0.2, color='red', label='Too few features')
    plt.axvspan(20, 30, alpha=0.2, color='orange', label='Diminishing returns')
    plt.axvspan(10, 20, alpha=0.2, color='green', label='Optimal zone')
    
    # Configuration du graphique
    plt.xlabel('Number of features', fontsize=12)
    plt.ylabel('Cross-validation accuracy', fontsize=12)
    plt.title('Model performance as a function of feature count', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Display exact values
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    for i, n in enumerate(n_range[:len(scores)]):
        print(f"{n:2d} features : {scores[i]:.4f} ± {stds[i]:.4f}")
    
    # Specific analysis for 15 features
    print("\n" + "="*50)
    print("ANALYSIS FOR 15 FEATURES")
    print("="*50)
    print(f"Performance at 15 features: {scores[idx_15]:.4f} (±{stds[idx_15]:.4f})")
    
    # Comparisons
    idx_10 = n_range.index(10)
    idx_20 = n_range.index(20)
    print(f"\nComparisons:")
    print(f"  vs 10 features: {scores[idx_15] - scores[idx_10]:+.4f} ({(scores[idx_15] - scores[idx_10])/scores[idx_10]*100:+.1f}%)")
    print(f"  vs 20 features: {scores[idx_15] - scores[idx_20]:+.4f} ({(scores[idx_15] - scores[idx_20])/scores[idx_20]*100:+.1f}%)")
    
    # Sauvegarder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'justification_15_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Obtenir les 15 meilleures features
    print("\n" + "="*50)
    print("TOP 15 FEATURES SELECTED")
    print("="*50)
    
    # RFE avec exactement 15 features pour obtenir la liste
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)
    estimator_final = LogisticRegression(solver='liblinear', C=0.1, random_state=42, max_iter=300)
    
    rfe_15 = RFE(estimator_final, n_features_to_select=15, step=1)
    rfe_15.fit(X_scaled_final, y)
    
    # Retourner aussi les indices des features sélectionnées
    selected_indices = np.where(rfe_15.support_)[0]
    
    return n_range[:len(scores)], scores, stds, selected_indices

def main():
    """Main function."""
    try:
        # Load data
        X, y, feature_names = load_data()
        
        # Create justification graph
        n_values, scores, stds, selected_indices = create_justification_graph(X, y)
        
        # Afficher les 15 features sélectionnées
        print("\nThe 15 selected features are:")
        selected_features = [feature_names[i] for i in selected_indices]
        for i, feat in enumerate(selected_features, 1):
            print(f"  {i:2d}. {feat}")
        
        # Sauvegarder la liste des 15 features
        pd.DataFrame({'feature': selected_features}).to_csv(
            OUTPUT_DIR / 'selected_15_features.csv', index=False
        )
        print(f"\n List of 15 features saved to: {OUTPUT_DIR / 'selected_15_features.csv'}")
        
        # Optional: test more values to verify
        print("\n" + "="*50)
        print("VERIFICATION WITH MORE POINTS (optional)")
        print("="*50)
        response = input("Do you want to test more values? (y/n): ")
        
        if response.lower() == 'y':
            # Test with all values from 5 to 30
            print("\nExhaustive test from 5 to 30 features...")
            n_range_detailed = list(range(5, min(31, X.shape[1]+1)))
            scores_detailed = []
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            estimator = LogisticRegression(solver='liblinear', C=0.1, random_state=42, max_iter=300)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for n in n_range_detailed:
                rfe = RFE(estimator, n_features_to_select=n, step=1)
                rfe.fit(X_scaled, y)
                X_selected = rfe.transform(X_scaled)
                cv_scores = cross_val_score(estimator, X_selected, y, cv=cv, scoring='accuracy')
                scores_detailed.append(cv_scores.mean())
                print(f"  {n:2d} features: {cv_scores.mean():.4f}")
            
            # Nouveau graphique détaillé
            plt.figure(figsize=(12, 6))
            plt.plot(n_range_detailed, scores_detailed, 'b-o', markersize=5)
            max_idx = np.argmax(scores_detailed)
            plt.plot(n_range_detailed[max_idx], scores_detailed[max_idx], 'ro', markersize=10)
            plt.annotate(f'Max: {n_range_detailed[max_idx]} features\n{scores_detailed[max_idx]:.4f}', 
                        xy=(n_range_detailed[max_idx], scores_detailed[max_idx]),
                        xytext=(n_range_detailed[max_idx]+2, scores_detailed[max_idx]-0.005),
                        arrowprops=dict(arrowstyle='->', color='red'))
            plt.xlabel('Number of features')
            plt.ylabel('Accuracy')
            plt.title('Cross-validation accuracy for all feature counts (5-30)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        print("\n Analysis completed!")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check dependencies
    try:
        import openpyxl
    except ImportError:
        print("  Please install openpyxl: pip install openpyxl")
        exit(1)
    
    main()