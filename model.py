#!/usr/bin/env python3
"""
models.py - Classification complète des bug types
=================================================

Script conforme aux exigences IG.2412/IG.2411 pour construire, évaluer et sauvegarder 
plusieurs modèles à partir des features CSV et étiquettes XLSX.

Usage:
    python models.py --csv output/features_robust_final.csv --xlsx train/classif.xlsx
    python models.py --csv features.csv --xlsx labels.xlsx --outdir my_models
"""

import pandas as pd
import numpy as np
import argparse
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import classification_report, confusion_matrix, f1_score, silhouette_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Supprimer spécifiquement les warnings de sklearn
import os
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

# Reproductibilité
np.random.seed(42)

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Classification des bug types')
    parser.add_argument('--csv', default='/Users/mouloudmerbouche/Library/Mobile Documents/com~apple~CloudDocs/Documents/ia+ml2/output/selected_feature_sets_rfe_subset/features_rfe_top_15_subset.csv', help='Chemin vers le CSV de features (IDs 1-250)')
    parser.add_argument('--xlsx', default='train/classif.xlsx', help='Chemin vers classif.xlsx')
    parser.add_argument('--outdir', default='output_models', help='Dossier de sortie')
    parser.add_argument('--predict', help='CSV de test pour prédictions (IDs 251-347)')
    return parser.parse_args()

def load_and_merge_data(csv_path, xlsx_path):
    """Charge et fusionne les données"""
    print("📂 Chargement des données...")
    
    # Lire CSV features
    try:
        df_feat = pd.read_csv(csv_path)
        print(f"   CSV features chargé: {len(df_feat)} lignes, {len(df_feat.columns)} colonnes")
    except Exception as e:
        raise ValueError(f"Erreur lecture CSV: {e}")
    
    # Lire XLSX labels (feuille 1)
    try:
        df_lbl = pd.read_excel(xlsx_path, sheet_name=0)  # Feuille 1
        print(f"   XLSX labels chargé: {len(df_lbl)} lignes, {len(df_lbl.columns)} colonnes")
    except Exception as e:
        raise ValueError(f"Erreur lecture XLSX: {e}")
    
    # Vérifications
    if 'ID' not in df_feat.columns:
        if 'image_id' in df_feat.columns:
            df_feat = df_feat.rename(columns={'image_id': 'ID'})
        else:
            raise ValueError("Colonne 'ID' ou 'image_id' manquante dans le CSV")
    
    if 'ID' not in df_lbl.columns:
        raise ValueError("Colonne 'ID' manquante dans le XLSX")
    
    if 'bug type' not in df_lbl.columns:
        raise ValueError("Colonne 'bug type' manquante dans le XLSX")
    
    # Vérifier que ID est entier
    df_feat['ID'] = df_feat['ID'].astype(int)
    df_lbl['ID'] = df_lbl['ID'].astype(int)
    
    # Fusion inner join
    df_merged = df_feat.merge(df_lbl[['ID', 'bug type']], on='ID', how='inner')
    
    # Affichage stats
    print(f"\n📊 STATISTIQUES FUSION:")
    print(f"   • N images fusionnées : {len(df_merged)} / 250")
    print(f"   • Distribution bug_type :")
    
    bug_counts = df_merged['bug type'].value_counts().sort_index()
    for bug_type, count in bug_counts.items():
        print(f"     {bug_type:12s} ... {count:3d}")
    
    # Contrôle qualité
    missing_count = df_merged.isnull().sum().sum()
    if missing_count > 0:
        print(f"\n ERREUR: {missing_count} valeurs manquantes détectées")
        print("Détail des valeurs manquantes par colonne:")
        missing_by_col = df_merged.isnull().sum()
        for col, count in missing_by_col[missing_by_col > 0].items():
            print(f"   {col}: {count}")
        raise ValueError("Valeurs manquantes détectées - arrêt du script")
    
    print(f"   Aucune valeur manquante")
    
    return df_merged

def check_and_clean_data(X_train, X_test):
    """Vérifie et nettoie les données pour éviter les problèmes numériques"""
    print("\n🔍 Vérification des données...")
    
    # Vérifier les valeurs infinies ou NaN
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print(" Valeurs NaN ou Inf détectées dans X_train")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
    
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        print("  Valeurs NaN ou Inf détectées dans X_test")
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Vérifier l'échelle des données
    max_abs_train = np.abs(X_train).max()
    max_abs_test = np.abs(X_test).max()
    
    if max_abs_train > 1e10 or max_abs_test > 1e10:
        print(f"  Valeurs très grandes détectées (max: {max(max_abs_train, max_abs_test):.2e})")
        print("   → Application d'un clipping pour éviter overflow")
        X_train = np.clip(X_train, -1e10, 1e10)
        X_test = np.clip(X_test, -1e10, 1e10)
    
    print(f" Données vérifiées - échelle max: {np.abs(X_train).max():.2e}")
    
    return X_train, X_test

def prepare_data(df_merged):
    """Prépare les données pour l'entraînement"""
    print("\n🔧 PRÉ-PROCESSING...")
    
    # Séparation X / y
    feature_columns = [col for col in df_merged.columns if col not in ['ID', 'bug type']]
    X = df_merged[feature_columns]
    y = df_merged['bug type']
    
    print(f"   • Features (X): {X.shape}")
    print(f"   • Labels (y): {y.shape}")
    print(f"   • Classes uniques: {sorted(y.unique())}")
    
    # Vérifier les classes avec trop peu d'échantillons (augmenté à 5)
    min_samples_per_class = 5
    class_counts = y.value_counts()
    classes_to_remove = class_counts[class_counts < min_samples_per_class].index.tolist()
    
    if classes_to_remove:
        print(f" Classes supprimées (< {min_samples_per_class} échantillons): {classes_to_remove}")
        # Supprimer les classes problématiques
        mask_valid = ~y.isin(classes_to_remove)
        X = X[mask_valid]
        y = y[mask_valid]
        print(f"   • Après filtrage: {X.shape[0]} échantillons, {len(y.unique())} classes")
    
    # Encodage y
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    
    print(f"   • Mapping classes finales:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(y == class_name)
        print(f"     {i} → {class_name:12s} ({count:3d} échantillons)")
    
    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # Nettoyer les données pour éviter les problèmes numériques
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    X_train_clean, X_test_clean = check_and_clean_data(X_train_np, X_test_np)
    
    # Reconvertir en DataFrame si nécessaire
    if hasattr(X_train, 'values'):
        X_train = pd.DataFrame(X_train_clean, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_clean, columns=X_test.columns, index=X_test.index)
    else:
        X_train = X_train_clean
        X_test = X_test_clean
    
    print(f"   • Train: {X_train.shape[0]} échantillons")
    print(f"   • Test:  {X_test.shape[0]} échantillons")
    
    return X_train, X_test, y_train, y_test, label_encoder, X, y_enc, feature_columns

def create_models_config():
    """Configuration des modèles et hyperparamètres"""
    return {
        'SVC': {
            'model': SVC(class_weight='balanced', random_state=42),  # Ajout class_weight
            'params': {
                'model__C': np.logspace(-3, 2, 10),  # Réduit de 20 à 10 valeurs
                'model__gamma': ['scale', 'auto'],
                'model__kernel': ['rbf', 'linear']  # Ajout kernel linear
            },
            'search': 'grid',
            'type': 'supervised',
            'use_scaler': True
        },
        'KNeighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': [3, 5, 7, 9, 11],  # Supprimé 2 car trop petit
                'model__weights': ['uniform', 'distance'],
                'model__metric': ['euclidean', 'cosine']  # Ajout metric cosine
            },
            'search': 'grid',
            'type': 'supervised',
            'use_scaler': True
        },
        'RandomForest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [None, 10, 20, 30, 50],  # Ajout None et valeurs plus élevées
                'model__min_samples_leaf': [1, 2, 4],  # Nouveau paramètre
                'model__max_features': ['sqrt', 'log2', None]  # Nouveau paramètre
            },
            'search': 'grid',  # Changé de random à grid
            'type': 'supervised',
            'use_scaler': False  # Pas de scaler pour RF
        },
        'LogisticRegression': {
            'model': LogisticRegression(class_weight='balanced', random_state=42, max_iter=5000),  # Augmenté max_iter
            'params': {
                'model__C': [0.001, 0.01, 0.1, 1.0],  # Valeurs plus petites pour éviter overflow
                'model__solver': ['lbfgs', 'liblinear'],  # Retiré saga qui est instable
                'model__penalty': ['l2']
            },
            'search': 'grid',
            'type': 'supervised',
            'use_scaler': True
        },
        'LogisticRegression_L1': {
            'model': LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', 
                                      random_state=42, max_iter=5000),
            'params': {
                'model__C': [0.01, 0.1, 1.0]  # Valeurs plus conservatrices
            },
            'search': 'grid',
            'type': 'supervised',
            'use_scaler': True
        },
        'KMeans': {
            'model': KMeans(random_state=42),
            'params': {},
            'search': None,
            'type': 'clustering'
        },
        'AgglomerativeClustering': {
            'model': AgglomerativeClustering(linkage='ward'),
            'params': {},
            'search': None,
            'type': 'clustering'
        }
    }

def train_supervised_model(name, config, X_train, X_test, y_train, y_test, n_classes, feature_names):
    """Entraîne un modèle supervisé avec recherche d'hyperparamètres"""
    print(f"\n=== {name} ===")
    
    # Pipeline avec ou sans StandardScaler selon le modèle
    if config.get('use_scaler', True):
        # Utiliser RobustScaler pour LogisticRegression pour éviter les overflow
        if 'LogisticRegression' in name:
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('model', config['model'])
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', config['model'])
            ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    if config['search'] and config['params']:
        # Recherche d'hyperparamètres avec GridSearchCV uniquement
        search = GridSearchCV(
            pipeline, config['params'], 
            cv=5, scoring='f1_macro', 
            n_jobs=-1,
            error_score='raise'  # Pour voir les erreurs
        )
        
        try:
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            cv_score_mean = search.best_score_
            cv_score_std = search.cv_results_['std_test_score'][search.best_index_]
            best_params = search.best_params_
            
            print(f"meilleur score CV (f1_macro) : {cv_score_mean:.3f} ± {cv_score_std:.3f}")
            print(f"params : {best_params}")
            
        except Exception as e:
            print(f"Problème lors de l'entraînement: {str(e)[:100]}...")
            # Essayer avec des paramètres plus conservateurs
            if 'LogisticRegression' in name:
                print("   → Réessai avec paramètres plus conservateurs")
                pipeline_simple = Pipeline([
                    ('scaler', RobustScaler()),
                    ('model', LogisticRegression(C=0.1, max_iter=10000, 
                                               class_weight='balanced', 
                                               random_state=42, solver='liblinear'))
                ])
                pipeline_simple.fit(X_train, y_train)
                best_pipeline = pipeline_simple
                cv_score_mean = np.nan
                cv_score_std = np.nan
                best_params = {'C': 0.1, 'solver': 'liblinear'}
            else:
                raise
    else:
        # Pas de recherche d'hyperparamètres
        best_pipeline = pipeline
        best_pipeline.fit(X_train, y_train)
        cv_score_mean = np.nan
        cv_score_std = np.nan
        best_params = {}
        print("Pas de recherche d'hyperparamètres")
    
    # Évaluation sur test
    y_pred = best_pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"Test f1_macro : {test_f1:.3f}")
    print(f"Test accuracy : {test_acc:.3f}")
    print("-" * 30)
    
    return {
        'name': name,
        'pipeline': best_pipeline,
        'cv_f1_mean': cv_score_mean,
        'cv_f1_std': cv_score_std,
        'test_f1': test_f1,
        'test_acc': test_acc,
        'best_params': best_params,
        'y_pred': y_pred,
        'feature_names': feature_names
    }

def train_clustering_model(name, config, X_train, X_test, y_train, y_test, n_classes):
    """Entraîne un modèle de clustering"""
    print(f"\n=== {name} ===")
    
    # Pipeline avec StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configuration clustering
    if name == 'KMeans':
        model = KMeans(n_clusters=n_classes, random_state=42)
    else:  # AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=n_classes, linkage='ward')
    
    # Entraînement
    cluster_labels_train = model.fit_predict(X_train_scaled)
    
    # Prédiction test
    if hasattr(model, 'predict'):
        cluster_labels_test = model.predict(X_test_scaled)
    else:
        # Pour AgglomerativeClustering, ré-entraîner sur test
        cluster_labels_test = model.fit_predict(X_test_scaled)
    
    # Silhouette score
    if len(X_test_scaled) > 1:
        sil_score = silhouette_score(X_test_scaled, cluster_labels_test)
    else:
        sil_score = np.nan
    
    print(f"Silhouette score global : {sil_score:.3f}")
    
    # Table de contingence
    print("Table de contingence {cluster ↔ bug_type}:")
    contingency = pd.crosstab(cluster_labels_test, y_test, margins=True)
    print(contingency)
    print("-" * 30)
    
    # Pipeline pour cohérence
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    return {
        'name': name,
        'pipeline': pipeline,
        'cv_f1_mean': np.nan,
        'cv_f1_std': np.nan,
        'test_f1': np.nan,  # Pas applicable pour clustering
        'test_acc': np.nan,
        'silhouette': sil_score,
        'best_params': {},
        'y_pred': cluster_labels_test,
        'contingency': contingency
    }

def evaluate_model(result, y_test, label_encoder, outdir):
    """Évalue un modèle et génère les rapports"""
    name = result['name']
    y_pred = result['y_pred']
    
    print(f"\nÉVALUATION DÉTAILLÉE - {name}")
    
    # Correction du bug principal : utiliser np.isnan au lieu de is not np.nan
    if not np.isnan(result['test_f1']):  # Modèle supervisé
        # Classification report
        report = classification_report(y_test, y_pred, 
                                     target_names=label_encoder.classes_, 
                                     zero_division=0)
        print("Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(cm)
        
        # Sauvegarde figure confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True class')
        plt.xlabel('Predict class')
        plt.tight_layout()
        
        cm_path = Path(outdir) / f'confusion_matrix_{name.lower().replace(" ", "_")}.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Matrice sauvegardée: {cm_path}")
        
        # Features importantes pour RandomForest avec vrais noms
        if name == 'RandomForest' and hasattr(result['pipeline'].named_steps['model'], 'feature_importances_'):
            importances = result['pipeline'].named_steps['model'].feature_importances_
            feature_names = result.get('feature_names', [f'feature_{i}' for i in range(len(importances))])
            
            # Top 10 features
            indices = np.argsort(importances)[::-1][:10]
            print(f"\n🔝 Top 10 features importantes (RandomForest):")
            for i, idx in enumerate(indices):
                print(f"   {i+1:2d}. {feature_names[idx]:30s} : {importances[idx]:.4f}")

def train_all_models(X_train, X_test, y_train, y_test, label_encoder, feature_names):
    """Entraîne tous les modèles"""
    print("\nENTRAÎNEMENT DES MODÈLES")
    
    models_config = create_models_config()
    results = []
    n_classes = len(label_encoder.classes_)
    
    for name, config in models_config.items():
        try:
            if config['type'] == 'supervised':
                result = train_supervised_model(name, config, X_train, X_test, y_train, y_test, n_classes, feature_names)
            else:  # clustering
                result = train_clustering_model(name, config, X_train, X_test, y_train, y_test, n_classes)
            
            results.append(result)
            
        except Exception as e:
            print(f"Erreur {name}: {e}")
            continue
    
    return results

def select_best_model(results):
    """Sélectionne le meilleur modèle basé sur f1_macro"""
    print("\nSÉLECTION DU MEILLEUR MODÈLE")
    
    # Filtrer les modèles supervisés avec score valide
    supervised_results = [r for r in results if not np.isnan(r['test_f1'])]
    
    if not supervised_results:
        print(" Aucun modèle supervisé valide")
        return None
    
    # Meilleur score f1_macro
    best_result = max(supervised_results, key=lambda x: x['test_f1'])
    
    print(f">>> Modèle retenu : {best_result['name']} (f1_macro = {best_result['test_f1']:.3f})")
    
    return best_result

def save_results(results, best_result, outdir):
    """Sauvegarde tous les résultats"""
    print(f"\nSAUVEGARDE RÉSULTATS")
    
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    
    # Log JSON
    log_data = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'models': []
    }
    
    for result in results:
        model_data = {
            'name': result['name'],
            'cv_f1_mean': float(result['cv_f1_mean']) if not np.isnan(result['cv_f1_mean']) else None,
            'cv_f1_std': float(result['cv_f1_std']) if not np.isnan(result['cv_f1_std']) else None,
            'test_f1': float(result['test_f1']) if not np.isnan(result['test_f1']) else None,
            'test_acc': float(result['test_acc']) if not np.isnan(result['test_acc']) else None,
            'best_params': result['best_params']
        }
        
        if 'silhouette' in result:
            model_data['silhouette'] = float(result['silhouette']) if not np.isnan(result['silhouette']) else None
        
        log_data['models'].append(model_data)
    
    if best_result:
        log_data['best_model'] = {
            'name': best_result['name'],
            'test_f1': float(best_result['test_f1']),
            'test_acc': float(best_result['test_acc'])
        }
    
    log_path = outdir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"   📋 Log sauvegardé: {log_path}")
    
    return log_path

def retrain_and_save_best_model(best_result, X_full, y_full, label_encoder, outdir):
    """Re-entraîne le meilleur modèle sur toutes les données et sauvegarde"""
    print(f"\nENTRAÎNEMENT FINAL & PERSISTANCE")
    
    if best_result is None:
        print(" Aucun meilleur modèle à sauvegarder")
        return None
    
    # Re-fit sur tout le dataset
    print(f"   • Re-entraînement {best_result['name']} sur {len(X_full)} échantillons...")
    final_pipeline = best_result['pipeline']
    final_pipeline.fit(X_full, y_full)
    
    # Sauvegarde
    outdir = Path(outdir)
    pipeline_path = outdir / 'best_pipeline.joblib'
    encoder_path = outdir / 'label_encoder.joblib'
    
    joblib.dump(final_pipeline, pipeline_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"   ✔️ Pipeline sauvegardé: {pipeline_path}")
    print(f"   ✔️ Encodeur sauvegardé: {encoder_path}")
    
    return final_pipeline

def predict_test_set(csv_test_path, outdir, feature_columns):
    """Prédit sur le set de test 251-347"""
    print(f"\n PRÉDICTION SUR SET TEST")
    
    outdir = Path(outdir)
    pipeline_path = outdir / 'best_pipeline.joblib'
    encoder_path = outdir / 'label_encoder.joblib'
    
    # Vérifier fichiers
    if not pipeline_path.exists():
        print(f" Pipeline non trouvé: {pipeline_path}")
        return
    
    if not encoder_path.exists():
        print(f" Encodeur non trouvé: {encoder_path}")
        return
    
    # Charger modèle et encodeur
    pipeline = joblib.load(pipeline_path)
    label_encoder = joblib.load(encoder_path)
    print(f"    Pipeline chargé")
    
    # Lire données test
    try:
        df_test = pd.read_csv(csv_test_path)
        print(f"   Test CSV chargé: {len(df_test)} lignes")
    except Exception as e:
        print(f" Erreur lecture CSV test: {e}")
        return
    
    # Préparer features
    if 'ID' not in df_test.columns:
        if 'image_id' in df_test.columns:
            df_test = df_test.rename(columns={'image_id': 'ID'})
        else:
            print(" Colonne 'ID' manquante dans test CSV")
            return
    
    test_ids = df_test['ID'].astype(int)
    
    # Utiliser les mêmes colonnes que pour l'entraînement
    X_test = df_test[feature_columns]
    
    # Prédictions
    y_pred_encoded = pipeline.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    
    # Créer submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'bug type': y_pred_labels
    })
    
    # Ordonner par ID croissant
    submission = submission.sort_values('ID').reset_index(drop=True)
    
    # Sauvegarder
    submission_path = outdir / 'submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f" submission.csv généré – {len(submission)} lignes")
    print(f" Fichier: {submission_path}")
    
    # Afficher distribution prédictions
    print(f"   Distribution prédictions:")
    pred_counts = submission['bug type'].value_counts().sort_index()
    for bug_type, count in pred_counts.items():
        print(f"     {bug_type:12s} ... {count:3d}")

def main():
    """Fonction principale"""
    print("CLASSIFICATION DES BUG TYPES")
    print("=" * 50)
    
    # Arguments
    args = parse_arguments()
    
    try:
        # 1. Chargement & fusion
        df_merged = load_and_merge_data(args.csv, args.xlsx)
        
        # 2. Pré-processing
        X_train, X_test, y_train, y_test, label_encoder, X_full, y_full, feature_columns = prepare_data(df_merged)
        
        # 3. Entraînement modèles
        results = train_all_models(X_train, X_test, y_train, y_test, label_encoder, feature_columns)
        
        # 4. Évaluation détaillée
        outdir = Path(args.outdir)
        outdir.mkdir(exist_ok=True)
        
        for result in results:
            evaluate_model(result, y_test, label_encoder, outdir)
        
        # 5. Sélection meilleur modèle
        best_result = select_best_model(results)
        
        # 6. Sauvegarde résultats
        save_results(results, best_result, outdir)
        
        # 7. Entraînement final & persistance
        final_pipeline = retrain_and_save_best_model(best_result, X_full, y_full, label_encoder, outdir)
        
        # 8. Prédiction test set si fourni
        if args.predict:
            predict_test_set(args.predict, outdir, feature_columns)
        
        print(f"\ end (finiiiii) ! Résultats dans: {outdir}")
        
    except Exception as e:
        print(f"\ ERREUR: {e}")
        raise

if __name__ == "__main__":
    main()