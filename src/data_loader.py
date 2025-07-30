# src/data_loader.py

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_class_map(meta_file_path="../data/raw/UrbanSound8K.csv"):
    """Carica la mappa delle classi dal file CSV."""
    df_meta = pd.read_csv(meta_file_path)
    class_map_df = df_meta[['classID', 'class']].drop_duplicates().sort_values('classID')
    class_names = dict(zip(class_map_df.classID, class_map_df['class']))
    return class_names

def load_feature(file_path, feature_key):
    """Carica una specifica feature da un file .npz."""
    with np.load(file_path, allow_pickle=True) as data:
        if feature_key not in data:
            raise KeyError(f"Chiave '{feature_key}' non trovata in {file_path}")
        
        feature = data[feature_key]
        # Aggiungi la dimensione del canale per le feature 2D
        if len(feature.shape) == 2 and feature_key in ['log_mel_spec', 'mfcc']:
            feature = feature[..., np.newaxis]
            
        label = data['class_id'].item()
    return feature, label

def load_vector_features(file_path):
    """
    Carica e combina correttamente tutte le feature (1D e 2D) in un'unica 
    matrice di feature nel formato (timesteps, features) per un modello 1D.
    """
    with np.load(file_path, allow_pickle=True) as data:
        # Carica le feature 1D e aggiungi un nuovo asse per renderle 2D (1, timesteps)
        centroid = data['centroid'][np.newaxis, :]  # Shape (1, 173)
        rolloff = data['rolloff'][np.newaxis, :]    # Shape (1, 173)
        zcr = data['zcr'][np.newaxis, :]            # Shape (1, 173)
        
        # Carica le feature che sono già 2D
        chroma = data['chroma']     # Shape (12, 173)
        contrast = data['contrast'] # Shape (7, 173)
        
        # Concatena tutte le feature lungo l'asse 0 (l'asse delle "feature")
        # Il risultato avrà shape (22, 173)
        vector_feature_matrix = np.concatenate([centroid, rolloff, zcr, chroma, contrast], axis=0)
        
        # Trasponi la matrice per ottenere il formato desiderato: (timesteps, features)
        # Da (22, 173) a (173, 22)
        vector_feature_transposed = vector_feature_matrix.T
        
        label = data['class_id'].item()
        
    return vector_feature_transposed, label

def collect_fold_data(fold_dir, feature_loader_fn, **kwargs):
    """Raccoglie tutti i dati da un fold usando una funzione di caricamento specifica."""
    npz_files = sorted(glob.glob(os.path.join(fold_dir, "*.npz")))
    all_features, all_labels = [], []
    
    for npz_file in npz_files:
        try:
            features, label = feature_loader_fn(npz_file, **kwargs)
            all_features.append(features)
            all_labels.append(label)
        except Exception as e:
            print(f"Errore nel caricare {os.path.basename(npz_file)}: {e}")
            
    return np.array(all_features), np.array(all_labels)

def get_train_test_split_from_folds(all_data, meta_file_path="../data/raw/UrbanSound8K.csv", test_size=0.2, random_state=42):
    """Crea uno split train/test stratificato per ID sorgente (fsID)."""
    df_meta = pd.read_csv(meta_file_path)
    filename_to_fsid = dict(zip(df_meta.slice_file_name, df_meta.fsID))
    all_original_ids = df_meta['fsID'].unique()

    train_ids, test_ids = train_test_split(
        all_original_ids, 
        test_size=test_size,       
        random_state=random_state
    )
    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)

    X_train, y_train, X_test, y_test = [], [], [], []

    for fold_name, (X_fold, y_fold) in all_data.items():
        fold_num = int(fold_name.replace('fold', ''))
        filenames_in_fold = df_meta[df_meta['fold'] == fold_num]['slice_file_name'].tolist()
        
        for i, filename_wav in enumerate(filenames_in_fold):
            fsid = filename_to_fsid.get(filename_wav)
            if fsid in train_ids_set:
                X_train.append(X_fold[i])
                y_train.append(y_fold[i])
            elif fsid in test_ids_set:
                X_test.append(X_fold[i])
                y_test.append(y_fold[i])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)