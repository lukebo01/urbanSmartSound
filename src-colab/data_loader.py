# src/data_loader.py

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_class_map(csv_path):
    """
    Carica la lista dei nomi delle classi, ordinata per classID, dal file CSV.

    Args:
        csv_path (str): Il percorso completo al file UrbanSound8K.csv.

    Returns:
        list: Una lista di stringhe con i nomi delle classi, es. ['air_conditioner', ...].
    """
    df_meta = pd.read_csv(csv_path)
    # 1. Prendi le colonne che ci interessano
    # 2. Rimuovi i duplicati per avere una sola voce per classe
    # 3. Ordina per classID per assicurare un ordine consistente
    # 4. Seleziona solo la colonna 'class'
    # 5. Convertila in una lista
    class_names = df_meta[['classID', 'class']].drop_duplicates().sort_values('classID')['class'].tolist()
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

# In src/data_loader.py

# =============================================================================
# === SOSTITUISCI LA TUA VECCHIA FUNZIONE load_vector_features CON QUESTA ===
# =============================================================================
def load_vector_features(file_path):
    """
    Carica e combina correttamente tutte le feature (1D e 2D) in un'unica 
    matrice di feature nel formato (timesteps, features) per un modello 1D.
    """
    with np.load(file_path, allow_pickle=True) as data:
        # Carica le feature 1D e assicurati che siano 2D (1, timesteps)
        # per la concatenazione.
        centroid = data['centroid'][np.newaxis, :]  # Shape (1, 173)
        rolloff = data['rolloff'][np.newaxis, :]    # Shape (1, 173)
        zcr = data['zcr'][np.newaxis, :]            # Shape (1, 173)
        
        # Carica le feature già 2D
        chroma = data['chroma']     # Shape (12, 173)
        contrast = data['contrast'] # Shape (7, 173)
        
        # Concatena tutte le feature lungo il primo asse (l'asse delle "feature")
        # Otterremo una matrice di shape (n_features_totali, n_timesteps)
        # n_features_totali = 1 (centroid) + 1 (rolloff) + 1 (zcr) + 12 (chroma) + 7 (contrast) = 22
        vector_feature_matrix = np.concatenate([centroid, rolloff, zcr, chroma, contrast], axis=0)
        
        # Trasponi la matrice per ottenere il formato desiderato dal modello 1D: (timesteps, features)
        # Da (22, 173) a (173, 22)
        vector_feature_transposed = vector_feature_matrix.T
        
        label = data['class_id'].item()
        
    return vector_feature_transposed, label
# =============================================================================

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

def get_train_test_split_from_folds(all_data, meta_file_path, test_size=0.2, random_state=42):
    """Crea uno split train/test stratificato per ID sorgente (fsID)."""
    # Ora la funzione usa il percorso che le viene passato
    df_meta = pd.read_csv(meta_file_path)
    
    # Questa parte del codice è complessa e assume un ordine specifico dei file.
    # Dobbiamo assicurarci che l'ordine dei dati in X_fold e y_fold corrisponda
    # all'ordine dei 'filenames_in_fold'. Questo è un punto critico.
    # La funzione 'collect_fold_data' deve garantire questo ordinamento.
    # Supponiamo che sia così.

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
    
    # Prendiamo i metadati e iteriamo su di essi, per trovare i dati corrispondenti
    # Questo è più sicuro che iterare sui dati e cercare i metadati.
    
    # Creiamo un dizionario per un accesso rapido ai dati caricati
    # Assumiamo che `collect_fold_data` carichi i file in ordine alfabetico, come fa `glob`
    data_map = {}
    for fold_name, (X_fold, y_fold) in all_data.items():
        fold_num = int(fold_name.replace('fold', ''))
        # Ottieni i nomi dei file per questo fold, ordinati
        filenames_in_fold = sorted(df_meta[df_meta['fold'] == fold_num]['slice_file_name'].tolist())
        for i, filename in enumerate(filenames_in_fold):
            data_map[filename] = (X_fold[i], y_fold[i])

    # Ora iteriamo sui set di ID e prendiamo i file giusti
    train_files = df_meta[df_meta['fsID'].isin(train_ids_set)]['slice_file_name']
    test_files = df_meta[df_meta['fsID'].isin(test_ids_set)]['slice_file_name']

    for filename in train_files:
        if filename in data_map:
            X, y = data_map[filename]
            X_train.append(X)
            y_train.append(y)

    for filename in test_files:
        if filename in data_map:
            X, y = data_map[filename]
            X_test.append(X)
            y_test.append(y)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    
    # ==============================================================================
# === NUOVA FUNZIONE PER LA PIPELINE AVANZATA (LEGGE LO STACK PRE-CALCOLATO) ===
# ==============================================================================
def load_precomputed_mfcc_stack(file_path):
    """
    Carica la feature 'mfcc_delta_stack' pre-calcolata da un file .npz.
    Questa feature è un tensore a 3 canali nel formato (n_mfcc, timesteps, 3).
    """
    with np.load(file_path, allow_pickle=True) as data:
        # Controlla se la chiave esiste per sicurezza
        if 'mfcc_delta_stack' not in data:
            raise KeyError(f"Chiave 'mfcc_delta_stack' non trovata in {file_path}. "
                           "Assicurati di aver eseguito il notebook di estrazione corretto.")
        
        # Carica direttamente il tensore a 3 canali
        feature_stack = data['mfcc_delta_stack'].astype(np.float32)
        
        # Carica l'etichetta
        label = data['class_id'].item()
        
    return feature_stack, label

# Nota: la funzione load_feature originale per mfcc caricherà solo la feature 'mfcc'
# a singolo canale, mantenendo la compatibilità con la tua Pipeline B originale.
# Questa nuova funzione è specifica per la pipeline avanzata.