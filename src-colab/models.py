
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# ==============================================================================
# === IMPLEMENTAZIONE MANUALE DELLA FOCAL LOSS ===
# ==============================================================================
# (Questa funzione rimane invariata, è già corretta e flessibile)
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    """
    Implementazione della Categorical Focal Loss come funzione utilizzabile da Keras.
    """
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss_weight = K.pow(1 - y_pred, gamma)
        focal_loss_value = alpha * loss_weight * cross_entropy
        return K.sum(focal_loss_value, axis=-1)
    return focal_loss
# ==============================================================================


# ==============================================================================
# === NUOVO MODELLO OTTIMIZZATO (2 GRU) PER PIPELINE B-TUNED ===
# ==============================================================================
def create_advanced_2d_cnn_3gru_model(input_shape, num_classes, pooling_type='max'):
    """
    Crea un modello ibrido ottimizzato con 2D-CNN + 2 GRU impilati.
    Questa versione riduce la complessità della parte ricorrente per evitare 
    l'overfitting su specifici pattern ritmici e affina la Focal Loss.
    """
    if pooling_type not in ['max', 'avg']:
        raise ValueError("pooling_type deve essere 'max' o 'avg'")
    
    PoolingLayer = layers.MaxPooling2D if pooling_type == 'max' else layers.AveragePooling2D
    
    model_input = keras.Input(shape=input_shape)
    
    # --- Parte CNN (invariata rispetto al modello advanced) ---
    x = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # --- Preparazione per la parte Ricorrente ---
    current_shape = x.shape
    x = layers.Reshape((current_shape[2], current_shape[1] * current_shape[3]))(x)
    
    # --- Parte GRU Gerarchica (2 Layer con Dropout aumentato) ---
    # 1. Primo layer GRU: return_sequences=True
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    # Dropout più aggressivo per una maggiore regolarizzazione
    x = layers.Dropout(0.5)(x) 

    # 2. Secondo (e ultimo) layer GRU: return_sequences=False
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    
    # --- Parte Dense (Classificatore) ---
    model_output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=model_input, outputs=model_output)
    
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    # --- Focal Loss Affinata: gamma ridotto a 1.5 ---
    loss_function = categorical_focal_loss(alpha=0.25, gamma=1.5)
        
    # ATTENZIONE: Questa loss si aspetta etichette in formato ONE-HOT.
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    
    return model

# --- NUOVO MODELLO PER LA PIPELINE C AVANZATA ---
def create_advanced_1d_cnn_2gru_model(input_shape, num_classes):
    """
    Crea un modello ibrido 1D-CNN + 2 GRU impilati (per Pipeline C Advanced).
    Incorpora gli stessi principi di ottimizzazione dei modelli 2D.
    """
    model_input = keras.Input(shape=input_shape)
    
    # --- Parte CNN 1D (leggermente potenziata) ---
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # --- Parte GRU Gerarchica (2 Layer) ---
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    
    # --- Parte Dense (Classificatore) ---
    model_output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=model_input, outputs=model_output)
    
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    loss_function = categorical_focal_loss(alpha=0.25, gamma=1.5)

    # Anche questo modello richiede etichette in formato ONE-HOT.
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    
    return model
# ==============================================================================
# === MODELLI PRECEDENTI (per riferimento e confronto) ===
# ==============================================================================

def create_2d_cnn_gru_model(input_shape, num_classes, pooling_type='max'):
    """
    Crea un modello ibrido 2D-CNN + GRU (versione originale per Pipeline A e B).
    """
    if pooling_type not in ['max', 'avg']:
        raise ValueError("pooling_type deve essere 'max' o 'avg'")
    
    PoolingLayer = layers.MaxPooling2D if pooling_type == 'max' else layers.AveragePooling2D
    
    model_input = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    current_shape = x.shape
    x = layers.Reshape((current_shape[2], current_shape[1] * current_shape[3]))(x)
    
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    
    model_output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=model_input, outputs=model_output)
    
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    
    # Questo modello usa ancora la loss standard
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_1d_cnn_gru_model(input_shape, num_classes):
    """
    Crea un modello ibrido 1D-CNN + GRU (versione originale per Pipeline C).
    """
    model_input = keras.Input(shape=input_shape)
    
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    
    model_output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=model_input, outputs=model_output)
    
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model