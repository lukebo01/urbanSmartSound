# src/models.py

from tensorflow import keras
from tensorflow.keras import layers

def create_2d_cnn_gru_model(input_shape, num_classes, pooling_type='max'):
    """
    Crea un modello ibrido 2D-CNN + GRU, ispirato dalla letteratura.
    Args:
        input_shape (tuple): La forma dell'input (altezza, larghezza, canali).
        num_classes (int): Numero di classi di output.
        pooling_type (str): 'max' per MaxPooling2D, 'avg' per AveragePooling2D.
    """
    # Validazione del tipo di pooling
    if pooling_type not in ['max', 'avg']:
        raise ValueError("pooling_type deve essere 'max' o 'avg'")
    
    PoolingLayer = layers.MaxPooling2D if pooling_type == 'max' else layers.AveragePooling2D
    
    model_input = keras.Input(shape=input_shape)
    
    # --- Parte CNN ---
    # Blocco 1
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Blocco 2
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Blocco 3 (più profondo, come da letteratura)
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # --- Preparazione per GRU ---
    # Reshape per passare le feature estratte dalla CNN alla GRU
    current_shape = x.shape
    # (batch, freq, time, channels) -> (batch, time, freq * channels)
    x = layers.Reshape((current_shape[2], current_shape[1] * current_shape[3]))(x)
    
    # --- Parte GRU ---
    # Bidirezionale per catturare il contesto temporale in entrambe le direzioni
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)  # Dropout più aggressivo prima del classificatore
    
    # --- Parte Dense (Classificatore) ---
    model_output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=model_input, outputs=model_output)
    
    # Ottimizzatore Adam con learning rate più conservativo, come da paper
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Il modello 1D rimane invariato per ora, è già una buona architettura di partenza.
def create_1d_cnn_gru_model(input_shape, num_classes):
    """Crea un modello ibrido 1D-CNN + GRU."""
    model_input = keras.Input(shape=input_shape)
    
    # Parte CNN 1D
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Parte GRU
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    
    # Parte Dense
    model_output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=model_input, outputs=model_output)
    
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model