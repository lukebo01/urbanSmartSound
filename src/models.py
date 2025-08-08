import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# ==============================================================================
#           NUOVA SEZIONE: DEFINIZIONE DEL LAYER DI AUGMENTATION
# ==============================================================================

class SpecAugmentLayer(layers.Layer):
    """
    Layer personalizzato per applicare SpecAugment (Time & Frequency/Coefficient Masking).
    Questo layer è attivo solo durante l'addestramento, come da best practice.
    """
    def __init__(self, time_mask_param, freq_mask_param, num_time_masks, num_freq_masks, **kwargs):
        super(SpecAugmentLayer, self).__init__(**kwargs)
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def call(self, inputs, training=None):
        if not training:
            return inputs # Non applica augmentation durante la validazione o il test

        augmented = inputs
        # Applica N maschere di frequenza/coefficiente
        for _ in range(self.num_freq_masks):
            augmented = self.frequency_mask(augmented)
        # Applica N maschere di tempo
        for _ in range(self.num_time_masks):
            augmented = self.time_mask(augmented)
        return augmented

    def frequency_mask(self, spec):
        # L'input ha shape (batch, num_coeffs, timesteps, channels) per Conv2D
        # La shape che Keras vede è (batch, H, W, C) dove H=num_coeffs, W=timesteps
        num_coeffs = tf.shape(spec)[1]
        f = tf.random.uniform(shape=(), minval=0, maxval=self.freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform(shape=(), minval=0, maxval=num_coeffs - f, dtype=tf.int32)

        # Crea una maschera che azzera una banda di coefficienti
        mask_start = tf.ones_like(spec[:, :f0, :, :])
        mask_zeros = tf.zeros_like(spec[:, f0:f0+f, :, :])
        mask_end = tf.ones_like(spec[:, f0+f:, :, :])

        return tf.concat([mask_start, mask_zeros, mask_end], axis=1) * spec

    def time_mask(self, spec):
        # L'input ha shape (batch, num_coeffs, timesteps, channels)
        num_timesteps = tf.shape(spec)[2]
        t = tf.random.uniform(shape=(), minval=0, maxval=self.time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform(shape=(), minval=0, maxval=num_timesteps - t, dtype=tf.int32)

        # Crea una maschera che azzera una banda di timesteps
        mask_start = tf.ones_like(spec[:, :, :t0, :])
        mask_zeros = tf.zeros_like(spec[:, :, t0:t0+t, :])
        mask_end = tf.ones_like(spec[:, :, t0+t:, :])

        return tf.concat([mask_start, mask_zeros, mask_end], axis=2) * spec

    def get_config(self):
        # Necessario per salvare e caricare il modello
        config = super().get_config()
        config.update({
            "time_mask_param": self.time_mask_param,
            "freq_mask_param": self.freq_mask_param,
            "num_time_masks": self.num_time_masks,
            "num_freq_masks": self.num_freq_masks,
        })
        return config

# ==============================================================================
# === IMPLEMENTAZIONE MANUALE DELLA FOCAL LOSS (INVARIATA) ===
# ==============================================================================
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

def binary_focal_loss(alpha=0.25, gamma=2.0):
    """Implementazione della Binary Focal Loss per Keras."""
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return K.mean(loss, axis=-1)
    return focal_loss
# ==============================================================================
# === MODELLO OTTIMIZZATO (2 GRU) CON SPEC-AUGMENT INTEGRATO ===
# ==============================================================================
def create_binary_classifier_model(input_shape, pooling_type='max'):
    """
    Crea l'architettura ottimizzata per un compito di classificazione BINARIA.
    Output: 1 neurone con attivazione sigmoid. Loss: Binary Focal Loss.
    """
    if pooling_type not in ['max', 'avg']:
        raise ValueError("pooling_type deve essere 'max' o 'avg'")
    
    PoolingLayer = layers.MaxPooling2D if pooling_type == 'max' else layers.AveragePooling2D
    model_input = keras.Input(shape=input_shape)
    x = SpecAugmentLayer(time_mask_param=25, freq_mask_param=12, num_time_masks=2, num_freq_masks=2)(model_input)
    
    # Parte CNN
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = PoolingLayer((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Parte Ricorrente
    current_shape = x.shape
    x = layers.Reshape((current_shape[2], current_shape[1] * current_shape[3]))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x) 
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    
    # Classificatore Binario
    model_output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=model_input, outputs=model_output)
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    loss_function = binary_focal_loss(gamma=1.5)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    
    return model
    
def create_advanced_2d_cnn_3gru_model(input_shape, num_classes, pooling_type='max'):
    """
    Crea un modello ibrido ottimizzato con 2D-CNN + 2 GRU impilati.
    *** NUOVO: Integra un layer di SpecAugment per la data augmentation on-the-fly. ***
    """
    if pooling_type not in ['max', 'avg']:
        raise ValueError("pooling_type deve essere 'max' o 'avg'")
    
    PoolingLayer = layers.MaxPooling2D if pooling_type == 'max' else layers.AveragePooling2D
    
    model_input = keras.Input(shape=input_shape)
    
    # --- MODIFICA CHIAVE: INSERIMENTO DEL LAYER DI AUGMENTATION ---
    # Parametri standard per SpecAugment, possono essere affinati.
    x = SpecAugmentLayer(
        time_mask_param=25,     # Maschera fino a 25 timesteps
        freq_mask_param=12,     # Maschera fino a 12 coefficienti/frequenze
        num_time_masks=2,       # Applica 2 maschere temporali
        num_freq_masks=2        # Applica 2 maschere di frequenza
    )(model_input)
    # --- FINE MODIFICA ---
    
    # --- Parte CNN (ora applicata all'output aumentato 'x') ---
    x = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(x) # Era model_input
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
    # La logica del reshape è errata nel tuo codice originale per un input (H, W, C).
    # Corretta per essere (batch, timesteps, features)
    current_shape = x.shape
    # (batch, H, W, C) -> (batch, W, H * C)
    # W diventa l'asse temporale, H*C diventa l'asse delle feature.
    x = layers.Reshape((current_shape[2], current_shape[1] * current_shape[3]))(x)
    
    # --- Parte GRU Gerarchica (2 Layer con Dropout aumentato) ---
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x) 

    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    
    # --- Parte Dense (Classificatore) ---
    model_output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=model_input, outputs=model_output)
    
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    loss_function = categorical_focal_loss(alpha=0.25, gamma=1.5)
        
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    
    return model

# --- MODELLO PER LA PIPELINE C AVANZATA (INVARIATO) ---
def create_advanced_1d_cnn_2gru_model(input_shape, num_classes):
    """
    Crea un modello ibrido 1D-CNN + 2 GRU impilati (per Pipeline C Advanced).
    """
    # (Codice invariato, lo lascio per completezza)
    model_input = keras.Input(shape=input_shape)
    
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    
    model_output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=model_input, outputs=model_output)
    
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    loss_function = categorical_focal_loss(alpha=0.25, gamma=1.5)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    
    return model