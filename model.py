"""
Classification Model with Fully Connected Layers
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import config
import os
import numpy as np


def create_classification_model(input_dim, num_classes=config.NUM_CLASSES):
    """
    Create fully connected neural network for classification
    
    Args:
        input_dim: Dimension of input features (after PCA)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # First dense block
        layers.Dense(config.DENSE_UNITS[0], activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),
        
        # Second dense block
        layers.Dense(config.DENSE_UNITS[1], activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),
        
        # Third dense block
        layers.Dense(config.DENSE_UNITS[2], activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE * 0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


class AlzheimerClassifier:
    """
    Complete classifier for Alzheimer's disease staging
    """
    def __init__(self, input_dim, learning_rate=0.001, num_classes=config.NUM_CLASSES):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        self.build_model()
    
    def build_model(self):
        """
        Build and compile the model
        """
        self.model = create_classification_model(self.input_dim, self.num_classes)
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model built with learning rate: {self.learning_rate}")
        print(f"Input dimension: {self.input_dim}")
        print(f"Number of classes: {self.num_classes}")
    
    def get_callbacks(self, model_name='model'):
        """
        Get training callbacks
        """
        checkpoint_path = os.path.join(config.MODEL_DIR, f'{model_name}_best.h5')
        
        callback_list = [
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callback_list
    
    def train(self, X_train, y_train, X_val, y_val, epochs=config.EPOCHS, 
              batch_size=config.BATCH_SIZE, model_name='model'):
        """
        Train the model
        """
        print(f"\nTraining model with learning rate: {self.learning_rate}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(model_name),
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        
        print(f"\nTest Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions
        """
        return self.model.predict(X, verbose=0)
    
    def predict_classes(self, X):
        """
        Predict class labels
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def save(self, filepath=None):
        """
        Save the model
        """
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, f'classifier_lr{self.learning_rate}.h5')
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def summary(self):
        """
        Print model summary
        """
        return self.model.summary()


if __name__ == "__main__":
    # Test model creation
    print("Creating test model...")
    test_model = AlzheimerClassifier(input_dim=100, learning_rate=0.001)
    test_model.summary()
    print("\nModel created successfully!")
