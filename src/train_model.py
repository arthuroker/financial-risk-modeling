import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import class_weight

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),  # No .shape here anymore
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model
def train_model(X, y):
    
    print("🔁 Starting training pipeline...")

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Data scaled.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print("✅ Data split: train:", len(X_train), "test:", len(X_test))

    # Model setup
    model = build_model(X.shape[1])
    model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(name='auc')
    ]
)
    print("✅ Model compiled.")

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    checkpoint_path = "best_model.keras"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy', save_best_only=True
    )

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        class_weight=class_weights_dict
    )
    print("✅ Training complete.")

    # Evaluation
    print("🔍 Evaluating model...")
    loss, accuracy, precision, recall, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"🧪 Test Loss: {loss:.4f}")
    print(f"🧪 Test Accuracy: {accuracy:.4f}")
    print(f"🧪 Test Precision: {precision:.4f}")
    print(f"🧪 Test Recall: {recall:.4f}")
    print(f"🧪 Test AUC: {auc:.4f}")

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("🧱 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.3).astype(int)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Visualization
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Training Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()