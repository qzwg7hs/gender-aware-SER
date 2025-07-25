import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("C:/Users/Aruay/Desktop/ra application/project/dataset/combined_speech_data.csv")

# Assuming the dataset has a "gender" column and feature columns
X = df.drop(columns=["gender"]).values  # Drop label column and convert to numpy
y = df["gender"].values  # Convert labels to numpy

# Shuffle the dataset
shuffle_indices = np.random.permutation(len(y))
X, y = X[shuffle_indices], y[shuffle_indices]

# Encode gender labels (0: female, 1: male)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
])

model.summary()

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save the model
model.save("C:/Users/Aruay/Desktop/ra application/project/gender_classification_model.h5")
print("Model saved as 'gender_classification_model.h5'")

# Save the scaler for future data preprocessing
import joblib
joblib.dump(scaler, "C:/Users/Aruay/Desktop/ra application/project/scaler.pkl")
print("Scaler saved as 'scaler.pkl'")


from tensorflow.keras.utils import plot_model

# Assuming your model is named `model`
plot_model(model, to_file='C:/Users/Aruay/Desktop/ra application/project/gender_model_architecture.png', show_shapes=True, show_layer_names=True)
