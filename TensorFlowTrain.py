import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = '/Users/jkrek/Downloads/disease_occurrences_optimized.csv'  # Adjust the path
df = pd.read_csv(data_path)

# Use 'Disease' as the feature for now, as 'Symptoms' column is not available
diseases = df['Disease'].values
occurrences = df['Occurrences'].values  # This could be used as a feature or target, depending on your task

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(diseases)

# Split the data into training and validation sets
diseases_train, diseases_val, labels_train, labels_val = train_test_split(diseases, encoded_labels, test_size=0.2, random_state=42)

# Define the TextVectorization layer
max_tokens = 10000
sequence_length = 100
vectorize_layer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Fit the TextVectorization layer to the training data
vectorize_layer.adapt(diseases_train)

# Build the model
model = Sequential([
    vectorize_layer,
    Embedding(max_tokens, 128, input_length=sequence_length),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(diseases_train, labels_train, epochs=100, validation_data=(diseases_val, labels_val))

model.save('/Users/jkrek/Downloads/trainset', save_format = 'tf')
