import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Dataset paths
train_path = "dataset/Training"
test_path = "dataset/Testing"

# Image settings
IMG_SIZE = 128
BATCH_SIZE = 32

# Training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Testing data
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Show classes
print("Class names:")
print(train_data.class_indices)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu",
           input_shape=(IMG_SIZE, IMG_SIZE, 3)),

    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),

    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),

    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.5),

    Dense(4, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Show model
model.summary()

# Train
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

# Test
test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# Save model
model.save("brain_tumor_cnn.keras")

print("Model saved successfully!")

# Accuracy graph
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

# Loss graph
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()