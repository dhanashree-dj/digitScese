import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import os

MODEL_FILE = "digit_scese_cnn.h5"

# =========================
# Load Dataset
# =========================
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# =========================
# Load or Train Model
# =========================
if os.path.exists(MODEL_FILE):
    print("✅ Loading saved model...")
    model = keras.models.load_model(MODEL_FILE)

else:
    print("🚀 Training new model (fast mode)...")

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ⚡ Fast training subset
    model.fit(x_train[:15000], y_train[:15000], epochs=2)

    model.save(MODEL_FILE)
    print("💾 Model saved")

# =========================
# Evaluate
# =========================
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# =========================
# MNIST Sample Prediction
# =========================
prediction = model.predict(x_test[:1])
print("Predicted Digit:", prediction.argmax())
print("Confidence:", prediction.max())

plt.imshow(x_test[0].reshape(28,28), cmap='gray')
plt.title("MNIST Sample")
plt.show()

# =========================
# Custom Image Prediction
# =========================
def predict_custom_image(path):

    if not os.path.exists(path):
        print("❌ Image not found:", path)
        return

    img = Image.open(path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((40, 40))

    img = np.array(img)
    img = np.where(img > 50, 255, 0)

    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        print("No digit detected")
        return

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    img = img[y0:y1, x0:x1]

    img = Image.fromarray(img).resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img)

    print("Custom Image Prediction:", pred.argmax())
    print("Confidence:", pred.max())

    plt.imshow(img[0].reshape(28,28), cmap="gray")
    plt.title("Processed Custom Image")
    plt.show()


# =========================
# Run Custom Test
# =========================
predict_custom_image("mydigit1.png")
