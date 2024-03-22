import tensorflow as tf
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes = 10, weights = None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

print("Loading data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Training model...")
model.fit(x_train, y_train, epochs = 1, batch_size = 32)
loss, accuracy = model.evaluate(x_test, y_test)

print("Loss: ",loss," Accuracy; ", accuracy)