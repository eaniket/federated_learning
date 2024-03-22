import tensorflow as tf
import ssl
import flwr as fl

ssl._create_default_https_context = ssl._create_unverified_context

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes = 10, weights = None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# 0: Airplane, 1: Automobile, 2: Bird, 3: Cat, 4: Deer, 5: Dog, 6: Frog, 7: Horse, 8: Ship, and 9: Truck.

print("Loading data on client 1...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#dividing data
x_train = x_train[:25000]
y_train = y_train[:25000]

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        print("Training model on client 1...")
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("Data of client 1 -> Loss: ",loss," Accuracy; ", accuracy)
        return loss, len(x_test), {"accuracy": float(accuracy)}

fl.client.start_client(server_address="[::]:8080", client=CifarClient().to_client())