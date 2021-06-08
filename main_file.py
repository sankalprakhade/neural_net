import mnist_loader
import number_recognisation

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = number_recognisation.Network([784, 100, 10])
net.grad_d(training_data, 30, 10, 3.0, test_data=test_data)