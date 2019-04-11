"""
Author: jose.saavedra
This is an example for training on a sketch classification problem
model_dir and data_dir should changed to the correct paths
"""
import sys
import os

# Use my_mlp.train() for training
# Use my_mlp.test() for testing
# Use my_mlp.save_model() for saving models that will be used for fast_prediction

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mlp.mlp as mlp

functions = ["sigmoid", "tanh", "relu", "leaky_relu"]

if __name__ == "__main__":
    params = {
        "device": "/gpu:0",
        "data_dir": "C:/Users/XOF/PycharmProjects/Training_MLP/MNIST-5000",
        "learning_rate": 0.001,
        "number_of_classes": 10,
        "number_of_iterations": 5000,
        "batch_size": 80,
        "data_size": 5000,
    }

    i = 2
    print("-----------------{}-------------------".format(functions[i]))
    params["model_dir"] = "C:/Users/XOF/PycharmProjects/Training_MLP/tf-MLP/models/optimizer/mnist/gd/alfa0001/{}".format(functions[i])
    params["activation_function"] = i
    my_mlp = mlp.MLP(params)
    print("MLP initialized ok")
    print("--------start training")
    my_mlp.train()
    print("--------end training")
