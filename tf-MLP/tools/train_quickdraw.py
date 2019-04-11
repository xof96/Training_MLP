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
    i = 0
    model_dir = 'C:/Users/XOF/PycharmProjects/Training_MLP/tf-MLP/models/optimizer/quickdraw/gd/alfa0001{}'.format(functions[i])
    params = {
        "device": "/gpu:0",
        "model_dir": model_dir,
        "data_dir": "C:/Users/XOF/PycharmProjects/Training_MLP/QuickDraw-Animals",
        "activation_function": i,
        "learning_rate": 0.001,
        "number_of_classes": 12,
        "number_of_iterations": 40000,
        "batch_size": 80,
        "data_size": 12000,
    }
    my_mlp = mlp.MLP(params)
    print("MLP initialized ok")
    print("--------start training")
    my_mlp.train()
    print("--------end training")
    print("-----------------{}-------------------".format(functions[i]))
