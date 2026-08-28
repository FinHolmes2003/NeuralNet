
# Neural Network

Basic neural network written in C++

Allows for creation of n-dimentional neural networks that have build-in back propagation

> ### `Example.cpp`
> Shows the neural network computing an XOR gate and using back propagation to set the weights. The weights are saved in `Weights/weights.json`

***

> ### `MNIST_Example.cpp`
> Is used to display and estimate values in the MNIST database.
> [The Files were taken from this website](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)
> The GUI is handled using [wxWidgets](https://wxwidgets.org/)
>
> ![](Images/MNIST_README_IMG.png)

***

> ### `include/Network.hpp`
> Is the actual network logic file. It contains all the logic like forward pass and backpropagation, along with the ability to save the weights and biases as a json file using the [nlohmann/json repo](https://github.com/nlohmann/json).

***

> ### Task list for next features
> - [ ] Allow parallel computation using [openMp](https://computing.stat.berkeley.edu/tutorial-parallelization/parallel-C.html)
> - [ ] Vectorise the operations to prevent waste whilst using `std::vector<>`
> - [ ] Add GPU accelleration using [CUDA](https://developer.nvidia.com/cuda/toolkit)