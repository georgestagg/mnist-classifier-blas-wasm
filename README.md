# MNIST Classifier - A WebAssembly BLAS Demo

A handwritten digit classifier demonstrating BLAS routines running in a web browser using WebAssembly.

A [multilayer perceptron network](https://en.wikipedia.org/wiki/Multilayer_perceptron) is used for the modeling and classification of digits. The pre-training of model weights has been performed ahead of time using [Scikit-learn](http://Scikit-learn.org), and the classification of digits runs interactively using JavaScript and WebAssembly.

The [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) provides the source data used to train the model. Classifier output is shown using [Observable Plot](https://observablehq.com/plot/). 

# Interactive Website

Draw a digit from 0-9 in the box and the classifier will try to label the handwritten digit. The resulting relative probabilities will be shown in a plot on the right.

https://georgestagg.github.io/mnist-classifier-blas-wasm/
