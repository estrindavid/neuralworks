# NeuralWorks

  

**NeuralWorks** is a lightweight, modular JavaScript library designed to create and train neural networks 
directly in the browser or on Node.js environments. With a focus on simplicity and flexibility, 
NeuralWorks empowers developers to build neural networks without the need for complex dependencies.

  

## Features

-  **Easy to Use**: Simple API for building, training, and predicting with neural networks.

-  **Modular**: Easily extendable for custom neural network designs.

-  **Client-Side Support**: Works directly in the browser using vanilla JavaScript, as well as on Node.js.

-  **Sigmoid Activation**: Supports the sigmoid activation function for smooth output scaling.

-  **Backpropagation**: Trains networks using the backpropagation algorithm.

  

## Installation

  

To install NeuralWorks in your project, run the following command:

  

```bash

npm  install  neuralworks

```

## Project Structure

The project is organized as follows:

- **`src/`**: Contains the source code for the library.
  - **`index.js`**: Main file containing the core functionality of the library.

- **`LICENSE`**: The open-source license for the project.

- **`README.md`**: The documentation file you are currently reading, providing details about the library and usage.

- **`package.json`**: Defines project metadata, dependencies, and scripts.


## Usage

  

Once installed, you can import and start using the library in your JavaScript or TypeScript code.

  

### Example Usage

  

```javascript

const  NeuralNetwork = require('neuralworks');

  

// Create a neural network instance

const  nn = new  NeuralNetwork(2, 4, 1); // 2 input nodes, 4 hidden nodes, 1 output node

  

// Training data (XOR Problem)

const  inputs = [

[0, 0],

[1, 0],

[0, 1],

[1, 1]

];

  

const  targets = [

[0], // Expected output for [0, 0]

[1], // Expected output for [1, 0]

[1], // Expected output for [0, 1]

[0] // Expected output for [1, 1]

];

  

// Training the network

for (let  i = 0; i < 10000; i++) {

const  index = Math.floor(Math.random() * inputs.length);

nn.train(inputs[index], targets[index]);

}

  

// Testing the network

console.log(nn.computeOutput([1, 0])); // Expected output: ~1 (close to 1 for XOR problem)

console.log(nn.computeOutput([0, 0])); // Expected output: ~0

```

  

### API

  

#### `NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate?)`

  

Constructor for creating a neural network.

  

-  `inputNodes` (number): Number of input nodes.

-  `hiddenNodes` (number): Number of hidden nodes.

-  `outputNodes` (number): Number of output nodes.

-  `learningRate` (number, optional): Learning rate (default is 0.01).

  

#### `computeOutput(inputArray)`

  

-  **inputArray** (array): Array of input values (e.g., `[0, 1]`).

-  **Returns**: An array of output values, typically between 0 and 1.

  

#### `train(inputArray, targetsArray)`

  

-  **inputArray** (array): Array of input values.

-  **targetsArray** (array): Array of expected output values.

-  **Purpose**: Trains the network using backpropagation.

  

### License


This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.


  

## Contributing

  

1. Fork the repository.

2. Create your feature branch (`git checkout -b feature/your-feature`).

3. Commit your changes (`git commit -am 'Add new feature'`).

4. Push to the branch (`git push origin feature/your-feature`).

5. Create a new Pull Request.

  

---

  

Made by [David Estrin](https://github.com/estrindavid).