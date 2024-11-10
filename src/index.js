"use strict";
class NeuralNetwork {
    _inputNodes;
    _hiddenNodes;
    _outputNodes;
    weightsInputHidden;
    weightsHiddenOutput;
    biasHidden;
    biasOutput;
    _learningRate = 0.01;
    constructor(_inputNodes, _hiddenNodes, _outputNodes, learningRate) {
        this._inputNodes = _inputNodes;
        this._hiddenNodes = _hiddenNodes;
        this._outputNodes = _outputNodes;
        this.weightsInputHidden = new Matrix(this._hiddenNodes, this._inputNodes);
        this.weightsHiddenOutput = new Matrix(this._outputNodes, this._hiddenNodes);
        if (learningRate) {
            this._learningRate = learningRate;
        }
        this.weightsInputHidden.randomize();
        this.weightsHiddenOutput.randomize();
        this.biasHidden = new Matrix(this._hiddenNodes, 1);
        this.biasOutput = new Matrix(this._outputNodes, 1);
        this.biasHidden.randomize();
        this.biasOutput.randomize();
    }
    get inputNodes() {
        return this._inputNodes;
    }
    get hiddenNodes() {
        return this._hiddenNodes;
    }
    get outputNodes() {
        return this._outputNodes;
    }
    get learningRate() {
        return this._learningRate;
    }
    /**
     * Predicts the output based on the provided input array by predicting it in the neural network.
     * @param {number[]} inputArray - An array of numbers representing the input(recommended between 0-1)
     * @returns {number[]} An array of numbers representing the predicted output (recommended between 0-1)
     */
    computeOutput(inputArray) {
        const inputs = Matrix.fromArray(inputArray);
        const hidden = Matrix.multiply(this.weightsInputHidden, inputs);
        hidden.add(this.biasHidden);
        hidden.map(SigmoidFunction.sigmoid);
        const output = Matrix.multiply(this.weightsHiddenOutput, hidden);
        output.add(this.biasOutput);
        output.map(SigmoidFunction.sigmoid);
        return output.toArray();
    }
    /**
     * Trains the neural network by the process of backpropagation.
     * It updates the network's weights and biases based on the provided input and target arrays.
     * @param {number[]} inputArray - An array of numbers representing the input.
     * @param {number[]} targetsArray - An array of numbers representing the target output.
     */
    train(inputArray, targetsArray) {
        const inputs = Matrix.fromArray(inputArray);
        const hidden = Matrix.multiply(this.weightsInputHidden, inputs);
        hidden.add(this.biasHidden);
        hidden.map(SigmoidFunction.sigmoid);
        const outputs = Matrix.multiply(this.weightsHiddenOutput, hidden);
        outputs.add(this.biasOutput);
        outputs.map(SigmoidFunction.sigmoid);
        const targets = Matrix.fromArray(targetsArray);
        const outputErrors = Matrix.subtract(targets, outputs);
        let gradients = Matrix.map(outputs, SigmoidFunction.derivative);
        gradients = Matrix.multiplyElementWise(gradients, outputErrors);
        gradients.multiply(this._learningRate);
        const hiddenTransposed = Matrix.transpose(hidden);
        const weightsHiddenOutputDeltas = Matrix.multiply(gradients, hiddenTransposed);
        this.weightsHiddenOutput.add(weightsHiddenOutputDeltas);
        this.biasOutput.add(gradients);
        const weightsHiddenOutputTransposed = Matrix.transpose(this.weightsHiddenOutput);
        const hidden_errors = Matrix.multiply(weightsHiddenOutputTransposed, outputErrors);
        let hiddenGradient = Matrix.map(hidden, SigmoidFunction.derivative);
        hiddenGradient = Matrix.multiplyElementWise(hiddenGradient, hidden_errors);
        hiddenGradient.multiply(this._learningRate);
        const inputsTransposed = Matrix.transpose(inputs);
        const weightsInputHiddenDeltas = Matrix.multiply(hiddenGradient, inputsTransposed);
        this.weightsInputHidden.add(weightsInputHiddenDeltas);
        this.biasHidden.add(hiddenGradient);
    }
}
class Matrix {
    _rows;
    _cols;
    data;
    constructor(_rows, _cols) {
        this._rows = _rows;
        this._cols = _cols;
        this.data = Array.from({ length: _rows }, () => new Array(_cols).fill(0));
    }
    get rows() {
        return this._rows;
    }
    get cols() {
        return this._cols;
    }
    /**
     * Multiplies two matrices together using the dot product and returns the resulting matrix.
     *
     * This method performs matrix multiplication, where the number of columns in the first
     * matrix (a) must equal the number of rows in the second matrix (b). The resulting matrix
     * will have dimensions equal to the number of rows of the first matrix (a) and the number
     * of columns of the second matrix (b).
     * @param {Matrix} a - The first matrix to be multiplied.
     * @param {Matrix} b - The second matrix to be multiplied.
     * @returns {Matrix} The resulting matrix from the multiplication.
     */
    static multiply(a, b) {
        if (a._cols !== b._rows) {
            throw new Error(`Matrix multiplication cannot occur. Columns of A must match rows of B.`);
        }
        const result = new Matrix(a._rows, b._cols);
        for (let i = 0; i < result._rows; i++) {
            for (let j = 0; j < result._cols; j++) {
                let sum = 0;
                for (let k = 0; k < a._cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }
    /**
     * Multiplies each element of the matrix by a scalar number.
     * @param {number} n - The scalar number to multiply each element of the matrix by.
     */
    multiply(n) {
        for (let i = 0; i < this._rows; i++) {
            for (let j = 0; j < this._cols; j++) {
                this.data[i][j] *= n;
            }
        }
    }
    /**
     * Performs the hadamard product (element-wise multiplication) of two matrices.
     * Inspired by ChatGPT
     * @param {Matrix} a - The first matrix multipled that must have the same dimensions as matrix b.
     * @param {Matrix} b - The second matrix multipled that must have the same dimensions as matrix a.
     * @returns {Matrix} A new matrix containing the result of the Hadamard product.
     */
    static multiplyElementWise(a, b) {
        if (a._rows !== b._rows || a._cols !== b._cols) {
            throw new Error(`Matrix dimensions must match for element-wise multiplication.`);
        }
        const result = new Matrix(a._rows, a._cols);
        for (let i = 0; i < a._rows; i++) {
            for (let j = 0; j < a._cols; j++) {
                result.data[i][j] = a.data[i][j] * b.data[i][j];
            }
        }
        return result;
    }
    static transpose(n) {
        const result = new Matrix(n._cols, n._rows);
        for (let i = 0; i < n._rows; i++) {
            for (let j = 0; j < n._cols; j++) {
                result.data[j][i] = n.data[i][j];
            }
        }
        return result;
    }
    /**
     * Applies a given function to each element of a given matrix.
     * @param {Matrix} matrix - The matrix for the function to be applied on.
     * @param {Function} func - The function to apply to each element of the matrix that
     *                          takes a single argument (the current element value) and
     *                          returns the new value.
     */
    static map(matrix, func) {
        const result = new Matrix(matrix._rows, matrix._cols);
        for (let i = 0; i < matrix._rows; i++) {
            for (let j = 0; j < matrix._cols; j++) {
                const val = matrix.data[i][j];
                result.data[i][j] = func(val);
            }
        }
        return result;
    }
    /**
     * Applies a given function to each element of the matrix.
     * @param {Function} func - The function to apply to each element of the matrix that
     *                          takes a single argument (the current element value) and
     *                          returns the new value.
     */
    map(func) {
        for (let i = 0; i < this._rows; i++) {
            for (let j = 0; j < this._cols; j++) {
                const val = this.data[i][j];
                this.data[i][j] = func(val);
            }
        }
    }
    add(n) {
        if (n instanceof Matrix) {
            for (let i = 0; i < this._rows; i++) {
                for (let j = 0; j < this._cols; j++) {
                    this.data[i][j] += n.data[i][j];
                }
            }
        }
        else {
            for (let i = 0; i < this._rows; i++) {
                for (let j = 0; j < this._cols; j++) {
                    this.data[i][j] += n;
                }
            }
        }
    }
    randomize() {
        for (let i = 0; i < this._rows; i++) {
            for (let j = 0; j < this._cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }
    toArray() {
        const arr = [];
        for (let i = 0; i < this._rows; i++) {
            for (let j = 0; j < this._cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        return arr;
    }
    /**
     * Creates a column matrix from a given array of numbers.
     * @param {Array<number>} arr - The input array of numbers.
     * @returns {Matrix} A new matrix where each element of the input array is placed in its own row.
     */
    static fromArray(arr) {
        const m = new Matrix(arr.length, 1);
        for (let i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }
    static subtract(a, b) {
        const result = new Matrix(a._rows, a._cols);
        for (let i = 0; i < result._rows; i++) {
            for (let j = 0; j < result._cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }
}
class SigmoidFunction {
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    /**
     * Calculates the derivative of the sigmoid function for a given output value.
     * This method assumes that the sigmoid function has already been applied.
     * @param {number} y - The output of the sigmoid function, a number between 0 and 1.
     * @returns The derivative of the sigmoid function at the given output value.
     */
    static derivative(y) {
        return y * (1 - y);
    }
}
module.exports = NeuralNetwork;