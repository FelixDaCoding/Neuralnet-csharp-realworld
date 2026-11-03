using System;
using System.Collections.Generic;

namespace backneural
{
    /// <summary>
    /// Custom 3-layer Multi-Layer Perceptron (Input, Hidden, Output layers) with Backpropagation.
    /// Coordinates INeuron, HNeuron, and ONeuron layers for forward propagation and gradient descent weight updates.
    /// </summary>
    public class NeuralNet
    {
        private INeuron[] ilayer;
        private HNeuron[] hlayer;
        private ONeuron[] olayer;
        private double learningRate;

        /// <summary>
        /// Gets the configured learning rate for weight updates.
        /// </summary>
        public double LearningRate => this.learningRate;

        /// <summary>
        /// Gets the number of input layer neurons.
        /// </summary>
        public int InputCount => this.ilayer.Length;

        /// <summary>
        /// Gets the number of hidden layer neurons.
        /// </summary>
        public int HiddenCount => this.hlayer.Length;

        /// <summary>
        /// Gets the number of output layer neurons.
        /// </summary>
        public int OutputCount => this.olayer.Length;

        /// <summary>
        /// Initializes a new 3-layer MLP with default 8 inputs, 8 hidden neurons, and 1 output neuron.
        /// </summary>
        public NeuralNet()
            : this(8, 8, 1, 0.1, 42)
        {
        }

        /// <summary>
        /// Initializes a new 3-layer MLP with specified layer sizes, learning rate, and random seed.
        /// </summary>
        /// <param name="iSize">Number of input neurons (features).</param>
        /// <param name="hSize">Number of hidden layer neurons.</param>
        /// <param name="oSize">Number of output layer neurons.</param>
        /// <param name="learningRate">Learning rate (eta) for backpropagation updates.</param>
        /// <param name="seed">Fixed random seed for deterministic weight initialization.</param>
        public NeuralNet(int iSize, int hSize, int oSize, double learningRate = 0.1, int seed = 42)
        {
            this.learningRate = learningRate;
            this.ilayer = new INeuron[iSize];
            this.hlayer = new HNeuron[hSize];
            this.olayer = new ONeuron[oSize];

            Random rand = new Random(seed);

            for (int x = 0; x < iSize; x++)
            {
                this.ilayer[x] = new INeuron(x, hSize, rand);
            }

            for (int y = 0; y < hSize; y++)
            {
                this.hlayer[y] = new HNeuron(y, oSize, rand);
            }

            for (int z = 0; z < oSize; z++)
            {
                this.olayer[z] = new ONeuron(z, rand);
            }
        }

        /// <summary>
        /// Sets the learning rate for gradient descent updates.
        /// </summary>
        /// <param name="rate">New learning rate.</param>
        public void setLearningRate(double rate)
        {
            this.learningRate = rate;
        }

        /// <summary>
        /// Sets the input activation for a single input neuron at the specified position.
        /// </summary>
        /// <param name="pos">Input neuron index.</param>
        /// <param name="data">Feature value.</param>
        public void setInputs(int pos, double data)
        {
            this.ilayer[pos].setAct(data);
        }

        /// <summary>
        /// Sets all input neuron activations from a feature vector.
        /// </summary>
        /// <param name="inputs">Array of feature values matching input layer size.</param>
        public void setInputs(double[] inputs)
        {
            for (int i = 0; i < this.ilayer.Length && i < inputs.Length; i++)
            {
                this.ilayer[i].setAct(inputs[i]);
            }
        }

        /// <summary>
        /// Sets the desired target output for a specific output neuron.
        /// </summary>
        /// <param name="pos">Output neuron index.</param>
        /// <param name="data">Target value (0.0 or 1.0).</param>
        public void setDesiredOutput(int pos, double data)
        {
            this.olayer[pos].setDesired(data);
        }

        /// <summary>
        /// Gets the predicted activation from a specific output neuron.
        /// </summary>
        /// <param name="pos">Output neuron index.</param>
        /// <returns>Sigmoid output activation in (0, 1).</returns>
        public double getOutput(int pos)
        {
            return this.olayer[pos].getAct();
        }

        /// <summary>
        /// Executes the forward pass across the network:
        /// 1. Each hidden neuron computes its weighted sum from the input layer and applies sigmoid activation.
        /// 2. Each output neuron computes its weighted sum from the hidden layer and applies sigmoid activation.
        /// </summary>
        public void run()
        {
            for (int x = 0; x < this.hlayer.Length; x++)
            {
                this.hlayer[x].calculateActivation(this.ilayer);
            }

            for (int y = 0; y < this.olayer.Length; y++)
            {
                this.olayer[y].calculateActivation(this.hlayer);
            }
        }

        /// <summary>
        /// Executes one full forward and backpropagation step for the currently loaded training sample:
        /// 1. Forward pass to compute current activations.
        /// 2. Compute output layer error deltas.
        /// 3. Compute hidden layer error deltas BEFORE mutating hidden-to-output weights.
        /// 4. Update hidden-to-output weights and output biases.
        /// 5. Update input-to-hidden weights and hidden biases.
        /// </summary>
        public void learn()
        {
            // Step 1: Forward propagation
            this.run();

            // Step 2: Calculate error delta at the output layer
            for (int x = 0; x < this.olayer.Length; x++)
            {
                this.olayer[x].calculateError();
            }

            // Step 3: Backpropagate error delta to the hidden layer using current weights
            for (int y = 0; y < this.hlayer.Length; y++)
            {
                this.hlayer[y].calculateError(this.olayer);
            }

            // Step 4: Update hidden-to-output weights and output biases
            for (int x = 0; x < this.olayer.Length; x++)
            {
                this.olayer[x].newWeight(this.hlayer, this.learningRate);
            }

            // Step 5: Update input-to-hidden weights and hidden biases
            for (int y = 0; y < this.hlayer.Length; y++)
            {
                this.hlayer[y].newWeight(this.ilayer, this.learningRate);
            }
        }

        /// <summary>
        /// Runs a forward pass on a feature vector and returns the sigmoid output probability of neuron 0.
        /// </summary>
        /// <param name="features">Normalized input feature vector.</param>
        /// <returns>Predicted probability in (0, 1).</returns>
        public double PredictProbability(double[] features)
        {
            this.setInputs(features);
            this.run();
            return this.getOutput(0);
        }

        /// <summary>
        /// Computes the average Binary Cross-Entropy (BCE) loss over a dataset:
        /// L = - (1/N) * sum( y * ln(y_hat) + (1 - y) * ln(1 - y_hat) )
        /// </summary>
        /// <param name="features">List of feature vectors.</param>
        /// <param name="labels">List of binary labels (0 or 1).</param>
        /// <returns>Mean Binary Cross-Entropy loss.</returns>
        public double ComputeBinaryCrossEntropyLoss(IReadOnlyList<double[]> features, IReadOnlyList<int> labels)
        {
            if (features.Count == 0)
            {
                return 0.0;
            }

            const double epsilon = 1e-15;
            double totalLoss = 0.0;

            for (int i = 0; i < features.Count; i++)
            {
                double yHat = this.PredictProbability(features[i]);
                if (yHat < epsilon) yHat = epsilon;
                if (yHat > 1.0 - epsilon) yHat = 1.0 - epsilon;

                double y = labels[i];
                totalLoss += -(y * Math.Log(yHat) + (1.0 - y) * Math.Log(1.0 - yHat));
            }

            return totalLoss / features.Count;
        }

        /// <summary>
        /// Computes the Mean Squared Error (MSE) over a dataset:
        /// MSE = (1/N) * sum( (y - y_hat)^2 )
        /// </summary>
        /// <param name="features">List of feature vectors.</param>
        /// <param name="labels">List of binary labels (0 or 1).</param>
        /// <returns>Mean Squared Error.</returns>
        public double ComputeMeanSquaredErrorLoss(IReadOnlyList<double[]> features, IReadOnlyList<int> labels)
        {
            if (features.Count == 0)
            {
                return 0.0;
            }

            double totalMse = 0.0;
            for (int i = 0; i < features.Count; i++)
            {
                double yHat = this.PredictProbability(features[i]);
                double diff = labels[i] - yHat;
                totalMse += diff * diff;
            }

            return totalMse / features.Count;
        }
    }
}
