using System;

namespace backneural
{
    /// <summary>
    /// Represents a Hidden Layer Neuron in the custom 3-layer Multi-Layer Perceptron (backneural).
    /// Computes sigmoid activation during forward propagation and propagates error gradients backward.
    /// </summary>
    public class HNeuron
    {
        private double[] weight;
        private int idnum;
        private double act;
        private double bias;
        private double error;
        private int weightSize;

        /// <summary>
        /// Initializes a new hidden neuron with default values.
        /// </summary>
        public HNeuron()
        {
            this.idnum = 0;
            this.act = 0.0;
            this.bias = 0.0;
            this.error = 0.0;
            this.weightSize = 10;
            this.weight = new double[this.weightSize];
        }

        /// <summary>
        /// Initializes a new hidden neuron connected to the specified number of output neurons.
        /// </summary>
        /// <param name="idnum">The neuron index in the hidden layer.</param>
        /// <param name="size">The number of output neurons in the next layer.</param>
        /// <param name="rand">Shared seeded random number generator for reproducible weight initialization.</param>
        public HNeuron(int idnum, int size, Random rand)
        {
            this.idnum = idnum;
            this.act = 0.0;
            this.error = 0.0;
            this.weightSize = size;
            this.weight = new double[this.weightSize];
            this.setRandomWeights(this.weightSize, rand);
        }

        /// <summary>
        /// Initializes hidden-to-output weights and hidden neuron bias symmetrically in [-0.5, 0.5].
        /// </summary>
        /// <param name="size">Number of output layer connections.</param>
        /// <param name="rand">Random number generator instance.</param>
        public void setRandomWeights(int size, Random rand)
        {
            for (int x = 0; x < size; x++)
            {
                this.weight[x] = rand.NextDouble() - 0.5;
            }
            this.bias = rand.NextDouble() - 0.5;
        }

        /// <summary>
        /// Sets the hidden neuron identifier.
        /// </summary>
        /// <param name="x">Neuron ID.</param>
        public void setId(int x)
        {
            this.idnum = x;
        }

        /// <summary>
        /// Gets the hidden neuron identifier.
        /// </summary>
        /// <returns>The neuron ID.</returns>
        public int getId()
        {
            return this.idnum;
        }

        /// <summary>
        /// Sets the current activation value of the hidden neuron.
        /// </summary>
        /// <param name="x">Activation value.</param>
        public void setAct(double x)
        {
            this.act = x;
        }

        /// <summary>
        /// Gets the current sigmoid activation value of the hidden neuron.
        /// </summary>
        /// <returns>Activation in (0, 1).</returns>
        public double getAct()
        {
            return this.act;
        }

        /// <summary>
        /// Sets the bias of this hidden neuron.
        /// </summary>
        /// <param name="b">Bias value.</param>
        public void setBias(double b)
        {
            this.bias = b;
        }

        /// <summary>
        /// Gets the bias of this hidden neuron.
        /// </summary>
        /// <returns>Bias value.</returns>
        public double getBias()
        {
            return this.bias;
        }

        /// <summary>
        /// Sets the backpropagated error delta for this hidden neuron.
        /// </summary>
        /// <param name="x">Error delta value.</param>
        public void setError(double x)
        {
            this.error = x;
        }

        /// <summary>
        /// Gets the backpropagated error delta for this hidden neuron.
        /// </summary>
        /// <returns>Error delta value.</returns>
        public double getError()
        {
            return this.error;
        }

        /// <summary>
        /// Sets the connection weight from this hidden neuron to a specific output neuron.
        /// </summary>
        /// <param name="pos">Output neuron index.</param>
        /// <param name="data">Weight value.</param>
        public void setWeight(int pos, double data)
        {
            this.weight[pos] = data;
        }

        /// <summary>
        /// Gets the connection weight from this hidden neuron to a specific output neuron.
        /// </summary>
        /// <param name="pos">Output neuron index.</param>
        /// <returns>Weight value.</returns>
        public double getWeight(int pos)
        {
            return this.weight[pos];
        }

        /// <summary>
        /// Forward pass step for this hidden neuron:
        /// Computes z_h = sum_i(x_i * w_{i,h}) + bias_h and applies logistic sigmoid activation 1 / (1 + e^{-z_h}).
        /// </summary>
        /// <param name="ilayer">Array of input layer neurons.</param>
        public void calculateActivation(INeuron[] ilayer)
        {
            double hsum = 0.0;
            for (int x = 0; x < ilayer.Length; x++)
            {
                hsum += ilayer[x].getAct() * ilayer[x].getWeight(this.idnum);
            }
            hsum += this.bias;
            this.act = 1.0 / (1.0 + Math.Exp(-hsum));
        }

        /// <summary>
        /// Backpropagation step for this hidden neuron:
        /// Computes delta_h = a_h * (1 - a_h) * sum_k(w_{h,k} * delta_k) using the current hidden-to-output weights.
        /// </summary>
        /// <param name="olayer">Array of output layer neurons with already computed output deltas.</param>
        public void calculateError(ONeuron[] olayer)
        {
            double errSum = 0.0;
            for (int x = 0; x < olayer.Length; x++)
            {
                errSum += olayer[x].getError() * this.weight[x];
            }
            this.error = this.act * (1.0 - this.act) * errSum;
        }

        /// <summary>
        /// Weight update step for incoming connections from the input layer to this hidden neuron:
        /// Updates w_{i,h} += learningRate * delta_h * x_i and bias_h += learningRate * delta_h.
        /// </summary>
        /// <param name="ilayer">Array of input layer neurons.</param>
        /// <param name="learningRate">Step size (eta) for gradient descent.</param>
        public void newWeight(INeuron[] ilayer, double learningRate)
        {
            for (int x = 0; x < ilayer.Length; x++)
            {
                double updated = ilayer[x].getWeight(this.idnum) + (learningRate * this.error * ilayer[x].getAct());
                ilayer[x].setWeight(this.idnum, updated);
            }
            this.bias += learningRate * this.error;
        }
    }
}

