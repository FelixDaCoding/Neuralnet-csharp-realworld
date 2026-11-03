using System;

namespace backneural
{
    /// <summary>
    /// Represents an Output Layer Neuron in the custom 3-layer Multi-Layer Perceptron (backneural).
    /// Produces a sigmoid probability prediction in (0, 1) and computes output error gradients.
    /// </summary>
    public class ONeuron
    {
        private int idnum;
        private double act;
        private double bias;
        private double desired;
        private double error;

        /// <summary>
        /// Initializes a new output neuron with default values.
        /// </summary>
        public ONeuron()
        {
            this.idnum = 0;
            this.act = 0.0;
            this.bias = 0.0;
            this.desired = 0.0;
            this.error = 0.0;
        }

        /// <summary>
        /// Initializes a new output neuron with a seeded random bias.
        /// </summary>
        /// <param name="idnum">The neuron index in the output layer.</param>
        /// <param name="rand">Shared seeded random number generator for reproducible bias initialization.</param>
        public ONeuron(int idnum, Random rand)
        {
            this.idnum = idnum;
            this.act = 0.0;
            this.desired = 0.0;
            this.error = 0.0;
            this.bias = rand.NextDouble() - 0.5;
        }

        /// <summary>
        /// Sets the output neuron identifier.
        /// </summary>
        /// <param name="x">Neuron ID.</param>
        public void setId(int x)
        {
            this.idnum = x;
        }

        /// <summary>
        /// Gets the output neuron identifier.
        /// </summary>
        /// <returns>The neuron ID.</returns>
        public int getId()
        {
            return this.idnum;
        }

        /// <summary>
        /// Sets the output activation value.
        /// </summary>
        /// <param name="x">Activation value.</param>
        public void setAct(double x)
        {
            this.act = x;
        }

        /// <summary>
        /// Gets the current sigmoid activation (predicted probability) of the output neuron.
        /// </summary>
        /// <returns>Predicted probability in (0, 1).</returns>
        public double getAct()
        {
            return this.act;
        }

        /// <summary>
        /// Sets the bias of this output neuron.
        /// </summary>
        /// <param name="b">Bias value.</param>
        public void setBias(double b)
        {
            this.bias = b;
        }

        /// <summary>
        /// Gets the bias of this output neuron.
        /// </summary>
        /// <returns>Bias value.</returns>
        public double getBias()
        {
            return this.bias;
        }

        /// <summary>
        /// Sets the target/desired output value for supervised learning.
        /// </summary>
        /// <param name="x">Target label (0.0 or 1.0).</param>
        public void setDesired(double x)
        {
            this.desired = x;
        }

        /// <summary>
        /// Gets the target/desired output value.
        /// </summary>
        /// <returns>Target value.</returns>
        public double getDesired()
        {
            return this.desired;
        }

        /// <summary>
        /// Sets the output error delta.
        /// </summary>
        /// <param name="x">Error delta value.</param>
        public void setError(double x)
        {
            this.error = x;
        }

        /// <summary>
        /// Gets the output error delta.
        /// </summary>
        /// <returns>Error delta value.</returns>
        public double getError()
        {
            return this.error;
        }

        /// <summary>
        /// Forward pass step for this output neuron:
        /// Computes z_o = sum_h(a_h * w_{h,o}) + bias_o and applies logistic sigmoid activation 1 / (1 + e^{-z_o}).
        /// </summary>
        /// <param name="hlayer">Array of hidden layer neurons.</param>
        public void calculateActivation(HNeuron[] hlayer)
        {
            double osum = 0.0;
            for (int x = 0; x < hlayer.Length; x++)
            {
                osum += hlayer[x].getAct() * hlayer[x].getWeight(this.idnum);
            }
            osum += this.bias;
            this.act = 1.0 / (1.0 + Math.Exp(-osum));
        }

        /// <summary>
        /// Backpropagation step for this output neuron:
        /// Computes output error signal delta_o = (desired - act) * act * (1 - act).
        /// </summary>
        public void calculateError()
        {
            this.error = (this.desired - this.act) * this.act * (1.0 - this.act);
        }

        /// <summary>
        /// Weight update step for incoming connections from the hidden layer to this output neuron:
        /// Updates w_{h,o} += learningRate * delta_o * a_h and bias_o += learningRate * delta_o.
        /// </summary>
        /// <param name="hlayer">Array of hidden layer neurons.</param>
        /// <param name="learningRate">Step size (eta) for gradient descent.</param>
        public void newWeight(HNeuron[] hlayer, double learningRate)
        {
            for (int x = 0; x < hlayer.Length; x++)
            {
                double updated = hlayer[x].getWeight(this.idnum) + (learningRate * this.error * hlayer[x].getAct());
                hlayer[x].setWeight(this.idnum, updated);
            }
            this.bias += learningRate * this.error;
        }
    }
}
