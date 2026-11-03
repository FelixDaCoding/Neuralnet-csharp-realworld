using System;

namespace backneural
{
    /// <summary>
    /// Represents an Input Layer Neuron in the custom 3-layer Multi-Layer Perceptron (backneural).
    /// Holds the raw/normalized feature value and outgoing weights connecting to the hidden layer.
    /// </summary>
    public class INeuron
    {
        private int idnum;
        private double act;
        private double[] weight;
        private int weightSize;

        /// <summary>
        /// Initializes a new input neuron with default values.
        /// </summary>
        public INeuron()
        {
            this.idnum = 0;
            this.act = 0.0;
            this.weightSize = 10;
            this.weight = new double[this.weightSize];
        }

        /// <summary>
        /// Initializes a new input neuron connected to the specified number of hidden neurons.
        /// </summary>
        /// <param name="idnum">The neuron index in the input layer.</param>
        /// <param name="size">The number of hidden neurons in the next layer.</param>
        /// <param name="rand">Shared seeded random number generator for reproducible weight initialization.</param>
        public INeuron(int idnum, int size, Random rand)
        {
            this.idnum = idnum;
            this.act = 0.0;
            this.weightSize = size;
            this.weight = new double[this.weightSize];
            this.setRandomWeights(this.weightSize, rand);
        }

        /// <summary>
        /// Initializes outgoing input-to-hidden weights symmetrically in [-0.5, 0.5]
        /// to prevent initial sigmoid saturation.
        /// </summary>
        /// <param name="size">Number of hidden layer connections.</param>
        /// <param name="rand">Random number generator instance.</param>
        public void setRandomWeights(int size, Random rand)
        {
            for (int x = 0; x < size; x++)
            {
                this.weight[x] = rand.NextDouble() - 0.5;
            }
        }

        /// <summary>
        /// Sets the neuron identifier.
        /// </summary>
        /// <param name="x">Neuron ID.</param>
        public void setId(int x)
        {
            this.idnum = x;
        }

        /// <summary>
        /// Gets the neuron identifier.
        /// </summary>
        /// <returns>The neuron ID.</returns>
        public int getId()
        {
            return this.idnum;
        }

        /// <summary>
        /// Sets the activation (input feature value) of this input neuron.
        /// </summary>
        /// <param name="x">Input feature value.</param>
        public void setAct(double x)
        {
            this.act = x;
        }

        /// <summary>
        /// Gets the current activation (input feature value) of this input neuron.
        /// </summary>
        /// <returns>The current activation value.</returns>
        public double getAct()
        {
            return this.act;
        }

        /// <summary>
        /// Sets the connection weight from this input neuron to a specific hidden neuron.
        /// </summary>
        /// <param name="pos">Hidden neuron index.</param>
        /// <param name="data">New weight value.</param>
        public void setWeight(int pos, double data)
        {
            this.weight[pos] = data;
        }

        /// <summary>
        /// Gets the connection weight from this input neuron to a specific hidden neuron.
        /// </summary>
        /// <param name="pos">Hidden neuron index.</param>
        /// <returns>Weight value.</returns>
        public double getWeight(int pos)
        {
            return this.weight[pos];
        }
    }
}
