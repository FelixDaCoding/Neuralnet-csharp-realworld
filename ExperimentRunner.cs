using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using backneural;

namespace CsharpMlpDiabetesClassifier
{
    /// <summary>
    /// Represents a single hyperparameter configuration and its resulting training and test metrics.
    /// </summary>
    public class ExperimentRecord
    {
        /// <summary>Gets the learning rate tested.</summary>
        public double LearningRate { get; }

        /// <summary>Gets the number of hidden layer neurons tested.</summary>
        public int HiddenNeurons { get; }

        /// <summary>Gets the number of training epochs tested.</summary>
        public int Epochs { get; }

        /// <summary>Gets the final training Binary Cross-Entropy loss at the end of training.</summary>
        public double FinalTrainLoss { get; }

        /// <summary>Gets the final test Binary Cross-Entropy loss at the end of training.</summary>
        public double FinalTestLoss { get; }

        /// <summary>Gets the test set evaluation metrics (confusion matrix, accuracy, precision, recall, F1).</summary>
        public EvaluationResult TestMetrics { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ExperimentRecord"/> class.
        /// </summary>
        public ExperimentRecord(
            double learningRate,
            int hiddenNeurons,
            int epochs,
            double finalTrainLoss,
            double finalTestLoss,
            EvaluationResult testMetrics)
        {
            LearningRate = learningRate;
            HiddenNeurons = hiddenNeurons;
            Epochs = epochs;
            FinalTrainLoss = finalTrainLoss;
            FinalTestLoss = finalTestLoss;
            TestMetrics = testMetrics;
        }
    }

    /// <summary>
    /// Runs systematic hyperparameter grid search experiments across learning rates, hidden layer sizes,
    /// and epoch counts, saves results to CSV, selects the best configuration, and retrains the best model.
    /// </summary>
    public static class ExperimentRunner
    {
        /// <summary>
        /// Trains a new <see cref="NeuralNet"/> instance with the specified hyperparameters on the training split.
        /// Optionally prints progress every 100 epochs and records the epoch loss history to CSV.
        /// </summary>
        /// <param name="split">Preprocessed train/test dataset split.</param>
        /// <param name="learningRate">Learning rate (eta) for backpropagation updates.</param>
        /// <param name="hiddenNeurons">Number of neurons in the hidden layer.</param>
        /// <param name="epochs">Total number of training epochs.</param>
        /// <param name="logProgress">Whether to print loss every 100 epochs and save training_log.csv.</param>
        /// <param name="trainingLogRelativePath">Relative file path for saving epoch loss log.</param>
        /// <param name="seed">Fixed random seed for deterministic weight initialization.</param>
        /// <returns>Tuple of the trained <see cref="NeuralNet"/>, final train BCE loss, and final test BCE loss.</returns>
        public static (NeuralNet Net, double FinalTrainLoss, double FinalTestLoss) TrainModel(
            DatasetSplit split,
            double learningRate,
            int hiddenNeurons,
            int epochs,
            bool logProgress = false,
            string? trainingLogRelativePath = null,
            int seed = 42)
        {
            int inputCount = split.FeatureNames.Length;
            NeuralNet net = new NeuralNet(inputCount, hiddenNeurons, 1, learningRate, seed);

            StringBuilder? logBuilder = null;
            if (logProgress && !string.IsNullOrEmpty(trainingLogRelativePath))
            {
                logBuilder = new StringBuilder();
                logBuilder.AppendLine("epoch,train_loss,test_loss");
            }

            double trainLoss = 0.0;
            double testLoss = 0.0;

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                // Online / per-sample backpropagation pass over all training examples
                for (int i = 0; i < split.TrainFeatures.Count; i++)
                {
                    net.setInputs(split.TrainFeatures[i]);
                    net.setDesiredOutput(0, split.TrainLabels[i]);
                    net.learn();
                }

                // Compute loss whenever logging is active or on the final epoch
                if (logProgress || epoch == epochs)
                {
                    trainLoss = net.ComputeBinaryCrossEntropyLoss(split.TrainFeatures, split.TrainLabels);
                    testLoss = net.ComputeBinaryCrossEntropyLoss(split.TestFeatures, split.TestLabels);

                    if (logBuilder != null)
                    {
                        logBuilder.AppendLine(string.Format(
                            CultureInfo.InvariantCulture,
                            "{0},{1:F6},{2:F6}",
                            epoch,
                            trainLoss,
                            testLoss));
                    }

                    if (logProgress && (epoch == 1 || epoch % 100 == 0 || epoch == epochs))
                    {
                        Console.WriteLine(string.Format(
                            CultureInfo.InvariantCulture,
                            "  Epoch {0,4}/{1} | Train BCE Loss: {2:F6} | Test BCE Loss: {3:F6}",
                            epoch,
                            epochs,
                            trainLoss,
                            testLoss));
                    }
                }
            }

            if (logBuilder != null && !string.IsNullOrEmpty(trainingLogRelativePath))
            {
                string? dir = Path.GetDirectoryName(trainingLogRelativePath);
                if (!string.IsNullOrEmpty(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                File.WriteAllText(trainingLogRelativePath, logBuilder.ToString());
            }

            return (net, trainLoss, testLoss);
        }

        /// <summary>
        /// Executes the full hyperparameter grid search, saves all runs to output/experiments.csv,
        /// selects the best configuration by test F1 score (with test accuracy as tie-breaker),
        /// and retrains the best model to generate output/training_log.csv and output/final_metrics.txt.
        /// </summary>
        /// <param name="split">Preprocessed train/test dataset split.</param>
        /// <param name="learningRates">Array of learning rates to evaluate.</param>
        /// <param name="hiddenNeuronSizes">Array of hidden neuron counts to evaluate.</param>
        /// <param name="epochCounts">Array of epoch counts to evaluate.</param>
        /// <param name="experimentsCsvPath">Relative path for saving experiments.csv.</param>
        /// <param name="trainingLogCsvPath">Relative path for saving training_log.csv.</param>
        /// <param name="finalMetricsTxtPath">Relative path for saving final_metrics.txt.</param>
        /// <param name="seed">Fixed random seed for reproducible weight initialization.</param>
        /// <returns>Tuple containing all experiment records and the best selected record.</returns>
        public static (List<ExperimentRecord> AllRuns, ExperimentRecord BestRun) RunGridSearchAndRetrainBest(
            DatasetSplit split,
            double[] learningRates,
            int[] hiddenNeuronSizes,
            int[] epochCounts,
            string experimentsCsvPath,
            string trainingLogCsvPath,
            string finalMetricsTxtPath,
            int seed = 42)
        {
            List<ExperimentRecord> records = new List<ExperimentRecord>();
            int totalRuns = learningRates.Length * hiddenNeuronSizes.Length * epochCounts.Length;
            int runIndex = 0;

            Console.WriteLine("--------------------------------------------------------------");
            Console.WriteLine($"Running Hyperparameter Grid Search ({totalRuns} combinations)...");
            Console.WriteLine("--------------------------------------------------------------");
            Console.WriteLine("Run |   LR   | Hidden | Epochs | TrainLoss | TestAcc |  Prec  | Recall |   F1   ");
            Console.WriteLine("----+--------+--------+--------+-----------+---------+--------+--------+--------");

            foreach (double lr in learningRates)
            {
                foreach (int hidden in hiddenNeuronSizes)
                {
                    foreach (int epochs in epochCounts)
                    {
                        runIndex++;
                        var (net, finalTrainLoss, finalTestLoss) = TrainModel(
                            split,
                            lr,
                            hidden,
                            epochs,
                            logProgress: false,
                            trainingLogRelativePath: null,
                            seed: seed);

                        EvaluationResult eval = Metrics.Evaluate(net, split.TestFeatures, split.TestLabels, threshold: 0.5);
                        ExperimentRecord record = new ExperimentRecord(lr, hidden, epochs, finalTrainLoss, finalTestLoss, eval);
                        records.Add(record);

                        Console.WriteLine(string.Format(
                            CultureInfo.InvariantCulture,
                            "{0,3} | {1,6:F2} | {2,6} | {3,6} |  {4,8:F5} |  {5,6:F4} | {6,6:F4} | {7,6:F4} | {8,6:F4}",
                            runIndex,
                            lr,
                            hidden,
                            epochs,
                            finalTrainLoss,
                            eval.Accuracy,
                            eval.Precision,
                            eval.Recall,
                            eval.F1Score));
                    }
                }
            }

            // Save all experiment runs to output/experiments.csv
            SaveExperimentsCsv(records, experimentsCsvPath);

            // Pick the best configuration by Test F1 Score (tie-breaking by Test Accuracy, then lower FinalTrainLoss)
            ExperimentRecord bestRun = records
                .OrderByDescending(r => r.TestMetrics.F1Score)
                .ThenByDescending(r => r.TestMetrics.Accuracy)
                .ThenBy(r => r.FinalTrainLoss)
                .First();

            Console.WriteLine();
            Console.WriteLine("--------------------------------------------------------------");
            Console.WriteLine(string.Format(
                CultureInfo.InvariantCulture,
                "Best Configuration Selected (by Test F1): LR={0}, Hidden={1}, Epochs={2} (F1={3:F4}, Acc={4:F4})",
                bestRun.LearningRate,
                bestRun.HiddenNeurons,
                bestRun.Epochs,
                bestRun.TestMetrics.F1Score,
                bestRun.TestMetrics.Accuracy));
            Console.WriteLine("Retraining best model and writing epoch loss log to " + trainingLogCsvPath + "...");
            Console.WriteLine("--------------------------------------------------------------");

            var (bestNet, retrainedTrainLoss, retrainedTestLoss) = TrainModel(
                split,
                bestRun.LearningRate,
                bestRun.HiddenNeurons,
                bestRun.Epochs,
                logProgress: true,
                trainingLogRelativePath: trainingLogCsvPath,
                seed: seed);

            EvaluationResult finalEval = Metrics.Evaluate(bestNet, split.TestFeatures, split.TestLabels, threshold: 0.5);
            Metrics.PrintAndSaveFinalMetrics(
                finalMetricsTxtPath,
                finalEval,
                bestRun.LearningRate,
                bestRun.HiddenNeurons,
                bestRun.Epochs,
                retrainedTrainLoss,
                retrainedTestLoss);

            return (records, bestRun);
        }

        /// <summary>
        /// Writes all experiment results to a CSV file with the required columns:
        /// learning_rate,hidden_neurons,epochs,final_train_loss,test_accuracy,precision,recall,f1
        /// </summary>
        private static void SaveExperimentsCsv(IReadOnlyList<ExperimentRecord> records, string csvPath)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("learning_rate,hidden_neurons,epochs,final_train_loss,test_accuracy,precision,recall,f1");

            foreach (ExperimentRecord r in records)
            {
                sb.AppendLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "{0},{1},{2},{3:F6},{4:F4},{5:F4},{6:F4},{7:F4}",
                    r.LearningRate.ToString("G", CultureInfo.InvariantCulture),
                    r.HiddenNeurons,
                    r.Epochs,
                    r.FinalTrainLoss,
                    r.TestMetrics.Accuracy,
                    r.TestMetrics.Precision,
                    r.TestMetrics.Recall,
                    r.TestMetrics.F1Score));
            }

            string? dir = Path.GetDirectoryName(csvPath);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }

            File.WriteAllText(csvPath, sb.ToString());
        }
    }
}

