using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using backneural;

namespace CsharpMlpDiabetesClassifier
{
    /// <summary>
    /// Stores binary classification evaluation metrics including the confusion matrix
    /// (TP, TN, FP, FN), Accuracy, Precision, Recall, and F1 Score.
    /// </summary>
    public class EvaluationResult
    {
        /// <summary>Gets the count of True Positives (actual 1, predicted 1).</summary>
        public int TruePositives { get; }

        /// <summary>Gets the count of True Negatives (actual 0, predicted 0).</summary>
        public int TrueNegatives { get; }

        /// <summary>Gets the count of False Positives (actual 0, predicted 1).</summary>
        public int FalsePositives { get; }

        /// <summary>Gets the count of False Negatives (actual 1, predicted 0).</summary>
        public int FalseNegatives { get; }

        /// <summary>Gets the overall classification accuracy: (TP + TN) / Total.</summary>
        public double Accuracy { get; }

        /// <summary>Gets the positive predictive value (precision): TP / (TP + FP).</summary>
        public double Precision { get; }

        /// <summary>Gets the sensitivity / true positive rate (recall): TP / (TP + FN).</summary>
        public double Recall { get; }

        /// <summary>Gets the harmonic mean of precision and recall: 2 * P * R / (P + R).</summary>
        public double F1Score { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="EvaluationResult"/> class.
        /// </summary>
        public EvaluationResult(
            int tp,
            int tn,
            int fp,
            int fn,
            double accuracy,
            double precision,
            double recall,
            double f1Score)
        {
            TruePositives = tp;
            TrueNegatives = tn;
            FalsePositives = fp;
            FalseNegatives = fn;
            Accuracy = accuracy;
            Precision = precision;
            Recall = recall;
            F1Score = f1Score;
        }
    }

    /// <summary>
    /// Computes confusion matrix and binary classification metrics (Accuracy, Precision, Recall, F1)
    /// using a 0.5 threshold on the neural network's sigmoid output, with safe divide-by-zero handling.
    /// </summary>
    public static class Metrics
    {
        /// <summary>
        /// Evaluates a trained <see cref="NeuralNet"/> on the provided feature and label lists
        /// using a decision threshold of 0.5.
        /// </summary>
        /// <param name="net">The trained custom MLP model.</param>
        /// <param name="features">Normalized input feature vectors.</param>
        /// <param name="labels">Ground-truth binary labels (0 or 1).</param>
        /// <param name="threshold">Decision threshold on the sigmoid output (default 0.5).</param>
        /// <returns>An <see cref="EvaluationResult"/> containing the confusion matrix and metrics.</returns>
        public static EvaluationResult Evaluate(
            NeuralNet net,
            IReadOnlyList<double[]> features,
            IReadOnlyList<int> labels,
            double threshold = 0.5)
        {
            int tp = 0;
            int tn = 0;
            int fp = 0;
            int fn = 0;

            for (int i = 0; i < features.Count; i++)
            {
                double prob = net.PredictProbability(features[i]);
                int predicted = prob >= threshold ? 1 : 0;
                int actual = labels[i];

                if (actual == 1 && predicted == 1)
                {
                    tp++;
                }
                else if (actual == 0 && predicted == 0)
                {
                    tn++;
                }
                else if (actual == 0 && predicted == 1)
                {
                    fp++;
                }
                else
                {
                    fn++;
                }
            }

            int total = tp + tn + fp + fn;
            double accuracy = total > 0 ? (double)(tp + tn) / total : 0.0;
            double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
            double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
            double f1 = (precision + recall) > 0.0 ? (2.0 * precision * recall) / (precision + recall) : 0.0;

            return new EvaluationResult(tp, tn, fp, fn, accuracy, precision, recall, f1);
        }

        /// <summary>
        /// Formats the evaluation metrics and confusion matrix into a human-readable report.
        /// </summary>
        /// <param name="result">Evaluation result to format.</param>
        /// <param name="learningRate">Learning rate used for the evaluated model.</param>
        /// <param name="hiddenNeurons">Hidden layer neuron count used for the evaluated model.</param>
        /// <param name="epochs">Number of training epochs used.</param>
        /// <param name="finalTrainLoss">Final training Binary Cross-Entropy loss.</param>
        /// <param name="finalTestLoss">Final test Binary Cross-Entropy loss.</param>
        /// <returns>Formatted report text.</returns>
        public static string FormatReport(
            EvaluationResult result,
            double learningRate,
            int hiddenNeurons,
            int epochs,
            double finalTrainLoss,
            double finalTestLoss)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("==============================================================");
            sb.AppendLine("   PIMA INDIANS DIABETES - CUSTOM C# MLP FINAL EVALUATION     ");
            sb.AppendLine("==============================================================");
            sb.AppendLine($"Best Hyperparameters:");
            sb.AppendLine($"  Learning Rate  : {learningRate.ToString("G", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"  Hidden Neurons : {hiddenNeurons}");
            sb.AppendLine($"  Epochs         : {epochs}");
            sb.AppendLine($"  Loss Function  : Binary Cross-Entropy (BCE)");
            sb.AppendLine($"  Threshold      : 0.5");
            sb.AppendLine();
            sb.AppendLine("Final Loss Values:");
            sb.AppendLine($"  Final Train BCE Loss : {finalTrainLoss.ToString("F6", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"  Final Test BCE Loss  : {finalTestLoss.ToString("F6", CultureInfo.InvariantCulture)}");
            sb.AppendLine();
            sb.AppendLine("Confusion Matrix (Test Set):");
            sb.AppendLine("                        Predicted Negative (0)   Predicted Positive (1)");
            sb.AppendLine($"  Actual Negative (0)   TN = {result.TrueNegatives,-19} FP = {result.FalsePositives}");
            sb.AppendLine($"  Actual Positive (1)   FN = {result.FalseNegatives,-19} TP = {result.TruePositives}");
            sb.AppendLine();
            sb.AppendLine("Classification Metrics (Test Set):");
            sb.AppendLine($"  Accuracy  : {result.Accuracy.ToString("F4", CultureInfo.InvariantCulture)} ({(result.Accuracy * 100.0).ToString("F2", CultureInfo.InvariantCulture)}%)");
            sb.AppendLine($"  Precision : {result.Precision.ToString("F4", CultureInfo.InvariantCulture)} ({(result.Precision * 100.0).ToString("F2", CultureInfo.InvariantCulture)}%)");
            sb.AppendLine($"  Recall    : {result.Recall.ToString("F4", CultureInfo.InvariantCulture)} ({(result.Recall * 100.0).ToString("F2", CultureInfo.InvariantCulture)}%)");
            sb.AppendLine($"  F1 Score  : {result.F1Score.ToString("F4", CultureInfo.InvariantCulture)} ({(result.F1Score * 100.0).ToString("F2", CultureInfo.InvariantCulture)}%)");
            sb.AppendLine("==============================================================");
            return sb.ToString();
        }

        /// <summary>
        /// Prints the formatted evaluation report to the console and saves it to a relative file path.
        /// </summary>
        /// <param name="outputRelativePath">Relative file path (e.g., "output/final_metrics.txt").</param>
        /// <param name="result">Evaluation result.</param>
        /// <param name="learningRate">Best learning rate.</param>
        /// <param name="hiddenNeurons">Best hidden neuron count.</param>
        /// <param name="epochs">Best epoch count.</param>
        /// <param name="finalTrainLoss">Final training loss.</param>
        /// <param name="finalTestLoss">Final test loss.</param>
        public static void PrintAndSaveFinalMetrics(
            string outputRelativePath,
            EvaluationResult result,
            double learningRate,
            int hiddenNeurons,
            int epochs,
            double finalTrainLoss,
            double finalTestLoss)
        {
            string report = FormatReport(result, learningRate, hiddenNeurons, epochs, finalTrainLoss, finalTestLoss);
            Console.WriteLine();
            Console.WriteLine(report);

            string? dir = Path.GetDirectoryName(outputRelativePath);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }

            File.WriteAllText(outputRelativePath, report);
        }
    }
}

