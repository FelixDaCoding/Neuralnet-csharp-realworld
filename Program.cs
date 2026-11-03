using System;
using System.Globalization;
using System.IO;
using System.Linq;

namespace CsharpMlpDiabetesClassifier
{
    /// <summary>
    /// Entry point for the Real-World Pima Indians Diabetes Binary Classification project
    /// using the custom C# 3-layer Multi-Layer Perceptron (backneural).
    /// </summary>
    public static class Program
    {
        /// <summary>
        /// Orchestrates data loading, preprocessing, hyperparameter experiments, final retraining,
        /// and evaluation report generation.
        /// </summary>
        /// <param name="args">Command-line arguments (optional).</param>
        public static void Main(string[] args)
        {
            const int randomSeed = 42;
            string datasetPath = Path.Combine("data", "diabetes.csv");
            string experimentsCsvPath = Path.Combine("output", "experiments.csv");
            string trainingLogCsvPath = Path.Combine("output", "training_log.csv");
            string finalMetricsTxtPath = Path.Combine("output", "final_metrics.txt");

            Console.WriteLine("==============================================================");
            Console.WriteLine(" Real-World Prediction with a Custom Multi-Layer Perceptron");
            Console.WriteLine(" Track: Binary Classification (Pima Indians Diabetes Dataset)");
            Console.WriteLine("==============================================================");
            Console.WriteLine();

            // 1. Load and preprocess dataset (training-only median imputation and Min-Max scaling)
            DatasetSplit split = DataLoader.LoadAndPreprocess(datasetPath, trainRatio: 0.80, seed: randomSeed);

            int trainPos = split.TrainLabels.Count(l => l == 1);
            int trainNeg = split.TrainLabels.Count - trainPos;
            int testPos = split.TestLabels.Count(l => l == 1);
            int testNeg = split.TestLabels.Count - testPos;

            Console.WriteLine("Dataset Summary:");
            Console.WriteLine($"  Total Valid Rows   : {split.ValidRowCount} (Skipped malformed rows: {split.SkippedRowCount})");
            Console.WriteLine($"  Training Set (80%) : {split.TrainFeatures.Count} rows (Class 0: {trainNeg}, Class 1: {trainPos})");
            Console.WriteLine($"  Test Set     (20%) : {split.TestFeatures.Count} rows (Class 0: {testNeg}, Class 1: {testPos})");
            Console.WriteLine();
            Console.WriteLine("Zero-Value Imputation (Medians Computed from Training Set Only):");
            foreach (int col in DataLoader.ZeroImputeColumns)
            {
                string colName = split.FeatureNames[col];
                double median = split.ImputedTrainingMedians[col];
                int replaced = split.ZeroCountsReplaced[col];
                Console.WriteLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "  {0,-15} -> Training Median: {1,7:F2} (Zeros imputed across dataset: {2})",
                    colName,
                    median,
                    replaced));
            }
            Console.WriteLine();

            Console.WriteLine("Min-Max Normalization Bounds (Computed from Imputed Training Set Only):");
            for (int col = 0; col < split.FeatureNames.Length; col++)
            {
                Console.WriteLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "  {0,-24} : Min = {1,7:F3}, Max = {2,7:F3} -> Scaled to [0, 1]",
                    split.FeatureNames[col],
                    split.TrainMin[col],
                    split.TrainMax[col]));
            }
            Console.WriteLine();

            // 2. Define hyperparameter search space
            double[] learningRates = { 0.01, 0.05, 0.1, 0.5 };
            int[] hiddenNeuronSizes = { 4, 8, 16, 32 };
            int[] epochCounts = { 500, 1000, 2000 };

            // 3. Run systematic hyperparameter experiments, retrain best model, and save outputs
            ExperimentRunner.RunGridSearchAndRetrainBest(
                split,
                learningRates,
                hiddenNeuronSizes,
                epochCounts,
                experimentsCsvPath,
                trainingLogCsvPath,
                finalMetricsTxtPath,
                seed: randomSeed);

            Console.WriteLine("All output artifacts saved successfully:");
            Console.WriteLine($"  - {experimentsCsvPath}");
            Console.WriteLine($"  - {trainingLogCsvPath}");
            Console.WriteLine($"  - {finalMetricsTxtPath}");
        }
    }
}

