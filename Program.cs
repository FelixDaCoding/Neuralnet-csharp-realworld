using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace CsharpMlpDiabetesClassifier
{
    /// <summary>
    /// Entry point for the Real-World Pima Indians Diabetes Binary Classification project
    /// using the custom C# 3-layer Multi-Layer Perceptron (backneural).
    /// Launches the interactive Windows Forms GUI by default, or runs in CLI mode if "--cli" is passed.
    /// </summary>
    public static class Program
    {
        /// <summary>
        /// Main entry point. Launches the Windows Forms GUI or runs the CLI batch pipeline.
        /// </summary>
        /// <param name="args">Command-line arguments. Pass "--cli" for console-only batch execution.</param>
        [STAThread]
        public static void Main(string[] args)
        {
            if (args != null && args.Any(a => string.Equals(a, "--cli", StringComparison.OrdinalIgnoreCase)))
            {
                RunCliPipeline();
                return;
            }

            Application.SetHighDpiMode(HighDpiMode.SystemAware);
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }

        /// <summary>
        /// Runs the headless batch pipeline (data loading, preprocessing, 48-run grid search,
        /// best model retraining, and saving output files).
        /// </summary>
        public static void RunCliPipeline()
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

            double[] learningRates = { 0.01, 0.05, 0.1, 0.5 };
            int[] hiddenNeuronSizes = { 4, 8, 16, 32 };
            int[] epochCounts = { 500, 1000, 2000 };

            ExperimentRunner.RunGridSearchAndRetrainBest(
                split,
                learningRates,
                hiddenNeuronSizes,
                epochCounts,
                experimentsCsvPath,
                trainingLogCsvPath,
                finalMetricsTxtPath,
                seed: randomSeed);
        }
    }
}
