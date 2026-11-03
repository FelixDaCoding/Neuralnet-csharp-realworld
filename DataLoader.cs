using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace CsharpMlpDiabetesClassifier
{
    /// <summary>
    /// Holds the preprocessed training and testing splits along with preprocessing statistics
    /// (training-set medians, min/max values, and row counts) to ensure zero data leakage.
    /// </summary>
    public class DatasetSplit
    {
        /// <summary>Gets the normalized training feature vectors.</summary>
        public List<double[]> TrainFeatures { get; }

        /// <summary>Gets the binary training labels (0 or 1).</summary>
        public List<int> TrainLabels { get; }

        /// <summary>Gets the normalized testing feature vectors.</summary>
        public List<double[]> TestFeatures { get; }

        /// <summary>Gets the binary testing labels (0 or 1).</summary>
        public List<int> TestLabels { get; }

        /// <summary>Gets the feature column names from the CSV header.</summary>
        public string[] FeatureNames { get; }

        /// <summary>Gets the training-set medians used to impute zero values (keyed by column index).</summary>
        public Dictionary<int, double> ImputedTrainingMedians { get; }

        /// <summary>Gets the count of zero values replaced per column in the entire dataset.</summary>
        public Dictionary<int, int> ZeroCountsReplaced { get; }

        /// <summary>Gets the per-feature minimum values computed on the imputed training set.</summary>
        public double[] TrainMin { get; }

        /// <summary>Gets the per-feature maximum values computed on the imputed training set.</summary>
        public double[] TrainMax { get; }

        /// <summary>Gets the total number of valid rows loaded from the CSV.</summary>
        public int ValidRowCount { get; }

        /// <summary>Gets the number of malformed or empty rows skipped during CSV parsing.</summary>
        public int SkippedRowCount { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="DatasetSplit"/> class.
        /// </summary>
        public DatasetSplit(
            List<double[]> trainFeatures,
            List<int> trainLabels,
            List<double[]> testFeatures,
            List<int> testLabels,
            string[] featureNames,
            Dictionary<int, double> imputedTrainingMedians,
            Dictionary<int, int> zeroCountsReplaced,
            double[] trainMin,
            double[] trainMax,
            int validRowCount,
            int skippedRowCount)
        {
            TrainFeatures = trainFeatures;
            TrainLabels = trainLabels;
            TestFeatures = testFeatures;
            TestLabels = testLabels;
            FeatureNames = featureNames;
            ImputedTrainingMedians = imputedTrainingMedians;
            ZeroCountsReplaced = zeroCountsReplaced;
            TrainMin = trainMin;
            TrainMax = trainMax;
            ValidRowCount = validRowCount;
            SkippedRowCount = skippedRowCount;
        }
    }

    /// <summary>
    /// Handles manual CSV loading, stratified train/test splitting, training-only median imputation
    /// for biologically invalid zero values, and training-only Min-Max feature normalization to [0, 1].
    /// </summary>
    public static class DataLoader
    {
        /// <summary>
        /// Indices of columns where a value of 0 represents a missing measurement:
        /// 1 = Glucose, 2 = BloodPressure, 3 = SkinThickness, 4 = Insulin, 5 = BMI.
        /// </summary>
        public static readonly int[] ZeroImputeColumns = { 1, 2, 3, 4, 5 };

        /// <summary>
        /// Standard feature names for the Pima Indians Diabetes dataset.
        /// </summary>
        public static readonly string[] DefaultFeatureNames =
        {
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        };

        /// <summary>
        /// Resolves a relative file path by checking the current working directory first
        /// and then ascending parent directories if run from bin/Debug/net8.0.
        /// </summary>
        /// <param name="relativePath">Relative path such as "data/diabetes.csv".</param>
        /// <returns>Resolved relative path.</returns>
        public static string ResolveRelativePath(string relativePath)
        {
            if (File.Exists(relativePath) || Directory.Exists(relativePath))
            {
                return relativePath;
            }

            string current = ".";
            for (int depth = 0; depth < 4; depth++)
            {
                current = Path.Combine(current, "..");
                string candidate = Path.Combine(current, relativePath);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            return relativePath;
        }

        /// <summary>
        /// Loads the Pima Indians Diabetes CSV file, performs a seeded stratified 80/20 train/test split,
        /// imputes missing zeros using medians from the training set only, and applies Min-Max scaling
        /// to [0, 1] using min/max bounds from the training set only.
        /// </summary>
        /// <param name="csvRelativePath">Relative path to the CSV file (e.g., "data/diabetes.csv").</param>
        /// <param name="trainRatio">Proportion of data used for training (default 0.80).</param>
        /// <param name="seed">Fixed random seed for deterministic shuffling (default 42).</param>
        /// <returns>A <see cref="DatasetSplit"/> containing normalized train/test data and metadata.</returns>
        public static DatasetSplit LoadAndPreprocess(string csvRelativePath, double trainRatio = 0.80, int seed = 42)
        {
            string resolvedPath = ResolveRelativePath(csvRelativePath);
            if (!File.Exists(resolvedPath))
            {
                throw new FileNotFoundException(
                    $"Dataset not found at '{csvRelativePath}'. Please download the Pima Indians Diabetes dataset " +
                    "from Kaggle (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) or UCI " +
                    "and place it in 'data/diabetes.csv'.");
            }

            string[] lines = File.ReadAllLines(resolvedPath);
            if (lines.Length <= 1)
            {
                throw new InvalidDataException($"The dataset file '{resolvedPath}' contains no data rows.");
            }

            // Parse header names if present
            string[] headerParts = lines[0].Split(',');
            string[] featureNames = DefaultFeatureNames;
            if (headerParts.Length == 9)
            {
                featureNames = headerParts.Take(8).Select(h => h.Trim()).ToArray();
            }

            List<(double[] Features, int Label)> allRecords = new List<(double[] Features, int Label)>();
            int skippedRows = 0;

            // Parse CSV rows manually (skipping line 0 header)
            for (int lineIdx = 1; lineIdx < lines.Length; lineIdx++)
            {
                string rawLine = lines[lineIdx].Trim();
                if (string.IsNullOrWhiteSpace(rawLine))
                {
                    skippedRows++;
                    continue;
                }

                string[] tokens = rawLine.Split(',');
                if (tokens.Length != 9)
                {
                    skippedRows++;
                    continue;
                }

                double[] features = new double[8];
                bool validRow = true;

                for (int col = 0; col < 8; col++)
                {
                    if (!double.TryParse(tokens[col].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double val)
                        || double.IsNaN(val) || double.IsInfinity(val))
                    {
                        validRow = false;
                        break;
                    }
                    features[col] = val;
                }

                if (!validRow)
                {
                    skippedRows++;
                    continue;
                }

                if (!int.TryParse(tokens[8].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int label)
                    || (label != 0 && label != 1))
                {
                    // Also handle "0.0" or "1.0" formatted labels safely
                    if (double.TryParse(tokens[8].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double dLabel)
                        && (Math.Abs(dLabel - 0.0) < 1e-9 || Math.Abs(dLabel - 1.0) < 1e-9))
                    {
                        label = (int)Math.Round(dLabel);
                    }
                    else
                    {
                        skippedRows++;
                        continue;
                    }
                }

                allRecords.Add((features, label));
            }

            if (allRecords.Count == 0)
            {
                throw new InvalidDataException("No valid records were parsed from the CSV file.");
            }

            // Step 1: Stratified 80/20 split after shuffling with fixed random seed
            Random rng = new Random(seed);
            List<(double[] Features, int Label)> class0 = allRecords.Where(r => r.Label == 0).ToList();
            List<(double[] Features, int Label)> class1 = allRecords.Where(r => r.Label == 1).ToList();

            ShuffleInPlace(class0, rng);
            ShuffleInPlace(class1, rng);

            int trainCount0 = (int)Math.Round(class0.Count * trainRatio);
            int trainCount1 = (int)Math.Round(class1.Count * trainRatio);

            List<(double[] Features, int Label)> trainRecords = new List<(double[] Features, int Label)>();
            List<(double[] Features, int Label)> testRecords = new List<(double[] Features, int Label)>();

            trainRecords.AddRange(class0.Take(trainCount0));
            trainRecords.AddRange(class1.Take(trainCount1));
            testRecords.AddRange(class0.Skip(trainCount0));
            testRecords.AddRange(class1.Skip(trainCount1));

            // Shuffle combined train and test splits so classes are interleaved
            ShuffleInPlace(trainRecords, rng);
            ShuffleInPlace(testRecords, rng);

            List<double[]> trainFeatures = trainRecords.Select(r => (double[])r.Features.Clone()).ToList();
            List<int> trainLabels = trainRecords.Select(r => r.Label).ToList();
            List<double[]> testFeatures = testRecords.Select(r => (double[])r.Features.Clone()).ToList();
            List<int> testLabels = testRecords.Select(r => r.Label).ToList();

            // Step 2: Compute column medians for non-zero values from the TRAINING SET ONLY
            Dictionary<int, double> trainingMedians = new Dictionary<int, double>();
            Dictionary<int, int> zeroCountsReplaced = new Dictionary<int, int>();

            foreach (int col in ZeroImputeColumns)
            {
                List<double> nonZeroTrainValues = trainFeatures
                    .Select(row => row[col])
                    .Where(val => val > 0.0)
                    .OrderBy(val => val)
                    .ToList();

                double median = ComputeMedianFromSorted(nonZeroTrainValues);
                trainingMedians[col] = median;

                int replacedCount = 0;
                for (int i = 0; i < trainFeatures.Count; i++)
                {
                    if (Math.Abs(trainFeatures[i][col]) < 1e-12)
                    {
                        trainFeatures[i][col] = median;
                        replacedCount++;
                    }
                }

                for (int i = 0; i < testFeatures.Count; i++)
                {
                    if (Math.Abs(testFeatures[i][col]) < 1e-12)
                    {
                        testFeatures[i][col] = median;
                        replacedCount++;
                    }
                }

                zeroCountsReplaced[col] = replacedCount;
            }

            // Step 3: Compute Min and Max per feature column from the imputed TRAINING SET ONLY
            int numFeatures = 8;
            double[] trainMin = new double[numFeatures];
            double[] trainMax = new double[numFeatures];

            for (int col = 0; col < numFeatures; col++)
            {
                double minVal = double.MaxValue;
                double maxVal = double.MinValue;

                for (int i = 0; i < trainFeatures.Count; i++)
                {
                    double v = trainFeatures[i][col];
                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                }

                trainMin[col] = minVal;
                trainMax[col] = maxVal;
            }

            // Step 4: Apply Min-Max scaling to [0, 1] on both training and testing sets using training bounds
            ApplyMinMaxScaling(trainFeatures, trainMin, trainMax);
            ApplyMinMaxScaling(testFeatures, trainMin, trainMax);

            return new DatasetSplit(
                trainFeatures,
                trainLabels,
                testFeatures,
                testLabels,
                featureNames,
                trainingMedians,
                zeroCountsReplaced,
                trainMin,
                trainMax,
                allRecords.Count,
                skippedRows);
        }

        /// <summary>
        /// Performs a Fisher-Yates shuffle in-place using the provided random number generator.
        /// </summary>
        private static void ShuffleInPlace<T>(IList<T> list, Random rng)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int swapIdx = rng.Next(i + 1);
                T temp = list[i];
                list[i] = list[swapIdx];
                list[swapIdx] = temp;
            }
        }

        /// <summary>
        /// Computes the median value of a pre-sorted list of numbers.
        /// </summary>
        private static double ComputeMedianFromSorted(IReadOnlyList<double> sortedValues)
        {
            if (sortedValues.Count == 0)
            {
                return 0.0;
            }

            int mid = sortedValues.Count / 2;
            if (sortedValues.Count % 2 == 1)
            {
                return sortedValues[mid];
            }

            return (sortedValues[mid - 1] + sortedValues[mid]) / 2.0;
        }

        /// <summary>
        /// Scales each feature column in-place using (x - trainMin) / (trainMax - trainMin).
        /// </summary>
        private static void ApplyMinMaxScaling(List<double[]> rows, double[] trainMin, double[] trainMax)
        {
            for (int i = 0; i < rows.Count; i++)
            {
                for (int col = 0; col < trainMin.Length; col++)
                {
                    double range = trainMax[col] - trainMin[col];
                    if (Math.Abs(range) < 1e-12)
                    {
                        rows[i][col] = 0.0;
                    }
                    else
                    {
                        rows[i][col] = (rows[i][col] - trainMin[col]) / range;
                    }
                }
            }
        }
    }
}
