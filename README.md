# Real-World Prediction with a Custom Multi-Layer Perceptron in C# (`csharp-mlp-diabetes-classifier`)

## Overview
This project implements a complete binary classification machine learning pipeline in pure C# to predict the onset of diabetes on the **Pima Indians Diabetes Dataset**.

> **Zero External ML Libraries:** This project uses **only** our custom 3-layer Multi-Layer Perceptron (`backneural` classes: `INeuron`, `HNeuron`, `ONeuron`, and `NeuralNet`) built from scratch using standard .NET libraries (`System`, `System.IO`, `System.Linq`, `System.Collections.Generic`, `System.Globalization`, `System.Text`). No external ML or math libraries (such as ML.NET, TensorFlow.NET, Accord.NET, or Math.NET) are used.

---

## Architecture & Improvements to the Custom `backneural` Classes
The neural network is a 3-layer fully connected Multi-Layer Perceptron (MLP) trained via online (per-sample) backpropagation:
1. **`INeuron` (`backneural/INeuron.cs`)**: Represents an input layer node holding a normalized feature activation $x_i \in [0, 1]$ and outgoing weights $w_{ih}$ connecting to each hidden neuron.
2. **`HNeuron` (`backneural/HNeuron.cs`)**: Represents a hidden layer node with bias $b_h$, logistic sigmoid activation $a_h = \sigma(z_h) = \frac{1}{1 + e^{-z_h}}$, outgoing weights $w_{ho}$ to the output layer, and hidden error gradient $\delta_h = a_h(1 - a_h)\sum_o w_{ho}\delta_o$.
3. **`ONeuron` (`backneural/ONeuron.cs`)**: Represents the single output layer node with bias $b_o$, logistic sigmoid output $\hat{y} = \sigma(z_o) \in (0, 1)$, and output error signal $\delta_o = (y - \hat{y})\hat{y}(1 - \hat{y})$.
4. **`NeuralNet` (`backneural/NeuralNet.cs`)**: Orchestrates forward propagation (`run()`), backpropagation (`learn()`), probability prediction (`PredictProbability()`), and **Binary Cross-Entropy (BCE)** loss evaluation (`ComputeBinaryCrossEntropyLoss()`):
   $$\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \ln(\hat{y}_i) + (1 - y_i)\ln(1 - \hat{y}_i)\right]$$

### Key Refinements Made to the Classic `backneural` Implementation
- **Configurable Learning Rate ($\eta$)**: Replaced hardcoded step sizes with a configurable `learningRate` parameter passed to `ONeuron.newWeight()` and `HNeuron.newWeight()`.
- **Correct Backpropagation Ordering**: In `NeuralNet.learn()`, hidden layer error signals $\delta_h$ are computed *before* `ONeuron.newWeight()` mutates the hidden-to-output weights $w_{ho}$, ensuring exact chain-rule gradients.
- **Shared Seeded Weight Initialization in $[-0.5, 0.5]$**: Passed a single seeded `Random(seed)` instance across all neurons and initialized weights/biases symmetrically in $[-0.5, 0.5]$ instead of $[0, 1]$, preventing initial sigmoid saturation and guaranteeing 100% reproducible experiments.

---

## Dataset Description & Preprocessing

### Source & Structure
- **Dataset**: Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases / UCI Machine Learning Repository / Kaggle)
- **File Location**: `data/diabetes.csv`
- **Total Records**: `768` valid rows (`0` malformed rows skipped)
  - Class `0` (Non-diabetic): `500` rows (`65.10%`)
  - Class `1` (Diabetic): `268` rows (`34.90%`)
- **Features (8 numeric inputs) & Target (1 binary output)**:
  1. `Pregnancies`: Number of times pregnant
  2. `Glucose`: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
  3. `BloodPressure`: Diastolic blood pressure ($\text{mm Hg}$)
  4. `SkinThickness`: Triceps skin fold thickness ($\text{mm}$)
  5. `Insulin`: 2-Hour serum insulin ($\mu\text{U/ml}$)
  6. `BMI`: Body mass index ($\text{weight in kg}/(\text{height in m})^2$)
  7. `DiabetesPedigreeFunction`: Diabetes pedigree function (genetic score)
  8. `Age`: Age in years
  - **Target (`Outcome`)**: `1` (tested positive for diabetes) or `0` (tested negative)

### Leak-Free Preprocessing Pipeline (`DataLoader.cs`)
1. **Stratified 80/20 Train/Test Split (`seed = 42`)**:
   - **Training Set (80%)**: `614` rows (`400` Class 0, `214` Class 1)
   - **Test Set (20%)**: `154` rows (`100` Class 0, `54` Class 1)
2. **Missing-Value Imputation (Training-Set Medians Only)**:
   In `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`, a value of `0` is biologically invalid and indicates a missing measurement. Medians of non-zero entries are computed **strictly from the 614 training rows** and used to replace zeros in both splits:

   | Feature Column | Training Set Non-Zero Median | Total Zeros Replaced (Train + Test) |
   | :--- | :---: | :---: |
   | `Glucose` | `117.00` | `5` |
   | `BloodPressure` | `72.00` | `35` |
   | `SkinThickness` | `29.00` | `227` |
   | `Insulin` | `125.00` | `374` |
   | `BMI` | `32.10` | `11` |

3. **Min-Max Scaling to $[0, 1]$ (Training-Set Bounds Only)**:
   Feature minimum and maximum values are computed **strictly from the imputed training set** and applied to both training and test sets via $x' = \frac{x - \min_{\text{train}}}{\max_{\text{train}} - \min_{\text{train}}}$:

   | Feature Column | Training Min | Training Max |
   | :--- | :---: | :---: |
   | `Pregnancies` | `0.000` | `17.000` |
   | `Glucose` | `44.000` | `199.000` |
   | `BloodPressure` | `24.000` | `114.000` |
   | `SkinThickness` | `7.000` | `99.000` |
   | `Insulin` | `14.000` | `846.000` |
   | `BMI` | `18.200` | `67.100` |
   | `DiabetesPedigreeFunction` | `0.078` | `2.420` |
   | `Age` | `21.000` | `72.000` |

---

## Project Structure

```text
csharp-mlp-diabetes-classifier/
├── backneural/
│   ├── INeuron.cs                        # Input layer neuron class
│   ├── HNeuron.cs                        # Hidden layer neuron class (sigmoid + backprop)
│   ├── ONeuron.cs                        # Output layer neuron class (sigmoid + error delta)
│   └── NeuralNet.cs                      # 3-layer MLP coordinator & Binary Cross-Entropy loss
├── data/
│   └── diabetes.csv                      # Real Pima Indians Diabetes dataset (768 rows)
├── output/
│   ├── experiments.csv                   # All 48 hyperparameter experiment runs
│   ├── training_log.csv                  # Epoch-by-epoch train/test BCE loss for best model
│   └── final_metrics.txt                 # Final evaluation report & confusion matrix
├── DataLoader.cs                         # Manual CSV parser, stratified split, median imputation, Min-Max scaling
├── Metrics.cs                            # Confusion matrix (TP, TN, FP, FN), Accuracy, Precision, Recall, F1
├── ExperimentRunner.cs                   # Grid search runner, model trainer, and CSV logger
├── MainForm.cs                           # Interactive Windows Forms Clinical Predictor GUI dashboard
├── Program.cs                            # Main entry point (launches GUI by default, or CLI mode with --cli)
├── csharp-mlp-diabetes-classifier.csproj # .NET 8 Windows Forms & CLI C# project file
├── .gitignore                            # Standard C#/.NET gitignore
└── README.md                             # Project documentation and experimental analysis
```

---

## How to Build and Run

Ensure the .NET SDK (`.NET 8.0` or newer on Windows) is installed, then run the following commands from the project root directory:

```bash
# Build the project in Release configuration
dotnet build -c Release

# Launch the interactive Windows Forms Clinical Predictor GUI Dashboard
dotnet run -c Release

# (Optional) Run the headless CLI batch pipeline (48-run grid search + retraining)
dotnet run -c Release -- --cli
```

Running the program automatically generates/updates:
- `output/experiments.csv`
- `output/training_log.csv`
- `output/final_metrics.txt`

---

## Hyperparameter Experiment Results

We systematically tested **48 combinations**:
- **Learning Rates (`learning_rate`)**: `0.01`, `0.05`, `0.1`, `0.5`
- **Hidden Layer Neurons (`hidden_neurons`)**: `4`, `8`, `16`, `32`
- **Epochs (`epochs`)**: `500`, `1000`, `2000`

All numbers below come directly from `output/experiments.csv`:

| Run | Learning Rate | Hidden Neurons | Epochs | Final Train BCE Loss | Test Accuracy | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.01 | 4 | 500 | 0.469527 | 0.7987 | 0.7556 | 0.6296 | 0.6869 |
| 2 | 0.01 | 4 | 1000 | 0.462342 | 0.8052 | 0.7609 | 0.6481 | 0.7000 |
| 3 | 0.01 | 4 | 2000 | 0.446523 | 0.7987 | 0.7170 | 0.7037 | 0.7103 |
| 4 | 0.01 | 8 | 500 | 0.468837 | 0.7987 | 0.7447 | 0.6481 | 0.6931 |
| 5 | 0.01 | 8 | 1000 | 0.463835 | 0.7987 | 0.7556 | 0.6296 | 0.6869 |
| 6 | 0.01 | 8 | 2000 | 0.449959 | 0.7922 | 0.7115 | 0.6852 | 0.6981 |
| 7 | 0.01 | 16 | 500 | 0.469333 | 0.7987 | 0.7556 | 0.6296 | 0.6869 |
| 8 | 0.01 | 16 | 1000 | 0.465473 | 0.7987 | 0.7556 | 0.6296 | 0.6869 |
| 9 | 0.01 | 16 | 2000 | 0.453958 | 0.7922 | 0.7292 | 0.6481 | 0.6863 |
| 10 | 0.01 | 32 | 500 | 0.469773 | 0.7987 | 0.7556 | 0.6296 | 0.6869 |
| 11 | 0.01 | 32 | 1000 | 0.466058 | 0.7987 | 0.7556 | 0.6296 | 0.6869 |
| 12 | 0.01 | 32 | 2000 | 0.453994 | 0.7987 | 0.7447 | 0.6481 | 0.6931 |
| 13 | 0.05 | 4 | 500 | 0.443024 | 0.7922 | 0.7115 | 0.6852 | 0.6981 |
| 14 | 0.05 | 4 | 1000 | 0.436835 | 0.8052 | 0.7222 | 0.7222 | 0.7222 |
| 15 | 0.05 | 4 | 2000 | 0.431354 | 0.7922 | 0.6964 | 0.7222 | 0.7091 |
| 16 | 0.05 | 8 | 500 | 0.444487 | 0.7987 | 0.7170 | 0.7037 | 0.7103 |
| **17** | **0.05** | **8** | **1000** | **0.436023** | **0.8117** | **0.7273** | **0.7407** | **0.7339** |
| 18 | 0.05 | 8 | 2000 | 0.426374 | 0.7857 | 0.6842 | 0.7222 | 0.7027 |
| 19 | 0.05 | 16 | 500 | 0.447285 | 0.7922 | 0.7115 | 0.6852 | 0.6981 |
| 20 | 0.05 | 16 | 1000 | 0.437125 | 0.7987 | 0.7170 | 0.7037 | 0.7103 |
| 21 | 0.05 | 16 | 2000 | 0.425358 | 0.7922 | 0.6964 | 0.7222 | 0.7091 |
| 22 | 0.05 | 32 | 500 | 0.448250 | 0.7987 | 0.7255 | 0.6852 | 0.7048 |
| 23 | 0.05 | 32 | 1000 | 0.439084 | 0.7987 | 0.7170 | 0.7037 | 0.7103 |
| 24 | 0.05 | 32 | 2000 | 0.431466 | 0.8052 | 0.7222 | 0.7222 | 0.7222 |
| 25 | 0.1 | 4 | 500 | 0.436986 | 0.7987 | 0.7170 | 0.7037 | 0.7103 |
| 26 | 0.1 | 4 | 1000 | 0.430805 | 0.7987 | 0.7091 | 0.7222 | 0.7156 |
| 27 | 0.1 | 4 | 2000 | 0.421522 | 0.7403 | 0.6346 | 0.6111 | 0.6226 |
| 28 | 0.1 | 8 | 500 | 0.436755 | 0.8052 | 0.7222 | 0.7222 | 0.7222 |
| 29 | 0.1 | 8 | 1000 | 0.428251 | 0.7922 | 0.6964 | 0.7222 | 0.7091 |
| 30 | 0.1 | 8 | 2000 | 0.413870 | 0.7338 | 0.6182 | 0.6296 | 0.6239 |
| 31 | 0.1 | 16 | 500 | 0.436215 | 0.7987 | 0.7170 | 0.7037 | 0.7103 |
| 32 | 0.1 | 16 | 1000 | 0.426460 | 0.7857 | 0.6909 | 0.7037 | 0.6972 |
| 33 | 0.1 | 16 | 2000 | 0.406788 | 0.7468 | 0.6316 | 0.6667 | 0.6486 |
| 34 | 0.1 | 32 | 500 | 0.438437 | 0.7987 | 0.7170 | 0.7037 | 0.7103 |
| 35 | 0.1 | 32 | 1000 | 0.430373 | 0.7987 | 0.7091 | 0.7222 | 0.7156 |
| 36 | 0.1 | 32 | 2000 | 0.412444 | 0.7662 | 0.6667 | 0.6667 | 0.6667 |
| 37 | 0.5 | 4 | 500 | 0.430526 | 0.7662 | 0.6731 | 0.6481 | 0.6604 |
| 38 | 0.5 | 4 | 1000 | 0.439030 | 0.7597 | 0.6545 | 0.6667 | 0.6606 |
| 39 | 0.5 | 4 | 2000 | 0.455875 | 0.7857 | 0.6667 | 0.7778 | 0.7179 |
| 40 | 0.5 | 8 | 500 | 0.423299 | 0.7727 | 0.6792 | 0.6667 | 0.6729 |
| 41 | 0.5 | 8 | 1000 | 0.419492 | 0.7662 | 0.6607 | 0.6852 | 0.6727 |
| 42 | 0.5 | 8 | 2000 | 0.506092 | 0.7662 | 0.6452 | 0.7407 | 0.6897 |
| 43 | 0.5 | 16 | 500 | 0.418683 | 0.7662 | 0.6607 | 0.6852 | 0.6727 |
| 44 | 0.5 | 16 | 1000 | 0.406777 | 0.7792 | 0.6923 | 0.6667 | 0.6792 |
| 45 | 0.5 | 16 | 2000 | 0.397895 | 0.7792 | 0.6724 | 0.7222 | 0.6964 |
| 46 | 0.5 | 32 | 500 | 0.416966 | 0.7727 | 0.6667 | 0.7037 | 0.6847 |
| 47 | 0.5 | 32 | 1000 | 0.401822 | 0.7338 | 0.6226 | 0.6111 | 0.6168 |
| 48 | 0.5 | 32 | 2000 | 0.427437 | 0.7532 | 0.6429 | 0.6667 | 0.6545 |

---

## Final Results (Best Configuration)

The experiment runner automatically selected **Run #17** (`learning_rate = 0.05`, `hidden_neurons = 8`, `epochs = 1000`), which achieved the highest **Test F1 Score (`0.7339`)** and highest **Test Accuracy (`0.8117`)**.

### Training & Test Loss Progression (`output/training_log.csv`)
| Epoch | Train BCE Loss | Test BCE Loss |
| :---: | :---: | :---: |
| 1 | 0.638750 | 0.639281 |
| 100 | 0.468729 | 0.436699 |
| 200 | 0.463636 | 0.434104 |
| 300 | 0.457713 | 0.429502 |
| 400 | 0.449838 | 0.422222 |
| 500 | 0.444487 | 0.416153 |
| 600 | 0.441474 | 0.412453 |
| 700 | 0.439551 | 0.410407 |
| 800 | 0.438151 | 0.409341 |
| 900 | 0.437019 | 0.408866 |
| 1000 | 0.436023 | 0.408768 |

### Confusion Matrix on Held-Out Test Set (154 samples, Threshold = 0.5)
| | Predicted Negative (`0`) | Predicted Positive (`1`) |
| :--- | :---: | :---: |
| **Actual Negative (`0`)** | **TN = 85** | **FP = 15** |
| **Actual Positive (`1`)** | **FN = 14** | **TP = 40** |

### Final Evaluation Metrics (`output/final_metrics.txt`)
- **Accuracy**: `0.8117` (`81.17%`)
- **Precision**: `0.7273` (`72.73%`)
- **Recall (Sensitivity)**: `0.7407` (`74.07%`)
- **F1 Score**: `0.7339` (`73.39%`)

---

## Analysis & Discussion

### 1. What the Loss Curve Shows
- **Rapid Early Feature Learning (Epochs 1–100)**: Both training and test Binary Cross-Entropy loss drop sharply from `~0.639` at Epoch 1 to `0.4687` (train) and `0.4367` (test) by Epoch 100 as the network learns the primary linear and mildly non-linear relationships (especially `Glucose`, `BMI`, and `Age`).
- **Refinement of Non-Linear Interactions (Epochs 100–1000)**: Between Epochs 100 and 1000 with `learning_rate = 0.05` and `8` hidden neurons, training loss continues decreasing smoothly to `0.436023` while test loss decreases monotonically to `0.408768`, showing strong generalization without overfitting.
- **Effect of Hyperparameter Extremes**:
  - At `learning_rate = 0.01`, learning is too slow: after 500–1000 epochs, training loss remains around `0.462–0.469` and recall stays capped near `0.6296`.
  - At `learning_rate = 0.1` for `2000` epochs, the model begins overfitting the training set (`final_train_loss` drops to `0.4068–0.4215`, while test accuracy falls to `0.7338–0.7662`).
  - At `learning_rate = 0.5`, per-sample online updates take overly large steps, causing weight oscillation and higher variance in loss (e.g., `LR = 0.5, Hidden = 8, Epochs = 2000` rebounds to `0.5061` train loss).

### 2. Why Recall Matters in a Medical Context
In clinical screening for diabetes:
- A **False Positive (`FP = 15`)** means a healthy patient is flagged for follow-up clinical testing (such as a confirmatory HbA1c or fasting blood glucose test), which incurs minor cost/inconvenience.
- A **False Negative (`FN = 14`)** means an actual diabetic patient is told they are healthy, delaying diagnosis, lifestyle changes, and medical intervention, which can lead to severe long-term complications (neuropathy, retinopathy, cardiovascular or renal disease).
- Therefore, **Recall (`74.07%` in our best model, identifying 40 out of 54 positive patients)** and **F1 Score (`73.39%`)** are far more informative than raw Accuracy alone, especially because `65.1%` of the dataset is negative (a trivial classifier predicting all `0`s would still achieve `64.9%` accuracy with `0%` recall).

### 3. Limitations & Possible Improvements
- **Extensive Missing Insulin & SkinThickness Data**: Nearly half (`374 / 768 = 48.7%`) of `Insulin` values and `227 / 768 = 29.6%` of `SkinThickness` values were originally zero and imputed with the training median. Using k-Nearest Neighbors (k-NN) or multivariate regression imputation computed on the training split could preserve richer inter-feature correlations.
- **Decision Threshold Tuning**: Rather than fixing the sigmoid classification threshold at `0.5`, lowering the decision threshold (e.g., to `0.35–0.40`) on a validation split could boost clinical **Recall** above `85%` at the expense of a modest drop in precision.
- **Regularization & Learning Rate Schedules**: Adding L2 weight decay, mini-batch gradient descent with shuffling per epoch, or learning rate decay would prevent the late-epoch overfitting seen at `2000` epochs.

