using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using backneural;

namespace CsharpMlpDiabetesClassifier
{
    /// <summary>
    /// Stores a single epoch's training and testing metrics for chart rendering and log export.
    /// </summary>
    public class EpochSnapshot
    {
        /// <summary>Gets or sets the epoch number.</summary>
        public int Epoch { get; set; }

        /// <summary>Gets or sets the training Binary Cross-Entropy loss.</summary>
        public double TrainBce { get; set; }

        /// <summary>Gets or sets the testing Binary Cross-Entropy loss.</summary>
        public double TestBce { get; set; }

        /// <summary>Gets or sets the training Mean Squared Error.</summary>
        public double TrainMse { get; set; }

        /// <summary>Gets or sets the testing Mean Squared Error.</summary>
        public double TestMse { get; set; }

        /// <summary>Gets or sets the training classification accuracy.</summary>
        public double TrainAcc { get; set; }

        /// <summary>Gets or sets the testing classification accuracy.</summary>
        public double TestAcc { get; set; }
    }

    /// <summary>
    /// Double-buffered custom canvas for rendering loss curves and probability gauges without flicker.
    /// </summary>
    public class BufferedCanvas : Panel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BufferedCanvas"/> class.
        /// </summary>
        public BufferedCanvas()
        {
            this.DoubleBuffered = true;
            this.ResizeRedraw = true;
            this.BackColor = Color.White;
        }
    }

    /// <summary>
    /// Custom clinical analytics dashboard for the Pima Indians Diabetes 3-Layer MLP classifier.
    /// Features a top KPI header, horizontal pipeline toolbar, split visual analytics dashboard,
    /// interactive patient diagnostic simulator with sigmoid probability gauge, and grid search explorer.
    /// </summary>
    public class MainForm : Form
    {
        // Theme Colors (Modern Teal & Slate Clinical Palette)
        private static readonly Color HeaderDarkBg = Color.FromArgb(15, 23, 42);
        private static readonly Color AccentTeal = Color.FromArgb(13, 148, 136);
        private static readonly Color AccentBlue = Color.FromArgb(37, 99, 235);
        private static readonly Color AccentAmber = Color.FromArgb(217, 119, 6);
        private static readonly Color SurfaceLight = Color.FromArgb(241, 245, 249);
        private static readonly Color CardBorder = Color.FromArgb(203, 213, 225);

        // Application State
        private string currentCsvPath = Path.Combine("data", "diabetes.csv");
        private DatasetSplit? currentSplit;
        private NeuralNet? currentNet;
        private int totalEpochsTrained = 0;
        private readonly List<EpochSnapshot> lossHistory = new List<EpochSnapshot>();
        private CancellationTokenSource? trainCts;
        private bool showMseCurve = false;
        private double lastPatientProbability = 0.0;

        // Top KPI Labels
        private Label lblKpiDataset = null!;
        private Label lblKpiArchitecture = null!;
        private Label lblKpiTestAccuracy = null!;
        private Label lblKpiTestF1 = null!;

        // Top Toolbar Hyperparameter Controls
        private NumericUpDown numTrainSplit = null!;
        private NumericUpDown numSeed = null!;
        private NumericUpDown numHidden = null!;
        private NumericUpDown numLearningRate = null!;
        private NumericUpDown numEpochs = null!;
        private NumericUpDown numLogStep = null!;
        private CheckBox chkStratified = null!;
        private CheckBox chkImpute = null!;

        // Top Toolbar Action Buttons
        private Button btnLoadCsv = null!;
        private Button btnResetModel = null!;
        private Button btnTrainModel = null!;
        private Button btnStopTraining = null!;
        private Button btnExportCsv = null!;

        // Main Workspace Tabs
        private TabControl mainTabs = null!;

        // Tab 1: Dashboard & Evaluation (Chart + Confusion Matrix + Metrics Bars)
        private BufferedCanvas canvasLossCurve = null!;
        private CheckBox chkToggleMse = null!;
        private Label lblCmTn = null!;
        private Label lblCmFp = null!;
        private Label lblCmFn = null!;
        private Label lblCmTp = null!;
        private Label lblMetricAccVal = null!;
        private Label lblMetricPrecVal = null!;
        private Label lblMetricRecVal = null!;
        private Label lblMetricF1Val = null!;
        private Panel barAccFill = null!;
        private Panel barPrecFill = null!;
        private Panel barRecFill = null!;
        private Panel barF1Fill = null!;
        private DataGridView dgvSummaryMetrics = null!;

        // Tab 2: Hyperparameter Experiments (48 Runs)
        private TextBox txtGridLrs = null!;
        private TextBox txtGridHiddens = null!;
        private TextBox txtGridEpochs = null!;
        private Button btnRunExperiments = null!;
        private Button btnApplySelectedConfig = null!;
        private DataGridView dgvExperiments = null!;

        // Tab 3: Dataset & Imputation Explorer
        private RadioButton rbViewRaw = null!;
        private RadioButton rbViewImputed = null!;
        private RadioButton rbViewNormalized = null!;
        private DataGridView dgvDataset = null!;
        private DataGridView dgvImputationStats = null!;

        // Tab 4: Training Console Log
        private TextBox txtConsoleLog = null!;

        // Right Sidebar: Patient Diagnostic Simulator
        private NumericUpDown numPreg = null!;
        private NumericUpDown numGluc = null!;
        private NumericUpDown numBp = null!;
        private NumericUpDown numSkin = null!;
        private NumericUpDown numIns = null!;
        private NumericUpDown numBmi = null!;
        private NumericUpDown numPed = null!;
        private NumericUpDown numAge = null!;
        private BufferedCanvas canvasSigmoidGauge = null!;
        private Label lblDiagnosisVerdict = null!;
        private Label lblDiagnosisDetail = null!;

        // Bottom Status Bar
        private StatusStrip statusStrip = null!;
        private ToolStripStatusLabel lblStatusText = null!;
        private ToolStripProgressBar statusProgress = null!;

        /// <summary>
        /// Initializes a new instance of the <see cref="MainForm"/> class.
        /// </summary>
        public MainForm()
        {
            BuildUi();
            this.Shown += OnMainFormShown;
        }

        private void BuildUi()
        {
            this.Text = "Pima Diabetes Clinical Predictor — Custom C# Multi-Layer Perceptron (backneural)";
            this.Size = new Size(1320, 850);
            this.MinimumSize = new Size(1180, 780);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.Font = new Font("Segoe UI", 9F, FontStyle.Regular);
            this.BackColor = SurfaceLight;

            // 1. Bottom Status Strip
            statusStrip = new StatusStrip { BackColor = Color.White, SizingGrip = true };
            lblStatusText = new ToolStripStatusLabel("Ready.")
            {
                Spring = true,
                TextAlign = ContentAlignment.MiddleLeft,
                ForeColor = Color.FromArgb(51, 65, 85)
            };
            statusProgress = new ToolStripProgressBar
            {
                Size = new Size(200, 16),
                Style = ProgressBarStyle.Continuous
            };
            statusStrip.Items.Add(lblStatusText);
            statusStrip.Items.Add(statusProgress);

            // 2. Top Banner with Live KPI Badges
            Panel topBanner = BuildTopBanner();

            // 3. Horizontal Control & Hyperparameter Ribbon
            Panel controlRibbon = BuildControlRibbon();

            // 4. Right-Side Patient Simulator Drawer
            Panel patientDrawer = BuildPatientSimulatorDrawer();

            // 5. Center Tabbed Analytics Workspace
            Panel centerWorkspace = BuildCenterWorkspace();

            // Add in docking order
            this.Controls.Add(centerWorkspace);
            this.Controls.Add(patientDrawer);
            this.Controls.Add(controlRibbon);
            this.Controls.Add(topBanner);
            this.Controls.Add(statusStrip);
        }

        private Panel BuildTopBanner()
        {
            Panel banner = new Panel
            {
                Dock = DockStyle.Top,
                Height = 68,
                BackColor = HeaderDarkBg,
                Padding = new Padding(16, 10, 16, 10)
            };

            Label lblTitle = new Label
            {
                Text = "PIMA DIABETES CLINICAL PREDICTOR",
                Location = new Point(18, 10),
                AutoSize = true,
                Font = new Font("Segoe UI Semibold", 13.5F, FontStyle.Bold),
                ForeColor = Color.White
            };

            Label lblSub = new Label
            {
                Text = "Custom C# 3-Layer Backpropagation MLP (INeuron · HNeuron · ONeuron · NeuralNet) — Zero External ML Libraries",
                Location = new Point(20, 38),
                AutoSize = true,
                Font = new Font("Segoe UI", 8.75F, FontStyle.Regular),
                ForeColor = Color.FromArgb(148, 163, 184)
            };

            // Right-aligned KPI pills inside banner
            FlowLayoutPanel kpiFlow = new FlowLayoutPanel
            {
                Dock = DockStyle.Right,
                AutoSize = true,
                FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false,
                BackColor = Color.Transparent,
                Padding = new Padding(0, 2, 0, 0)
            };

            lblKpiDataset = CreateKpiBadge(kpiFlow, "DATASET", "768 Rows (80/20)", Color.FromArgb(30, 41, 59));
            lblKpiArchitecture = CreateKpiBadge(kpiFlow, "MLP TOPOLOGY", "8 → 8 → 1 (LR 0.05)", Color.FromArgb(30, 41, 59));
            lblKpiTestAccuracy = CreateKpiBadge(kpiFlow, "TEST ACCURACY", "81.17%", Color.FromArgb(15, 118, 110));
            lblKpiTestF1 = CreateKpiBadge(kpiFlow, "TEST F1 / RECALL", "0.7339 / 74.1%", Color.FromArgb(29, 78, 216));

            banner.Controls.Add(kpiFlow);
            banner.Controls.Add(lblTitle);
            banner.Controls.Add(lblSub);
            return banner;
        }

        private static Label CreateKpiBadge(Control parent, string caption, string initialValue, Color bgColor)
        {
            Panel pill = new Panel
            {
                Size = new Size(148, 46),
                BackColor = bgColor,
                Margin = new Padding(8, 0, 0, 0)
            };

            Label lblCap = new Label
            {
                Text = caption,
                Location = new Point(8, 5),
                AutoSize = true,
                Font = new Font("Segoe UI", 7F, FontStyle.Bold),
                ForeColor = Color.FromArgb(148, 163, 184)
            };

            Label lblVal = new Label
            {
                Text = initialValue,
                Location = new Point(8, 21),
                AutoSize = true,
                Font = new Font("Segoe UI Semibold", 10F, FontStyle.Bold),
                ForeColor = Color.White
            };

            pill.Controls.Add(lblCap);
            pill.Controls.Add(lblVal);
            parent.Controls.Add(pill);
            return lblVal;
        }

        private Panel BuildControlRibbon()
        {
            Panel ribbon = new Panel
            {
                Dock = DockStyle.Top,
                Height = 84,
                BackColor = Color.White,
                Padding = new Padding(14, 8, 14, 8)
            };
            ribbon.Paint += (s, e) =>
            {
                using Pen p = new Pen(CardBorder, 1F);
                e.Graphics.DrawLine(p, 0, ribbon.Height - 1, ribbon.Width, ribbon.Height - 1);
            };

            // Row 1: Hyperparameter inputs
            int y1 = 10;
            AddInlineLabel(ribbon, "Train %:", 14, y1 + 3);
            numTrainSplit = new NumericUpDown { Location = new Point(68, y1), Size = new Size(54, 23), Minimum = 50, Maximum = 95, Value = 80 };

            AddInlineLabel(ribbon, "Seed:", 132, y1 + 3);
            numSeed = new NumericUpDown { Location = new Point(170, y1), Size = new Size(64, 23), Minimum = 0, Maximum = 99999, Value = 42 };

            chkStratified = new CheckBox { Text = "Stratified 80/20", Location = new Point(246, y1 + 2), AutoSize = true, Checked = true };
            chkImpute = new CheckBox { Text = "Training-Median Zero Imputation", Location = new Point(366, y1 + 2), AutoSize = true, Checked = true };

            AddInlineLabel(ribbon, "Hidden Neurons:", 590, y1 + 3);
            numHidden = new NumericUpDown { Location = new Point(692, y1), Size = new Size(58, 23), Minimum = 1, Maximum = 128, Value = 8 };

            AddInlineLabel(ribbon, "Learning Rate:", 764, y1 + 3);
            numLearningRate = new NumericUpDown
            {
                Location = new Point(854, y1),
                Size = new Size(68, 23),
                DecimalPlaces = 3,
                Increment = 0.01M,
                Minimum = 0.001M,
                Maximum = 2.0M,
                Value = 0.050M
            };

            AddInlineLabel(ribbon, "Epochs:", 936, y1 + 3);
            numEpochs = new NumericUpDown { Location = new Point(988, y1), Size = new Size(72, 23), Minimum = 10, Maximum = 20000, Increment = 100, Value = 1000 };

            AddInlineLabel(ribbon, "Log Step:", 1072, y1 + 3);
            numLogStep = new NumericUpDown { Location = new Point(1134, y1), Size = new Size(62, 23), Minimum = 10, Maximum = 2000, Increment = 50, Value = 100 };

            // Row 2: Pipeline Action Buttons
            int y2 = 42;
            btnLoadCsv = CreateActionButton("1. Load & Preprocess CSV", new Point(14, y2), new Size(182, 32), Color.FromArgb(51, 65, 85), Color.White);
            btnLoadCsv.Click += OnLoadAndPreprocessClick;

            btnResetModel = CreateActionButton("2. Initialize MLP Weights", new Point(206, y2), new Size(176, 32), AccentBlue, Color.White);
            btnResetModel.Click += (s, e) => InitializeNewMlp();

            btnTrainModel = CreateActionButton("3. Train Network (+Epochs)", new Point(392, y2), new Size(190, 32), AccentTeal, Color.White);
            btnTrainModel.Click += async (s, e) => await TrainCurrentModelAsync((int)numEpochs.Value, (int)numLogStep.Value);

            btnStopTraining = CreateActionButton("Stop", new Point(590, y2), new Size(76, 32), Color.FromArgb(220, 38, 38), Color.White);
            btnStopTraining.Enabled = false;
            btnStopTraining.Click += (s, e) => trainCts?.Cancel();

            Button btnBrowseCsv = CreateActionButton("Browse CSV File...", new Point(676, y2), new Size(136, 32), Color.FromArgb(226, 232, 240), Color.FromArgb(30, 41, 59));
            btnBrowseCsv.Click += OnBrowseCsvClick;

            btnExportCsv = CreateActionButton("Export Loss Log CSV...", new Point(820, y2), new Size(160, 32), Color.FromArgb(226, 232, 240), Color.FromArgb(30, 41, 59));
            btnExportCsv.Click += OnExportLossCsvClick;

            Button btnJumpGrid = CreateActionButton("Open 48-Run Grid Search →", new Point(988, y2), new Size(208, 32), Color.FromArgb(254, 243, 199), Color.FromArgb(146, 64, 14));
            btnJumpGrid.Click += (s, e) => mainTabs.SelectedIndex = 1;

            ribbon.Controls.AddRange(new Control[]
            {
                numTrainSplit, numSeed, chkStratified, chkImpute,
                numHidden, numLearningRate, numEpochs, numLogStep,
                btnLoadCsv, btnResetModel, btnTrainModel, btnStopTraining,
                btnBrowseCsv, btnExportCsv, btnJumpGrid
            });

            return ribbon;
        }

        private static void AddInlineLabel(Control parent, string text, int x, int y)
        {
            Label lbl = new Label
            {
                Text = text,
                Location = new Point(x, y),
                AutoSize = true,
                ForeColor = Color.FromArgb(51, 65, 85),
                Font = new Font("Segoe UI", 8.75F, FontStyle.Regular)
            };
            parent.Controls.Add(lbl);
        }

        private static Button CreateActionButton(string text, Point location, Size size, Color backColor, Color foreColor)
        {
            Button btn = new Button
            {
                Text = text,
                Location = location,
                Size = size,
                FlatStyle = FlatStyle.Flat,
                BackColor = backColor,
                ForeColor = foreColor,
                Font = new Font("Segoe UI Semibold", 8.75F, FontStyle.Bold),
                Cursor = Cursors.Hand
            };
            btn.FlatAppearance.BorderSize = 0;
            return btn;
        }

        private Panel BuildPatientSimulatorDrawer()
        {
            Panel drawer = new Panel
            {
                Dock = DockStyle.Right,
                Width = 320,
                Padding = new Padding(8, 10, 12, 10),
                BackColor = SurfaceLight
            };

            Panel card = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.White,
                Padding = new Padding(14)
            };
            card.Paint += DrawCardBorder;

            Label lblHeader = new Label
            {
                Text = "PATIENT DIAGNOSTIC SIMULATOR",
                Location = new Point(14, 14),
                AutoSize = true,
                Font = new Font("Segoe UI Semibold", 10F, FontStyle.Bold),
                ForeColor = HeaderDarkBg
            };

            Label lblSub = new Label
            {
                Text = "Enter raw clinical biomarkers (auto-scaled via training min/max):",
                Location = new Point(14, 36),
                Size = new Size(276, 30),
                Font = new Font("Segoe UI", 8.25F),
                ForeColor = Color.FromArgb(100, 116, 139)
            };

            // Preset sample buttons
            Button btnPresetDiabetic = CreateActionButton("Load High-Risk Case", new Point(14, 68), new Size(134, 26), Color.FromArgb(254, 226, 226), Color.FromArgb(153, 27, 27));
            btnPresetDiabetic.Click += (s, e) =>
            {
                numPreg.Value = 6; numGluc.Value = 148; numBp.Value = 72; numSkin.Value = 35;
                numIns.Value = 168; numBmi.Value = 33.6M; numPed.Value = 0.627M; numAge.Value = 50;
                EvaluateSinglePatient();
            };

            Button btnPresetHealthy = CreateActionButton("Load Healthy Case", new Point(154, 68), new Size(134, 26), Color.FromArgb(209, 250, 229), Color.FromArgb(6, 95, 70));
            btnPresetHealthy.Click += (s, e) =>
            {
                numPreg.Value = 1; numGluc.Value = 89; numBp.Value = 66; numSkin.Value = 23;
                numIns.Value = 94; numBmi.Value = 28.1M; numPed.Value = 0.167M; numAge.Value = 21;
                EvaluateSinglePatient();
            };

            int startY = 106;
            int rowH = 33;
            numPreg = AddBiomarkerInput(card, "Pregnancies (count)", 14, startY + rowH * 0, 0, 25, 6, 0, 1M);
            numGluc = AddBiomarkerInput(card, "Plasma Glucose (mg/dL)", 14, startY + rowH * 1, 0, 300, 148, 0, 1M);
            numBp = AddBiomarkerInput(card, "Diastolic BP (mm Hg)", 14, startY + rowH * 2, 0, 200, 72, 0, 1M);
            numSkin = AddBiomarkerInput(card, "Skin Thickness (mm)", 14, startY + rowH * 3, 0, 120, 35, 0, 1M);
            numIns = AddBiomarkerInput(card, "2-Hr Serum Insulin (uU/ml)", 14, startY + rowH * 4, 0, 900, 168, 0, 5M);
            numBmi = AddBiomarkerInput(card, "Body Mass Index (kg/m²)", 14, startY + rowH * 5, 0, 80, 33.6M, 1, 0.5M);
            numPed = AddBiomarkerInput(card, "Diabetes Pedigree Func", 14, startY + rowH * 6, 0, 3, 0.627M, 3, 0.02M);
            numAge = AddBiomarkerInput(card, "Patient Age (years)", 14, startY + rowH * 7, 18, 100, 50, 0, 1M);

            Button btnPredictNow = CreateActionButton("Run Clinical Prediction", new Point(14, startY + rowH * 8 + 6), new Size(274, 34), AccentTeal, Color.White);
            btnPredictNow.Click += (s, e) => EvaluateSinglePatient();

            // Sigmoid Gauge Canvas
            canvasSigmoidGauge = new BufferedCanvas
            {
                Location = new Point(14, startY + rowH * 8 + 48),
                Size = new Size(274, 72),
                BackColor = Color.FromArgb(248, 250, 252)
            };
            canvasSigmoidGauge.Paint += OnPaintSigmoidGauge;

            // Verdict Banner
            Panel verdictBox = new Panel
            {
                Location = new Point(14, startY + rowH * 8 + 128),
                Size = new Size(274, 68),
                BackColor = Color.FromArgb(254, 242, 242),
                Padding = new Padding(8)
            };
            verdictBox.Paint += DrawCardBorder;

            lblDiagnosisVerdict = new Label
            {
                Text = "DIABETIC RISK DETECTED (1)",
                Dock = DockStyle.Top,
                Height = 28,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 10F, FontStyle.Bold),
                ForeColor = Color.FromArgb(185, 28, 28)
            };

            lblDiagnosisDetail = new Label
            {
                Text = "Sigmoid Output: 0.784  (Threshold >= 0.50)",
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 8.5F, FontStyle.Regular),
                ForeColor = Color.FromArgb(71, 85, 105)
            };

            verdictBox.Controls.Add(lblDiagnosisDetail);
            verdictBox.Controls.Add(lblDiagnosisVerdict);

            card.Controls.AddRange(new Control[]
            {
                lblHeader, lblSub, btnPresetDiabetic, btnPresetHealthy,
                btnPredictNow, canvasSigmoidGauge, verdictBox
            });

            drawer.Controls.Add(card);
            return drawer;
        }

        private NumericUpDown AddBiomarkerInput(
            Control parent,
            string labelText,
            int x,
            int y,
            decimal min,
            decimal max,
            decimal val,
            int decimals,
            decimal step)
        {
            Label lbl = new Label
            {
                Text = labelText,
                Location = new Point(x, y + 3),
                Size = new Size(184, 20),
                ForeColor = Color.FromArgb(51, 65, 85)
            };

            NumericUpDown num = new NumericUpDown
            {
                Location = new Point(x + 190, y),
                Size = new Size(84, 23),
                Minimum = min,
                Maximum = max,
                DecimalPlaces = decimals,
                Increment = step,
                Value = val
            };
            num.ValueChanged += (s, e) => EvaluateSinglePatient();

            parent.Controls.Add(lbl);
            parent.Controls.Add(num);
            return num;
        }

        private Panel BuildCenterWorkspace()
        {
            Panel wrapper = new Panel
            {
                Dock = DockStyle.Fill,
                Padding = new Padding(12, 10, 4, 10),
                BackColor = SurfaceLight
            };

            mainTabs = new TabControl
            {
                Dock = DockStyle.Fill,
                Font = new Font("Segoe UI Semibold", 9.25F, FontStyle.Regular)
            };

            TabPage tabDashboard = new TabPage("1. Visual Analytics & Evaluation") { BackColor = SurfaceLight, Padding = new Padding(6) };
            TabPage tabExperiments = new TabPage("2. Hyperparameter Grid Search (48 Runs)") { BackColor = Color.White, Padding = new Padding(10) };
            TabPage tabDataset = new TabPage("3. Dataset & Imputation Inspector") { BackColor = Color.White, Padding = new Padding(10) };
            TabPage tabConsole = new TabPage("4. Training & Backprop Log") { BackColor = Color.FromArgb(15, 23, 42), Padding = new Padding(6) };

            BuildDashboardTab(tabDashboard);
            BuildExperimentsTab(tabExperiments);
            BuildDatasetTab(tabDataset);
            BuildConsoleTab(tabConsole);

            mainTabs.TabPages.AddRange(new[] { tabDashboard, tabExperiments, tabDataset, tabConsole });
            wrapper.Controls.Add(mainTabs);
            return wrapper;
        }

        private void BuildDashboardTab(TabPage page)
        {
            TableLayoutPanel splitGrid = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 1,
                BackColor = SurfaceLight
            };
            splitGrid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 56F));
            splitGrid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 44F));

            // Left Card: Live Loss Curve
            Panel chartCard = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.White,
                Margin = new Padding(0, 0, 6, 0),
                Padding = new Padding(10)
            };
            chartCard.Paint += DrawCardBorder;

            chkToggleMse = new CheckBox
            {
                Text = "Plot Mean Squared Error (MSE) instead of Binary Cross-Entropy (BCE)",
                AutoSize = true,
                Location = new Point(16, 12),
                BackColor = Color.White,
                Font = new Font("Segoe UI", 8.5F)
            };
            chkToggleMse.CheckedChanged += (s, e) =>
            {
                showMseCurve = chkToggleMse.Checked;
                canvasLossCurve.Invalidate();
            };

            canvasLossCurve = new BufferedCanvas
            {
                Dock = DockStyle.Fill
            };
            canvasLossCurve.Paint += OnPaintLossChart;

            chartCard.Controls.Add(chkToggleMse);
            chartCard.Controls.Add(canvasLossCurve);

            // Right Card: Confusion Matrix Tiles + Metric Bars + Per-Class Table
            Panel evalCard = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.White,
                Margin = new Padding(6, 0, 0, 0),
                Padding = new Padding(14)
            };
            evalCard.Paint += DrawCardBorder;

            Label lblEvalTitle = new Label
            {
                Text = "HELD-OUT TEST SET EVALUATION (154 SAMPLES)",
                Location = new Point(14, 12),
                AutoSize = true,
                Font = new Font("Segoe UI Semibold", 10F, FontStyle.Bold),
                ForeColor = HeaderDarkBg
            };

            // 4 Metric Progress Bars (Accuracy, Precision, Recall, F1)
            int mY = 40;
            (lblMetricAccVal, barAccFill) = CreateMetricBarRow(evalCard, "Accuracy", 14, mY + 0, AccentTeal);
            (lblMetricPrecVal, barPrecFill) = CreateMetricBarRow(evalCard, "Precision", 14, mY + 30, AccentBlue);
            (lblMetricRecVal, barRecFill) = CreateMetricBarRow(evalCard, "Recall (Sensitivity)", 14, mY + 60, AccentAmber);
            (lblMetricF1Val, barF1Fill) = CreateMetricBarRow(evalCard, "F1 Score", 14, mY + 90, Color.FromArgb(124, 58, 237));

            // 2x2 Visual Confusion Matrix Tiles
            Label lblCmHeader = new Label
            {
                Text = "CONFUSION MATRIX (THRESHOLD = 0.50)",
                Location = new Point(14, 174),
                AutoSize = true,
                Font = new Font("Segoe UI Semibold", 9F, FontStyle.Bold),
                ForeColor = Color.FromArgb(71, 85, 105)
            };

            TableLayoutPanel cmLayout = new TableLayoutPanel
            {
                Location = new Point(14, 198),
                Size = new Size(375, 128),
                ColumnCount = 2,
                RowCount = 2
            };
            cmLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            cmLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            cmLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 50F));
            cmLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 50F));

            lblCmTn = CreateConfusionTile(cmLayout, 0, 0, "TRUE NEGATIVE (TN)", "85", "Actual 0 · Predicted 0", Color.FromArgb(209, 250, 229), Color.FromArgb(6, 95, 70));
            lblCmFp = CreateConfusionTile(cmLayout, 1, 0, "FALSE POSITIVE (FP)", "15", "Actual 0 · Predicted 1", Color.FromArgb(254, 226, 226), Color.FromArgb(153, 27, 27));
            lblCmFn = CreateConfusionTile(cmLayout, 0, 1, "FALSE NEGATIVE (FN)", "14", "Actual 1 · Predicted 0", Color.FromArgb(254, 226, 226), Color.FromArgb(153, 27, 27));
            lblCmTp = CreateConfusionTile(cmLayout, 1, 1, "TRUE POSITIVE (TP)", "40", "Actual 1 · Predicted 1", Color.FromArgb(209, 250, 229), Color.FromArgb(6, 95, 70));

            Label lblClassTableTitle = new Label
            {
                Text = "PER-CLASS BREAKDOWN",
                Location = new Point(14, 336),
                AutoSize = true,
                Font = new Font("Segoe UI Semibold", 9F, FontStyle.Bold),
                ForeColor = Color.FromArgb(71, 85, 105)
            };

            dgvSummaryMetrics = CreateCleanGrid();
            dgvSummaryMetrics.Location = new Point(14, 358);
            dgvSummaryMetrics.Size = new Size(375, 118);
            dgvSummaryMetrics.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Bottom;
            dgvSummaryMetrics.Columns.Add("Class", "Class");
            dgvSummaryMetrics.Columns.Add("Prec", "Precision");
            dgvSummaryMetrics.Columns.Add("Rec", "Recall");
            dgvSummaryMetrics.Columns.Add("F1", "F1");
            dgvSummaryMetrics.Columns.Add("N", "Samples");

            evalCard.Controls.AddRange(new Control[]
            {
                lblEvalTitle, lblCmHeader, cmLayout, lblClassTableTitle, dgvSummaryMetrics
            });

            splitGrid.Controls.Add(chartCard, 0, 0);
            splitGrid.Controls.Add(evalCard, 1, 0);
            page.Controls.Add(splitGrid);
        }

        private static (Label ValueLabel, Panel FillBar) CreateMetricBarRow(Control parent, string name, int x, int y, Color fillColor)
        {
            Label lblName = new Label
            {
                Text = name,
                Location = new Point(x, y),
                Size = new Size(125, 20),
                Font = new Font("Segoe UI", 8.75F, FontStyle.Regular),
                ForeColor = Color.FromArgb(51, 65, 85)
            };

            Panel track = new Panel
            {
                Location = new Point(x + 128, y + 4),
                Size = new Size(175, 12),
                BackColor = Color.FromArgb(226, 232, 240)
            };

            Panel fill = new Panel
            {
                Location = new Point(0, 0),
                Size = new Size(140, 12),
                BackColor = fillColor
            };
            track.Controls.Add(fill);

            Label lblVal = new Label
            {
                Text = "0.00%",
                Location = new Point(x + 310, y),
                Size = new Size(68, 20),
                TextAlign = ContentAlignment.MiddleRight,
                Font = new Font("Segoe UI Semibold", 8.75F, FontStyle.Bold),
                ForeColor = HeaderDarkBg
            };

            parent.Controls.Add(lblName);
            parent.Controls.Add(track);
            parent.Controls.Add(lblVal);
            return (lblVal, fill);
        }

        private static Label CreateConfusionTile(
            TableLayoutPanel parent,
            int col,
            int row,
            string title,
            string countText,
            string subtitle,
            Color bg,
            Color fg)
        {
            Panel tile = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = bg,
                Margin = new Padding(3),
                Padding = new Padding(6, 4, 6, 4)
            };

            Label lblT = new Label
            {
                Text = title,
                Dock = DockStyle.Top,
                Height = 15,
                Font = new Font("Segoe UI", 7.25F, FontStyle.Bold),
                ForeColor = fg
            };

            Label lblVal = new Label
            {
                Text = countText,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 14F, FontStyle.Bold),
                ForeColor = fg
            };

            Label lblSub = new Label
            {
                Text = subtitle,
                Dock = DockStyle.Bottom,
                Height = 14,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 7.25F, FontStyle.Regular),
                ForeColor = fg
            };

            tile.Controls.Add(lblVal);
            tile.Controls.Add(lblT);
            tile.Controls.Add(lblSub);
            parent.Controls.Add(tile, col, row);
            return lblVal;
        }

        private void BuildExperimentsTab(TabPage page)
        {
            Panel topPanel = new Panel
            {
                Dock = DockStyle.Top,
                Height = 52,
                BackColor = Color.White
            };

            AddInlineLabel(topPanel, "Learning Rates:", 4, 14);
            txtGridLrs = new TextBox { Text = "0.01, 0.05, 0.1, 0.5", Location = new Point(98, 11), Size = new Size(125, 23) };

            AddInlineLabel(topPanel, "Hidden Neurons:", 234, 14);
            txtGridHiddens = new TextBox { Text = "4, 8, 16, 32", Location = new Point(334, 11), Size = new Size(95, 23) };

            AddInlineLabel(topPanel, "Epochs:", 440, 14);
            txtGridEpochs = new TextBox { Text = "500, 1000, 2000", Location = new Point(492, 11), Size = new Size(115, 23) };

            btnRunExperiments = CreateActionButton("Run Full Grid Search", new Point(620, 8), new Size(156, 32), AccentTeal, Color.White);
            btnRunExperiments.Click += OnRunGridSearchClick;

            btnApplySelectedConfig = CreateActionButton("Train Selected Row →", new Point(784, 8), new Size(156, 32), AccentBlue, Color.White);
            btnApplySelectedConfig.Click += OnApplySelectedGridRowClick;

            topPanel.Controls.AddRange(new Control[]
            {
                txtGridLrs, txtGridHiddens, txtGridEpochs, btnRunExperiments, btnApplySelectedConfig
            });

            dgvExperiments = CreateCleanGrid();
            dgvExperiments.Dock = DockStyle.Fill;
            dgvExperiments.Columns.Add("Rank", "#");
            dgvExperiments.Columns[0].FillWeight = 45;
            dgvExperiments.Columns.Add("LR", "Learning Rate");
            dgvExperiments.Columns.Add("Hidden", "Hidden Neurons");
            dgvExperiments.Columns.Add("Epochs", "Epochs");
            dgvExperiments.Columns.Add("TrainLoss", "Final Train BCE");
            dgvExperiments.Columns.Add("TestAcc", "Test Accuracy");
            dgvExperiments.Columns.Add("Prec", "Precision");
            dgvExperiments.Columns.Add("Rec", "Recall");
            dgvExperiments.Columns.Add("F1", "Test F1 Score");

            page.Controls.Add(dgvExperiments);
            page.Controls.Add(topPanel);
        }

        private void BuildDatasetTab(TabPage page)
        {
            Panel topFilter = new Panel
            {
                Dock = DockStyle.Top,
                Height = 44,
                BackColor = Color.White
            };

            AddInlineLabel(topFilter, "Inspect Dataset Stage:", 6, 11);
            rbViewRaw = new RadioButton { Text = "1. Raw CSV (zeros highlighted in red)", Location = new Point(140, 9), AutoSize = true, Checked = true };
            rbViewImputed = new RadioButton { Text = "2. Median-Imputed (training medians in green)", Location = new Point(380, 9), AutoSize = true };
            rbViewNormalized = new RadioButton { Text = "3. Min-Max Normalized [0, 1]", Location = new Point(665, 9), AutoSize = true };

            rbViewRaw.CheckedChanged += (s, e) => { if (rbViewRaw.Checked) PopulateDatasetGrid(); };
            rbViewImputed.CheckedChanged += (s, e) => { if (rbViewImputed.Checked) PopulateDatasetGrid(); };
            rbViewNormalized.CheckedChanged += (s, e) => { if (rbViewNormalized.Checked) PopulateDatasetGrid(); };

            topFilter.Controls.AddRange(new Control[] { rbViewRaw, rbViewImputed, rbViewNormalized });

            Panel bottomStatsPanel = new Panel
            {
                Dock = DockStyle.Bottom,
                Height = 148,
                Padding = new Padding(0, 8, 0, 0),
                BackColor = Color.White
            };

            Label lblStatsHeader = new Label
            {
                Text = "TRAINING-SET ONLY PREPROCESSING PARAMETERS (NO DATA LEAKAGE INTO TEST SET)",
                Dock = DockStyle.Top,
                Height = 20,
                Font = new Font("Segoe UI Semibold", 8.75F, FontStyle.Bold),
                ForeColor = HeaderDarkBg
            };

            dgvImputationStats = CreateCleanGrid();
            dgvImputationStats.Dock = DockStyle.Fill;
            dgvImputationStats.Columns.Add("Feature", "Feature Column");
            dgvImputationStats.Columns.Add("ZeroRule", "Zero Handling Rule");
            dgvImputationStats.Columns.Add("Median", "Training Non-Zero Median");
            dgvImputationStats.Columns.Add("Replaced", "Zeros Imputed");
            dgvImputationStats.Columns.Add("TrainMin", "Training Min");
            dgvImputationStats.Columns.Add("TrainMax", "Training Max");

            bottomStatsPanel.Controls.Add(dgvImputationStats);
            bottomStatsPanel.Controls.Add(lblStatsHeader);

            dgvDataset = CreateCleanGrid();
            dgvDataset.Dock = DockStyle.Fill;

            page.Controls.Add(dgvDataset);
            page.Controls.Add(bottomStatsPanel);
            page.Controls.Add(topFilter);
        }

        private void BuildConsoleTab(TabPage page)
        {
            txtConsoleLog = new TextBox
            {
                Dock = DockStyle.Fill,
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
                WordWrap = false,
                BorderStyle = BorderStyle.None,
                BackColor = Color.FromArgb(15, 23, 42),
                ForeColor = Color.FromArgb(226, 232, 240),
                Font = new Font("Consolas", 9.75F, FontStyle.Regular)
            };
            page.Controls.Add(txtConsoleLog);
        }

        private static DataGridView CreateCleanGrid()
        {
            DataGridView dgv = new DataGridView
            {
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AllowUserToResizeRows = false,
                ReadOnly = true,
                RowHeadersVisible = false,
                BackgroundColor = Color.White,
                BorderStyle = BorderStyle.FixedSingle,
                GridColor = Color.FromArgb(226, 232, 240),
                SelectionMode = DataGridViewSelectionMode.FullRowSelect,
                MultiSelect = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
                ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing,
                ColumnHeadersHeight = 28
            };
            dgv.RowTemplate.Height = 24;
            dgv.EnableHeadersVisualStyles = false;
            dgv.ColumnHeadersDefaultCellStyle.BackColor = Color.FromArgb(241, 245, 249);
            dgv.ColumnHeadersDefaultCellStyle.ForeColor = Color.FromArgb(30, 41, 59);
            dgv.ColumnHeadersDefaultCellStyle.Font = new Font("Segoe UI Semibold", 8.5F, FontStyle.Bold);
            dgv.DefaultCellStyle.SelectionBackColor = Color.FromArgb(219, 234, 254);
            dgv.DefaultCellStyle.SelectionForeColor = Color.Black;
            return dgv;
        }

        private static void DrawCardBorder(object? sender, PaintEventArgs e)
        {
            if (sender is Control c)
            {
                using Pen p = new Pen(CardBorder, 1F);
                e.Graphics.DrawRectangle(p, 0, 0, c.Width - 1, c.Height - 1);
            }
        }

        private async void OnMainFormShown(object? sender, EventArgs e)
        {
            LogLine("================================================================================");
            LogLine(" PIMA INDIANS DIABETES CLASSIFIER — CUSTOM C# 3-LAYER MLP (backneural)");
            LogLine("================================================================================");
            LogLine("");

            LoadAndPreprocessDataset();
            LoadExperimentsCsvIntoGrid();
            InitializeNewMlp();
            await TrainCurrentModelAsync((int)numEpochs.Value, (int)numLogStep.Value);
        }

        private void OnLoadAndPreprocessClick(object? sender, EventArgs e)
        {
            LoadAndPreprocessDataset();
            InitializeNewMlp();
        }

        private void OnBrowseCsvClick(object? sender, EventArgs e)
        {
            using OpenFileDialog ofd = new OpenFileDialog
            {
                Title = "Select Pima Indians Diabetes CSV",
                Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*"
            };
            if (ofd.ShowDialog(this) == DialogResult.OK)
            {
                currentCsvPath = ofd.FileName;
                LoadAndPreprocessDataset();
                InitializeNewMlp();
            }
        }

        private void LoadAndPreprocessDataset()
        {
            try
            {
                double trainRatio = (double)numTrainSplit.Value / 100.0;
                int seed = (int)numSeed.Value;
                bool stratified = chkStratified.Checked;
                bool impute = chkImpute.Checked;

                currentSplit = DataLoader.LoadAndPreprocess(currentCsvPath, trainRatio, seed, stratified, impute);

                lblKpiDataset.Text = $"{currentSplit.ValidRowCount} ({currentSplit.TrainFeatures.Count}/{currentSplit.TestFeatures.Count})";

                LogLine($"[DataLoader] Loaded '{Path.GetFileName(currentCsvPath)}': {currentSplit.ValidRowCount} valid rows ({currentSplit.SkippedRowCount} skipped).");
                LogLine($"[DataLoader] Split ({(stratified ? "stratified" : "random")}, seed={seed}): {currentSplit.TrainFeatures.Count} train, {currentSplit.TestFeatures.Count} test.");

                PopulateImputationStatsGrid();
                PopulateDatasetGrid();

                lblStatusText.Text = $"Loaded {currentSplit.ValidRowCount} rows from {Path.GetFileName(currentCsvPath)}. Zero-value medians & Min-Max fitted on training set only.";
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "Dataset Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void PopulateImputationStatsGrid()
        {
            if (currentSplit == null) return;
            dgvImputationStats.Rows.Clear();

            for (int col = 0; col < currentSplit.FeatureNames.Length; col++)
            {
                bool isZeroImputed = DataLoader.ZeroImputeColumns.Contains(col);
                string rule = isZeroImputed ? "0 = Missing (Impute w/ Train Median)" : "Valid Numeric (0 Allowed)";
                string medText = isZeroImputed ? currentSplit.ImputedTrainingMedians[col].ToString("F2", CultureInfo.InvariantCulture) : "N/A";
                string repText = isZeroImputed ? currentSplit.ZeroCountsReplaced[col].ToString(CultureInfo.InvariantCulture) : "0";

                int r = dgvImputationStats.Rows.Add(
                    currentSplit.FeatureNames[col],
                    rule,
                    medText,
                    repText,
                    currentSplit.TrainMin[col].ToString("F3", CultureInfo.InvariantCulture),
                    currentSplit.TrainMax[col].ToString("F3", CultureInfo.InvariantCulture));

                if (isZeroImputed)
                {
                    dgvImputationStats.Rows[r].DefaultCellStyle.BackColor = Color.FromArgb(240, 253, 244);
                }
            }
            dgvImputationStats.ClearSelection();
        }

        private void PopulateDatasetGrid()
        {
            if (currentSplit == null) return;

            dgvDataset.SuspendLayout();
            dgvDataset.Rows.Clear();
            dgvDataset.Columns.Clear();

            dgvDataset.Columns.Add("Row", "#");
            dgvDataset.Columns[0].FillWeight = 42;
            for (int c = 0; c < currentSplit.FeatureNames.Length; c++)
            {
                dgvDataset.Columns.Add("F" + c, currentSplit.FeatureNames[c]);
                dgvDataset.Columns[c + 1].FillWeight = 75;
            }
            dgvDataset.Columns.Add("Outcome", "Outcome");
            dgvDataset.Columns[9].FillWeight = 75;
            dgvDataset.Columns.Add("Split", "Split");
            dgvDataset.Columns[10].FillWeight = 55;

            bool isRaw = rbViewRaw.Checked;
            bool isImputed = rbViewImputed.Checked;
            List<double[]> rows = isRaw
                ? currentSplit.AllRawFeatures
                : (isImputed ? currentSplit.AllCleanedFeatures : currentSplit.AllScaledFeatures);

            Color redCell = Color.FromArgb(254, 215, 215);
            Color greenCell = Color.FromArgb(209, 250, 229);

            for (int i = 0; i < rows.Count; i++)
            {
                double[] vals = rows[i];
                double[] raw = currentSplit.AllRawFeatures[i];
                object[] rowObjs = new object[11];
                rowObjs[0] = i + 1;
                for (int c = 0; c < 8; c++)
                {
                    rowObjs[c + 1] = (!isRaw && !isImputed)
                        ? vals[c].ToString("F3", CultureInfo.InvariantCulture)
                        : vals[c].ToString("G", CultureInfo.InvariantCulture);
                }
                rowObjs[9] = currentSplit.AllLabels[i] == 1 ? "1 (Diabetic)" : "0 (Healthy)";
                rowObjs[10] = currentSplit.RowIsTrain[i] ? "Train" : "Test";

                int rIdx = dgvDataset.Rows.Add(rowObjs);
                foreach (int zCol in DataLoader.ZeroImputeColumns)
                {
                    if (Math.Abs(raw[zCol]) < 1e-12)
                    {
                        dgvDataset.Rows[rIdx].Cells[zCol + 1].Style.BackColor = isRaw ? redCell : greenCell;
                    }
                }
            }

            dgvDataset.ClearSelection();
            dgvDataset.ResumeLayout();
        }

        private void InitializeNewMlp()
        {
            if (currentSplit == null) return;

            int hidden = (int)numHidden.Value;
            double lr = (double)numLearningRate.Value;
            int seed = (int)numSeed.Value;

            currentNet = new NeuralNet(8, hidden, 1, lr, seed);
            totalEpochsTrained = 0;
            lossHistory.Clear();

            double trBce = currentNet.ComputeBinaryCrossEntropyLoss(currentSplit.TrainFeatures, currentSplit.TrainLabels);
            double teBce = currentNet.ComputeBinaryCrossEntropyLoss(currentSplit.TestFeatures, currentSplit.TestLabels);
            double trMse = currentNet.ComputeMeanSquaredErrorLoss(currentSplit.TrainFeatures, currentSplit.TrainLabels);
            double teMse = currentNet.ComputeMeanSquaredErrorLoss(currentSplit.TestFeatures, currentSplit.TestLabels);
            var trEval = Metrics.Evaluate(currentNet, currentSplit.TrainFeatures, currentSplit.TrainLabels);
            var teEval = Metrics.Evaluate(currentNet, currentSplit.TestFeatures, currentSplit.TestLabels);

            lossHistory.Add(new EpochSnapshot
            {
                Epoch = 0,
                TrainBce = trBce,
                TestBce = teBce,
                TrainMse = trMse,
                TestMse = teMse,
                TrainAcc = trEval.Accuracy,
                TestAcc = teEval.Accuracy
            });

            lblKpiArchitecture.Text = $"8 → {hidden} → 1 (LR {lr:G})";
            LogLine($"[NeuralNet] Initialized 8-{hidden}-1 MLP (LR={lr:G}, seed={seed}) — Initial Train BCE={trBce:F5}, Test Acc={teEval.Accuracy * 100.0:F1}%.");

            RefreshEvaluationWidgets();
            EvaluateSinglePatient();
            canvasLossCurve.Invalidate();
        }

        private async Task TrainCurrentModelAsync(int epochsToAdd, int logInterval)
        {
            if (currentSplit == null || currentNet == null) return;

            currentNet.setLearningRate((double)numLearningRate.Value);
            lblKpiArchitecture.Text = $"8 → {currentNet.HiddenCount} → 1 (LR {currentNet.LearningRate:G})";

            SetTrainingBusyState(true);
            trainCts = new CancellationTokenSource();
            CancellationToken token = trainCts.Token;

            int startEpoch = totalEpochsTrained + 1;
            int endEpoch = totalEpochsTrained + epochsToAdd;

            LogLine("");
            LogLine($"[Training] Running {epochsToAdd} epochs on 8-{currentNet.HiddenCount}-1 MLP (LR={currentNet.LearningRate:G})...");
            LogLine("  Epoch |  Train BCE |   Test BCE |  Train MSE |   Test MSE |  Train Acc |   Test Acc");
            LogLine("--------+------------+------------+------------+------------+------------+-----------");

            statusProgress.Minimum = 0;
            statusProgress.Maximum = epochsToAdd;
            statusProgress.Value = 0;

            DatasetSplit split = currentSplit;
            NeuralNet net = currentNet;

            try
            {
                await Task.Run(() =>
                {
                    for (int ep = startEpoch; ep <= endEpoch; ep++)
                    {
                        if (token.IsCancellationRequested) break;

                        for (int i = 0; i < split.TrainFeatures.Count; i++)
                        {
                            net.setInputs(split.TrainFeatures[i]);
                            net.setDesiredOutput(0, split.TrainLabels[i]);
                            net.learn();
                        }

                        double trBce = net.ComputeBinaryCrossEntropyLoss(split.TrainFeatures, split.TrainLabels);
                        double teBce = net.ComputeBinaryCrossEntropyLoss(split.TestFeatures, split.TestLabels);
                        double trMse = net.ComputeMeanSquaredErrorLoss(split.TrainFeatures, split.TrainLabels);
                        double teMse = net.ComputeMeanSquaredErrorLoss(split.TestFeatures, split.TestLabels);

                        bool isMilestone = ((ep - startEpoch + 1) % logInterval == 0) || (ep == endEpoch) || (ep == 1);
                        double trAcc = 0.0;
                        double teAcc = 0.0;

                        if (isMilestone)
                        {
                            trAcc = Metrics.Evaluate(net, split.TrainFeatures, split.TrainLabels).Accuracy;
                            teAcc = Metrics.Evaluate(net, split.TestFeatures, split.TestLabels).Accuracy;
                        }

                        EpochSnapshot snap = new EpochSnapshot
                        {
                            Epoch = ep,
                            TrainBce = trBce,
                            TestBce = teBce,
                            TrainMse = trMse,
                            TestMse = teMse,
                            TrainAcc = trAcc,
                            TestAcc = teAcc
                        };

                        int progressDone = ep - startEpoch + 1;
                        this.BeginInvoke(new Action(() =>
                        {
                            totalEpochsTrained = ep;
                            lossHistory.Add(snap);
                            statusProgress.Value = Math.Min(statusProgress.Maximum, progressDone);

                            if (isMilestone)
                            {
                                LogLine(string.Format(
                                    CultureInfo.InvariantCulture,
                                    " {0,6} |   {1,8:F6} |   {2,8:F6} |   {3,8:F6} |   {4,8:F6} |     {5,5:F1}% |     {6,5:F1}%",
                                    ep, trBce, teBce, trMse, teMse, trAcc * 100.0, teAcc * 100.0));
                                canvasLossCurve.Invalidate();
                            }
                        }));
                    }
                }, token);
            }
            finally
            {
                SetTrainingBusyState(false);
            }

            ExportTrainingLogToFile(Path.Combine("output", "training_log.csv"));
            RefreshEvaluationWidgets();
            EvaluateSinglePatient();
            canvasLossCurve.Invalidate();
            lblStatusText.Text = $"Completed {totalEpochsTrained} epochs. Outputs updated in output/training_log.csv and output/final_metrics.txt.";
        }

        private void RefreshEvaluationWidgets()
        {
            if (currentSplit == null || currentNet == null) return;

            EvaluationResult testEval = Metrics.Evaluate(currentNet, currentSplit.TestFeatures, currentSplit.TestLabels);
            double trainBce = currentNet.ComputeBinaryCrossEntropyLoss(currentSplit.TrainFeatures, currentSplit.TrainLabels);
            double testBce = currentNet.ComputeBinaryCrossEntropyLoss(currentSplit.TestFeatures, currentSplit.TestLabels);

            // Top KPI banner updates
            lblKpiTestAccuracy.Text = $"{(testEval.Accuracy * 100.0).ToString("F2", CultureInfo.InvariantCulture)}%";
            lblKpiTestF1.Text = $"{testEval.F1Score.ToString("F4", CultureInfo.InvariantCulture)} / {(testEval.Recall * 100.0).ToString("F1", CultureInfo.InvariantCulture)}%";

            // Metric progress bars
            UpdateMetricBar(lblMetricAccVal, barAccFill, testEval.Accuracy);
            UpdateMetricBar(lblMetricPrecVal, barPrecFill, testEval.Precision);
            UpdateMetricBar(lblMetricRecVal, barRecFill, testEval.Recall);
            UpdateMetricBar(lblMetricF1Val, barF1Fill, testEval.F1Score);

            // Confusion matrix tiles
            lblCmTn.Text = testEval.TrueNegatives.ToString(CultureInfo.InvariantCulture);
            lblCmFp.Text = testEval.FalsePositives.ToString(CultureInfo.InvariantCulture);
            lblCmFn.Text = testEval.FalseNegatives.ToString(CultureInfo.InvariantCulture);
            lblCmTp.Text = testEval.TruePositives.ToString(CultureInfo.InvariantCulture);

            // Per-class table
            int negTotal = testEval.TrueNegatives + testEval.FalsePositives;
            int posTotal = testEval.TruePositives + testEval.FalseNegatives;
            double prec0 = (testEval.TrueNegatives + testEval.FalseNegatives) > 0
                ? (double)testEval.TrueNegatives / (testEval.TrueNegatives + testEval.FalseNegatives)
                : 0.0;
            double rec0 = negTotal > 0 ? (double)testEval.TrueNegatives / negTotal : 0.0;
            double f1_0 = (prec0 + rec0) > 0 ? 2.0 * prec0 * rec0 / (prec0 + rec0) : 0.0;

            double macroP = (prec0 + testEval.Precision) / 2.0;
            double macroR = (rec0 + testEval.Recall) / 2.0;
            double macroF1 = (f1_0 + testEval.F1Score) / 2.0;

            dgvSummaryMetrics.Rows.Clear();
            dgvSummaryMetrics.Rows.Add(
                "0 (Non-Diabetic)",
                (prec0 * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                (rec0 * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                f1_0.ToString("F4", CultureInfo.InvariantCulture),
                negTotal);

            int rPos = dgvSummaryMetrics.Rows.Add(
                "1 (Diabetic +)",
                (testEval.Precision * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                (testEval.Recall * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                testEval.F1Score.ToString("F4", CultureInfo.InvariantCulture),
                posTotal);
            dgvSummaryMetrics.Rows[rPos].DefaultCellStyle.BackColor = Color.FromArgb(240, 253, 244);
            dgvSummaryMetrics.Rows[rPos].DefaultCellStyle.Font = new Font("Segoe UI Semibold", 8.75F, FontStyle.Bold);

            int rMacro = dgvSummaryMetrics.Rows.Add(
                "Macro Average",
                (macroP * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                (macroR * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                macroF1.ToString("F4", CultureInfo.InvariantCulture),
                negTotal + posTotal);
            dgvSummaryMetrics.Rows[rMacro].DefaultCellStyle.Font = new Font("Segoe UI Semibold", 8.75F, FontStyle.Bold);
            dgvSummaryMetrics.ClearSelection();

            // Save final_metrics.txt
            string report = Metrics.FormatReport(testEval, currentNet.LearningRate, currentNet.HiddenCount, totalEpochsTrained, trainBce, testBce);
            Directory.CreateDirectory("output");
            File.WriteAllText(Path.Combine("output", "final_metrics.txt"), report);
        }

        private static void UpdateMetricBar(Label lbl, Panel fill, double ratio)
        {
            double clamped = Math.Max(0.0, Math.Min(1.0, ratio));
            lbl.Text = (clamped * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%";
            if (fill.Parent != null)
            {
                fill.Width = (int)Math.Round(fill.Parent.Width * clamped);
            }
        }

        private void EvaluateSinglePatient()
        {
            if (currentSplit == null || currentNet == null) return;

            double[] raw =
            {
                (double)numPreg.Value,
                (double)numGluc.Value,
                (double)numBp.Value,
                (double)numSkin.Value,
                (double)numIns.Value,
                (double)numBmi.Value,
                (double)numPed.Value,
                (double)numAge.Value
            };

            double[] scaled = currentSplit.NormalizeSinglePatient(raw);
            lastPatientProbability = currentNet.PredictProbability(scaled);

            bool isDiabetic = lastPatientProbability >= 0.5;
            if (lblDiagnosisVerdict.Parent is Panel box)
            {
                box.BackColor = isDiabetic ? Color.FromArgb(254, 226, 226) : Color.FromArgb(209, 250, 229);
            }

            lblDiagnosisVerdict.Text = isDiabetic ? "DIABETIC RISK DETECTED (1)" : "NEGATIVE — LOW RISK (0)";
            lblDiagnosisVerdict.ForeColor = isDiabetic ? Color.FromArgb(185, 28, 28) : Color.FromArgb(6, 95, 70);
            lblDiagnosisDetail.Text = string.Format(
                CultureInfo.InvariantCulture,
                "Sigmoid Output: {0:F4} ({1:F1}% Diabetic)",
                lastPatientProbability,
                lastPatientProbability * 100.0);

            canvasSigmoidGauge.Invalidate();
        }

        private void OnPaintSigmoidGauge(object? sender, PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.Clear(canvasSigmoidGauge.BackColor);

            int w = canvasSigmoidGauge.Width;
            int h = canvasSigmoidGauge.Height;

            using Font smallFont = new Font("Segoe UI", 7.75F, FontStyle.Bold);
            using Brush textBrush = new SolidBrush(Color.FromArgb(51, 65, 85));

            g.DrawString("0.0 (Healthy)", smallFont, textBrush, 8, 8);
            g.DrawString("0.5", smallFont, textBrush, w / 2 - 10, 8);
            g.DrawString("1.0 (Diabetic)", smallFont, textBrush, w - 82, 8);

            Rectangle barRect = new Rectangle(12, 28, w - 24, 18);
            using (LinearGradientBrush grad = new LinearGradientBrush(
                barRect,
                Color.FromArgb(16, 185, 129),
                Color.FromArgb(239, 68, 68),
                LinearGradientMode.Horizontal))
            {
                g.FillRectangle(grad, barRect);
            }

            using Pen borderPen = new Pen(Color.FromArgb(148, 163, 184), 1F);
            g.DrawRectangle(borderPen, barRect);

            // Threshold 0.5 dashed marker
            int midX = barRect.Left + barRect.Width / 2;
            using Pen midPen = new Pen(Color.FromArgb(15, 23, 42), 1.5F);
            g.DrawLine(midPen, midX, barRect.Top - 3, midX, barRect.Bottom + 3);

            // Pointer for current probability
            int ptrX = barRect.Left + (int)Math.Round(Math.Max(0.0, Math.Min(1.0, lastPatientProbability)) * barRect.Width);
            Point[] triangle =
            {
                new Point(ptrX, barRect.Bottom + 1),
                new Point(ptrX - 6, barRect.Bottom + 11),
                new Point(ptrX + 6, barRect.Bottom + 11)
            };
            using Brush ptrBrush = new SolidBrush(HeaderDarkBg);
            g.FillPolygon(ptrBrush, triangle);
        }

        private void OnPaintLossChart(object? sender, PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.Clear(Color.White);

            int w = canvasLossCurve.ClientSize.Width;
            int h = canvasLossCurve.ClientSize.Height;
            if (w < 220 || h < 180) return;

            int padLeft = 62;
            int padRight = 24;
            int padTop = 58;
            int padBottom = 48;
            Rectangle plot = new Rectangle(padLeft, padTop, w - padLeft - padRight, h - padTop - padBottom);

            using Font titleFont = new Font("Segoe UI Semibold", 10F, FontStyle.Bold);
            using Font axisFont = new Font("Segoe UI", 8.25F, FontStyle.Regular);
            using Brush darkBrush = new SolidBrush(HeaderDarkBg);
            using Brush mutedBrush = new SolidBrush(Color.FromArgb(100, 116, 139));
            using Pen gridPen = new Pen(Color.FromArgb(241, 245, 249), 1F);
            using Pen framePen = new Pen(CardBorder, 1F);

            string title = showMseCurve
                ? "TRAINING vs. TEST MEAN SQUARED ERROR (MSE)"
                : "TRAINING vs. TEST BINARY CROSS-ENTROPY (BCE) LOSS";
            g.DrawString(title, titleFont, darkBrush, padLeft, 32);

            double maxVal = showMseCurve ? 0.30 : 0.70;
            if (lossHistory.Count > 0)
            {
                double peak = lossHistory.Max(s => showMseCurve ? Math.Max(s.TrainMse, s.TestMse) : Math.Max(s.TrainBce, s.TestBce));
                maxVal = Math.Max(showMseCurve ? 0.25 : 0.65, Math.Ceiling(peak * 20.0) / 20.0);
            }

            int maxEpoch = Math.Max(100, lossHistory.Count > 0 ? lossHistory.Max(s => s.Epoch) : 1000);

            int ySteps = 5;
            for (int i = 0; i <= ySteps; i++)
            {
                double v = maxVal * (ySteps - i) / ySteps;
                int y = plot.Top + (int)Math.Round((double)plot.Height * i / ySteps);
                g.DrawLine(gridPen, plot.Left, y, plot.Right, y);
                string lbl = v.ToString("F3", CultureInfo.InvariantCulture);
                SizeF sz = g.MeasureString(lbl, axisFont);
                g.DrawString(lbl, axisFont, mutedBrush, plot.Left - sz.Width - 6, y - sz.Height / 2);
            }

            int xSteps = 5;
            for (int i = 0; i <= xSteps; i++)
            {
                int ep = (int)Math.Round((double)maxEpoch * i / xSteps);
                int x = plot.Left + (int)Math.Round((double)plot.Width * i / xSteps);
                g.DrawLine(gridPen, x, plot.Top, x, plot.Bottom);
                string lbl = ep.ToString(CultureInfo.InvariantCulture);
                SizeF sz = g.MeasureString(lbl, axisFont);
                g.DrawString(lbl, axisFont, mutedBrush, x - sz.Width / 2, plot.Bottom + 6);
            }

            g.DrawString("Epoch", axisFont, darkBrush, plot.Left + plot.Width / 2 - 18, plot.Bottom + 24);
            g.DrawRectangle(framePen, plot);

            if (lossHistory.Count >= 2)
            {
                PointF[] trPts = new PointF[lossHistory.Count];
                PointF[] tePts = new PointF[lossHistory.Count];

                for (int i = 0; i < lossHistory.Count; i++)
                {
                    var s = lossHistory[i];
                    double tr = showMseCurve ? s.TrainMse : s.TrainBce;
                    double te = showMseCurve ? s.TestMse : s.TestBce;

                    float x = plot.Left + (float)((double)s.Epoch / maxEpoch * plot.Width);
                    float yTr = plot.Bottom - (float)(Math.Min(maxVal, Math.Max(0.0, tr)) / maxVal * plot.Height);
                    float yTe = plot.Bottom - (float)(Math.Min(maxVal, Math.Max(0.0, te)) / maxVal * plot.Height);

                    trPts[i] = new PointF(x, yTr);
                    tePts[i] = new PointF(x, yTe);
                }

                using Pen trPen = new Pen(AccentBlue, 2.0F);
                using Pen tePen = new Pen(AccentTeal, 2.0F);
                g.DrawLines(trPen, trPts);
                g.DrawLines(tePen, tePts);
            }

            // Legend Pill
            Rectangle leg = new Rectangle(plot.Right - 185, plot.Top + 10, 175, 48);
            using Brush legFill = new SolidBrush(Color.FromArgb(248, 250, 252));
            g.FillRectangle(legFill, leg);
            g.DrawRectangle(framePen, leg);

            using Pen p1 = new Pen(AccentBlue, 2.5F);
            using Pen p2 = new Pen(AccentTeal, 2.5F);
            g.DrawLine(p1, leg.Left + 10, leg.Top + 15, leg.Left + 32, leg.Top + 15);
            g.DrawString("Training Set Loss", axisFont, darkBrush, leg.Left + 38, leg.Top + 7);
            g.DrawLine(p2, leg.Left + 10, leg.Top + 34, leg.Left + 32, leg.Top + 34);
            g.DrawString("Held-Out Test Loss", axisFont, darkBrush, leg.Left + 38, leg.Top + 26);
        }

        private void LoadExperimentsCsvIntoGrid()
        {
            string expPath = DataLoader.ResolveRelativePath(Path.Combine("output", "experiments.csv"));
            if (!File.Exists(expPath)) return;

            string[] lines = File.ReadAllLines(expPath);
            dgvExperiments.Rows.Clear();

            int bestIdx = -1;
            double bestF1 = -1.0;
            double bestAcc = -1.0;

            for (int i = 1; i < lines.Length; i++)
            {
                string[] parts = lines[i].Trim().Split(',');
                if (parts.Length < 8) continue;

                double.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out double trLoss);
                double.TryParse(parts[4], NumberStyles.Float, CultureInfo.InvariantCulture, out double acc);
                double.TryParse(parts[5], NumberStyles.Float, CultureInfo.InvariantCulture, out double prec);
                double.TryParse(parts[6], NumberStyles.Float, CultureInfo.InvariantCulture, out double rec);
                double.TryParse(parts[7], NumberStyles.Float, CultureInfo.InvariantCulture, out double f1);

                int idx = dgvExperiments.Rows.Add(
                    i,
                    parts[0],
                    parts[1],
                    parts[2],
                    trLoss.ToString("F6", CultureInfo.InvariantCulture),
                    (acc * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                    (prec * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                    (rec * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                    f1.ToString("F4", CultureInfo.InvariantCulture));

                if (f1 > bestF1 || (Math.Abs(f1 - bestF1) < 1e-9 && acc > bestAcc))
                {
                    bestF1 = f1;
                    bestAcc = acc;
                    bestIdx = idx;
                }
            }

            if (bestIdx >= 0)
            {
                dgvExperiments.Rows[bestIdx].DefaultCellStyle.BackColor = Color.FromArgb(209, 250, 229);
                dgvExperiments.Rows[bestIdx].DefaultCellStyle.Font = new Font("Segoe UI Semibold", 9F, FontStyle.Bold);
            }
            dgvExperiments.ClearSelection();
        }

        private async void OnApplySelectedGridRowClick(object? sender, EventArgs e)
        {
            if (dgvExperiments.SelectedRows.Count == 0) return;
            DataGridViewRow row = dgvExperiments.SelectedRows[0];

            if (decimal.TryParse(Convert.ToString(row.Cells[1].Value), NumberStyles.Float, CultureInfo.InvariantCulture, out decimal lr) &&
                int.TryParse(Convert.ToString(row.Cells[2].Value), NumberStyles.Integer, CultureInfo.InvariantCulture, out int hid) &&
                int.TryParse(Convert.ToString(row.Cells[3].Value), NumberStyles.Integer, CultureInfo.InvariantCulture, out int ep))
            {
                numLearningRate.Value = lr;
                numHidden.Value = hid;
                numEpochs.Value = ep;

                mainTabs.SelectedIndex = 0;
                InitializeNewMlp();
                await TrainCurrentModelAsync(ep, (int)numLogStep.Value);
            }
        }

        private async void OnRunGridSearchClick(object? sender, EventArgs e)
        {
            if (currentSplit == null) return;

            double[] lrs = ParseDoubles(txtGridLrs.Text, new[] { 0.01, 0.05, 0.1, 0.5 });
            int[] hids = ParseInts(txtGridHiddens.Text, new[] { 4, 8, 16, 32 });
            int[] eps = ParseInts(txtGridEpochs.Text, new[] { 500, 1000, 2000 });
            int seed = (int)numSeed.Value;

            int total = lrs.Length * hids.Length * eps.Length;
            dgvExperiments.Rows.Clear();
            SetTrainingBusyState(true);
            statusProgress.Minimum = 0;
            statusProgress.Maximum = total;
            statusProgress.Value = 0;

            List<ExperimentRecord> records = new List<ExperimentRecord>();
            DatasetSplit split = currentSplit;
            int runNum = 0;
            int bestRow = -1;
            double bestF1 = -1.0;
            double bestAcc = -1.0;

            await Task.Run(() =>
            {
                foreach (double lr in lrs)
                {
                    foreach (int h in hids)
                    {
                        foreach (int ep in eps)
                        {
                            runNum++;
                            var (net, trLoss, teLoss) = ExperimentRunner.TrainModel(split, lr, h, ep, false, null, seed);
                            EvaluationResult ev = Metrics.Evaluate(net, split.TestFeatures, split.TestLabels);
                            ExperimentRecord rec = new ExperimentRecord(lr, h, ep, trLoss, teLoss, ev);
                            records.Add(rec);

                            int cur = runNum;
                            this.BeginInvoke(new Action(() =>
                            {
                                statusProgress.Value = Math.Min(statusProgress.Maximum, cur);
                                int idx = dgvExperiments.Rows.Add(
                                    cur,
                                    lr.ToString("G", CultureInfo.InvariantCulture),
                                    h,
                                    ep,
                                    trLoss.ToString("F6", CultureInfo.InvariantCulture),
                                    (ev.Accuracy * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                                    (ev.Precision * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                                    (ev.Recall * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "%",
                                    ev.F1Score.ToString("F4", CultureInfo.InvariantCulture));

                                if (ev.F1Score > bestF1 || (Math.Abs(ev.F1Score - bestF1) < 1e-9 && ev.Accuracy > bestAcc))
                                {
                                    bestF1 = ev.F1Score;
                                    bestAcc = ev.Accuracy;
                                    bestRow = idx;
                                }
                            }));
                        }
                    }
                }
            });

            if (bestRow >= 0)
            {
                dgvExperiments.Rows[bestRow].DefaultCellStyle.BackColor = Color.FromArgb(209, 250, 229);
                dgvExperiments.Rows[bestRow].DefaultCellStyle.Font = new Font("Segoe UI Semibold", 9F, FontStyle.Bold);
            }

            // Save to output/experiments.csv
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("learning_rate,hidden_neurons,epochs,final_train_loss,test_accuracy,precision,recall,f1");
            foreach (var r in records)
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
            Directory.CreateDirectory("output");
            File.WriteAllText(Path.Combine("output", "experiments.csv"), sb.ToString());

            SetTrainingBusyState(false);
            lblStatusText.Text = $"Completed {total} grid search runs and saved output/experiments.csv.";
        }

        private void OnExportLossCsvClick(object? sender, EventArgs e)
        {
            using SaveFileDialog sfd = new SaveFileDialog
            {
                Title = "Export Epoch Loss History",
                Filter = "CSV files (*.csv)|*.csv",
                FileName = "training_log.csv"
            };
            if (sfd.ShowDialog(this) == DialogResult.OK)
            {
                ExportTrainingLogToFile(sfd.FileName);
                lblStatusText.Text = $"Exported loss history to {sfd.FileName}.";
            }
        }

        private void ExportTrainingLogToFile(string path)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("epoch,train_loss,test_loss");
            foreach (var s in lossHistory.Where(x => x.Epoch > 0))
            {
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0},{1:F6},{2:F6}", s.Epoch, s.TrainBce, s.TestBce));
            }
            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(path, sb.ToString());
        }

        private static double[] ParseDoubles(string input, double[] fallback)
        {
            List<double> list = new List<double>();
            foreach (string s in input.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries))
            {
                if (double.TryParse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && v > 0)
                    list.Add(v);
            }
            return list.Count > 0 ? list.ToArray() : fallback;
        }

        private static int[] ParseInts(string input, int[] fallback)
        {
            List<int> list = new List<int>();
            foreach (string s in input.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries))
            {
                if (int.TryParse(s.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0)
                    list.Add(v);
            }
            return list.Count > 0 ? list.ToArray() : fallback;
        }

        private void SetTrainingBusyState(bool busy)
        {
            btnLoadCsv.Enabled = !busy;
            btnResetModel.Enabled = !busy;
            btnTrainModel.Enabled = !busy;
            btnStopTraining.Enabled = busy;
            btnRunExperiments.Enabled = !busy;
            btnApplySelectedConfig.Enabled = !busy;
        }

        private void LogLine(string text)
        {
            if (txtConsoleLog.IsDisposed) return;
            txtConsoleLog.AppendText(text + Environment.NewLine);
        }
    }
}
