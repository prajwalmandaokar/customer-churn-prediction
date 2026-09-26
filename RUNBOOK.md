# 🏃 RUNBOOK — How to Run This Project

This guide assumes **no prior setup** on your machine. Follow it top to bottom.
Every command has already been tested against this exact repo.

---

## Step 0 — What you need installed

- **Python 3.10 or newer**
  Check with: `python3 --version` (Windows: `python --version`)
  If you don't have it: download from [python.org/downloads](https://www.python.org/downloads/)
- **Git** (only needed if you're cloning from GitHub — download from
  [git-scm.com](https://git-scm.com/downloads))
- A terminal (Terminal on Mac/Linux, Command Prompt / PowerShell / Git Bash on Windows)

You do **not** need Jupyter, MySQL, or anything else pre-installed —
everything else comes from `requirements.txt`.

---

## Step 1 — Get the project onto your computer

If you downloaded the ZIP:
```bash
# unzip it, then move into the folder
cd customer-churn-prediction
```

If you're pulling it from your own GitHub repo:
```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
```

---

## Step 2 — Create a virtual environment (recommended, keeps things clean)

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You'll know it worked because your terminal prompt now starts with `(venv)`.

> Skipping this step is fine too — it just means the packages install
> globally instead of in a project-specific sandbox.

---

## Step 3 — Install the required packages

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, and
streamlit. It usually takes 1-3 minutes.

**If `pip` isn't recognized:** try `pip3` instead of `pip`, or
`python3 -m pip install -r requirements.txt`.

---

## Step 4 — (Optional) Regenerate the dataset

The dataset is already included at `data/raw/customer_churn.csv`, so you can
skip this — but running it yourself proves the whole pipeline is
reproducible from scratch:

```bash
python src/generate_data.py
```

Expected output:
```
Generated 5,015 rows -> data/raw/customer_churn.csv
Churn rate: 33.00%
```

---

## Step 5 — Explore the data (optional but recommended)

If you have Jupyter installed (`pip install notebook`), open the EDA
notebook:
```bash
jupyter notebook notebooks/01_eda_and_cleaning.ipynb
```
Then click **Run All** from the notebook's toolbar. This walks through the
data-quality issues, cleaning, and every chart referenced in the README.

Don't want to install Jupyter? You can open the `.ipynb` file directly in
**VS Code** (with the Python extension) or **Google Colab** (upload the file
at [colab.research.google.com](https://colab.research.google.com)) — both
run notebooks without any local setup.

---

## Step 6 — Train the models

```bash
python src/train_model.py
```

This will:
1. Load and clean the raw data
2. Split it into train/test sets
3. Train Logistic Regression, Random Forest, and Gradient Boosting
4. Print each model's accuracy/precision/recall/F1/ROC-AUC
5. Save the best-performing model to `models/churn_model.pkl`
6. Save charts to `reports/figures/` (confusion matrix, ROC curve, feature importance)

Expect it to take under a minute on a normal laptop. You'll see output like:
```
Training logistic_regression...
  {'accuracy': 0.7353, 'precision': 0.5783, 'recall': 0.7273, 'f1_score': 0.6443, 'roc_auc': 0.8188}
...
Best model: logistic_regression (ROC-AUC = 0.8188)
```

**Open the results:**
- `reports/metrics.json` — the numbers
- `reports/figures/confusion_matrix.png` — where the model gets things right/wrong
- `reports/figures/roc_curve.png` — how the 3 models compare
- `reports/figures/feature_importance.png` — what drives predictions most

---

## Step 7 — Score customers with the trained model

```bash
python src/predict.py --input data/raw/customer_churn.csv --output reports/predictions.csv
```

This adds two columns (`ChurnProbability`, `ChurnPrediction`) to every row
and saves the result. Open `reports/predictions.csv` in Excel to see it.

To score a **different** file of customers, point `--input` at any CSV with
the same columns as `data/raw/customer_churn.csv` (minus the `Churn` column,
which it doesn't need).

---

## Step 8 — Launch the interactive demo app

```bash
streamlit run app/streamlit_app.py
```

Your browser should open automatically to `http://localhost:8501`. Fill in a
sample customer's details on the left/right forms and click **Predict Churn
Risk**. If your browser doesn't open automatically, copy the URL printed in
the terminal into your browser manually.

To stop the app, go back to the terminal and press `Ctrl + C`.

---

## Step 9 — (Optional) Run the SQL analysis

The queries in `sql/churn_analysis_queries.sql` work against any SQL engine.
The easiest zero-install option is **SQLite** via Python, which you already have:

```bash
python3 -c "
import pandas as pd, sqlite3
df = pd.read_csv('data/raw/customer_churn.csv')
conn = sqlite3.connect('churn.db')
df.to_sql('customer_churn', conn, index=False, if_exists='replace')
print('Loaded into churn.db — open it with any SQLite browser, or run queries via sqlite3 CLI.')
"
sqlite3 churn.db < sql/churn_analysis_queries.sql
```

Or, if you use MySQL/PostgreSQL/DBeaver/TablePlus, the `CREATE TABLE` and
`LOAD DATA` statements are commented at the top of the `.sql` file — adjust
the file path and run it there instead.

---

## 🩹 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` (etc.) | Re-run `pip install -r requirements.txt`. Make sure your virtual environment is activated. |
| `python: command not found` | Try `python3` instead of `python`. |
| `streamlit: command not found` | Make sure your virtual environment is activated, or run `python -m streamlit run app/streamlit_app.py`. |
| `train_model.py` says it can't import `data_preprocessing` | Run it from inside the `src/` folder (`cd src && python train_model.py`), or from the project root using `python -m src.train_model` (advanced). The Quickstart commands in the README already handle this. |
| Streamlit app says "No trained model found" | Run `python src/train_model.py` first — the app needs `models/churn_model.pkl` to exist. |
| Port 8501 already in use | Run `streamlit run app/streamlit_app.py --server.port 8502` |

---

## 📤 Pushing this to your own GitHub repo

```bash
cd customer-churn-prediction
git init
git add .
git commit -m "Initial commit: customer churn prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/customer-churn-prediction.git
git push -u origin main
```

Then, on GitHub:
1. Add a short description + topics (`machine-learning`, `python`, `churn-prediction`, `streamlit`, `scikit-learn`) in the repo's "About" section.
2. Pin this repo on your GitHub profile.
3. (Optional) Deploy the Streamlit app for free at [share.streamlit.io](https://share.streamlit.io) and add the live link to the top of your README.
