# 📘 Computer Engineering Department — ML Project SOP Master Guide
## Loan Default Prediction System (CrediPulse AI)

Welcome to the comprehensive, complete 10-week implementation of the **Computer Engineering Department Machine Learning Project Standard Operating Procedure (SOP)**.

---

## 📅 Weekly Milestone Roadmap & Quick Links

| Week | Milestone Title | Prescribed Task List | Key Deliverable Link |
| :---: | :--- | :--- | :--- |
| **Week 1** | **Problem Definition & Dataset Exploration** | Problem statement, dataset summary, and initial observations. | 📓 [Week 1 Notebook](file:///f:/loan-default-prediction%20-%20Copy/notebooks/Week_01_Problem_Definition_and_Dataset_Exploration.ipynb) |
| **Week 2** | **Data Cleaning & Preprocessing** | Handle missing values, outliers, categorical encoding, scaling, EDA. | 📓 [Week 2 Notebook](file:///f:/loan-default-prediction%20-%20Copy/notebooks/Week_02_Data_Cleaning_and_Preprocessing.ipynb) |
| **Week 3** | **Model Creation** | Appropriate algorithm selection, Library model & **Implementation WITHOUT use of Library**. | 📓 [Week 3 Notebook](file:///f:/loan-default-prediction%20-%20Copy/notebooks/Week_03_Model_Creation_Library_and_Scratch.ipynb)<br>🐍 [Scratch Algorithm Source](file:///f:/loan-default-prediction%20-%20Copy/backend/models/scratch_decision_tree.py) |
| **Week 4** | **Model Evaluation** | Test metrics (Acc, Prec, Rec, F1, AUC), overfitting & underfitting diagnostics. | 📓 [Week 4 Notebook](file:///f:/loan-default-prediction%20-%20Copy/notebooks/Week_04_Model_Evaluation_and_Overfitting_Check.ipynb) |
| **Week 5** | **Advanced Model Training** | Advanced ensembles (Random Forest, Gradient Boosting), 5-Fold Stratified CV, Hyperparameter Tuning. | 📓 [Week 5 Notebook](file:///f:/loan-default-prediction%20-%20Copy/notebooks/Week_05_Advanced_Model_Training_CV_and_Tuning.ipynb) |
| **Week 6** | **Visualization of Metrics & Graphs** | Display all types of performance graphs (ROC, PR, Confusion Matrix, Feature Importance, Calibration, etc.). | 📓 [Week 6 Notebook](file:///f:/loan-default-prediction%20-%20Copy/notebooks/Week_06_Performance_Metrics_and_Visualizations.ipynb)<br>🖼️ [Saved Visualizations](file:///f:/loan-default-prediction%20-%20Copy/reports/visualizations) |
| **Week 7** | **Flask Project Setup** | Learn Flask setup, build Flask app with form handlers. | 🌐 [Flask Application Code](file:///f:/loan-default-prediction%20-%20Copy/flask_app/app.py) |
| **Week 8** | **Create Front End** | Interactive frontend for user to input data & evaluate risk. | 🎨 [Frontend Form Template](file:///f:/loan-default-prediction%20-%20Copy/flask_app/templates/predict.html)<br>🎨 [Result Display Template](file:///f:/loan-default-prediction%20-%20Copy/flask_app/templates/result.html) |
| **Week 9** | **Create Backend, Deployment** | Setup backend requests, REST API, deployment config for free hosting. | 🚀 [Runner Script](file:///f:/loan-default-prediction%20-%20Copy/run_flask.py)<br>📦 [Procfile](file:///f:/loan-default-prediction%20-%20Copy/Procfile)<br>⚙️ [render.yaml](file:///f:/loan-default-prediction%20-%20Copy/render.yaml)<br>🐳 [Dockerfile](file:///f:/loan-default-prediction%20-%20Copy/Dockerfile)<br>📖 [Free Hosting Guide](file:///f:/loan-default-prediction%20-%20Copy/DEPLOYMENT_GUIDE.md) |
| **Week 10** | **Project Evaluation** | Final comprehensive project evaluation report, business cost matrix, error analysis. | 📓 [Week 10 Notebook](file:///f:/loan-default-prediction%20-%20Copy/notebooks/Week_10_Project_Evaluation.ipynb)<br>📄 [Week 10 Evaluation Report](file:///f:/loan-default-prediction%20-%20Copy/WEEK_10_PROJECT_EVALUATION_REPORT.md) |

---

## 🚀 How to Run the Project

### Option A: Run the Flask Web Application (SOP Weeks 7, 8, 9)
```bash
python run_flask.py
```
Then navigate to:
- **Home:** `http://127.0.0.1:5000/`
- **Interactive Risk Form:** `http://127.0.0.1:5000/predict`
- **SOP Visualizations Gallery:** `http://127.0.0.1:5000/metrics`
- **About & Documentation:** `http://127.0.0.1:5000/about`

### Option B: Run the FastAPI + React Application
```bash
# Terminal 1: Backend
python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Option C: Run the Weekly Jupyter Notebooks
Launch Jupyter in the project directory:
```bash
jupyter notebook
```
Navigate to `notebooks/` and select any week (`Week_01` to `Week_06` or `Week_10`).
