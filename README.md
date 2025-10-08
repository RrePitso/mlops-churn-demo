Customer Churn Prediction (MLOps Project)

This project demonstrates a complete Machine Learning Operations (MLOps) workflow for customer churn prediction.
It integrates model training, automated retraining, and performance monitoring through a Streamlit dashboard, ensuring that business decisions are continuously informed by up-to-date models.


---

Project Overview

Customer churn is a critical metric for subscription-based businesses. This project predicts the likelihood of a customer leaving (churning) based on their demographic and service information.
The system is designed to be automated and production-ready, capable of retraining monthly as new data becomes available.


---

Key Features

Automated Model Retraining:
Configured via a GitHub Actions workflow (retrain.yml) to trigger retraining every 30 days or whenever new data is uploaded.

Streamlit Web App:
A clean and interactive web interface for making predictions and viewing model performance trends.

Performance Monitoring:
Automatically logs metrics (Accuracy, F1 Score) and visualizes trends over time using Plotly.

Reusable ML Pipeline:
Includes feature encoding, scaling, model training, and serialization for reproducible results.



---

Repository Structure

.
├── .github/workflows/
│   └── retrain.yml              # Automated retraining workflow (scheduled monthly)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Training dataset
├── model/
│   ├── churn_model.pkl          # Trained ML model
│   ├── encoder.pkl              # Categorical encoder
│   ├── scaler.pkl               # Feature scaler
│   └── metrics_history.csv      # Historical performance metrics
├── train.py                     # Initial training script
├── retrain.py                   # Retraining script (used by GitHub Actions)
├── app.py                       # Streamlit dashboard
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation


---

How It Works

1. Training:

The model is initially trained using train.py.

Preprocessing includes encoding categorical variables and scaling numeric features.

The trained model and preprocessors are saved in the model/ directory.



2. Deployment:

The Streamlit app (app.py) loads the model and provides a web interface for predictions.

The app also visualizes model performance trends using data from metrics_history.csv.



3. Retraining Automation:

The GitHub Actions workflow (retrain.yml) automatically triggers retrain.py every month.

It loads the latest dataset, retrains the model, and commits updated artifacts and metrics back to the repository.





---

Running Locally

Prerequisites

Python 3.9+

pip installed


Installation

git clone https://github.com/RrePitso/mlops-churn-demo.git
cd mlops-churn-demo
pip install -r requirements.txt

Run the App

streamlit run app.py

The dashboard will open in your browser at http://localhost:8501


---

Updating Data

To retrain with new data:

1. Replace the existing CSV file inside the data/ folder with your latest dataset.


2. Ensure the file name remains the same (WA_Fn-UseC_-Telco-Customer-Churn.csv).


3. Push changes to GitHub — the retraining workflow will execute automatically.




---

Technologies Used

Python (Pandas, NumPy, scikit-learn)

Streamlit (Interactive dashboard)

Plotly (Visualization)

Joblib (Model serialization)

GitHub Actions (Automation and CI/CD)



---

Future Improvements

Integrate MLflow for experiment tracking and model versioning.

Add alert notifications for model drift detection.

Deploy the Streamlit app to a cloud platform (Streamlit Cloud, AWS, or GCP).

Extend retraining triggers to detect data changes automatically.



---

Author

Ofentse Pitso
Data Scientist & MLOps Enthusiast
GitHub: github.com/RrePitso
LinkedIn: linkedin.com/in/ofentse-pitso
