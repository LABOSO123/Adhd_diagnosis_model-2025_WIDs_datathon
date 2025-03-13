ADHD Prediction and Gender Classification Model

📌 Overview

This repository contains a machine learning pipeline developed for the WiDS Datathon 2025, focusing on predicting ADHD and classifying gender based on neurophysiological data. The study explores feature engineering, model selection, and explainability techniques to enhance ADHD diagnosis, particularly in females.

🌟 Why This Matters

ADHD diagnosis in females is understudied and often misdiagnosed due to symptom variability. This model leverages machine learning to identify distinct brain activity patterns, enabling personalized interventions and improved early diagnosis.

📊 Dataset

The dataset used in this project is part of the WiDS Datathon 2025. Register and download it from the competition page.

        Folders: Train, Test, and a Data Dictionary (available on Kaggle)
        
        Data Type: Neurophysiological features linked to ADHD diagnosis in connectome matrices
        
        Features Include: EEG signals, response times (quantitative) & demographic attributes (categorical)
        
        Preprocessing Steps: Handling missing values, outliers, and feature transformations

📌 Methodology & Approach

The model development follows a structured machine learning pipeline:

        Exploratory Data Analysis (EDA) – Understanding feature distributions, correlations, and missing patterns.
        
        Feature Engineering (FE) – Creating meaningful features and reducing dimensionality.
        
        Data Preprocessing – Encoding categorical variables, scaling numerical data, handling imbalanced classes.
        
        Model Selection – Training and comparing multiple models:

        Logistic Regression
        
        Random Forest
        
        Gradient Boosting (Best Performing)
        
        Hyperparameter Tuning – Optimized models using GridSearchCV / Optuna.
        
        Evaluation Metrics – Performance measured using:
        
        Accuracy, Precision, Recall, F1-score (classification)
        
        ROC-AUC Score (for ADHD prediction)
        
        SHAP & LIME for interpretability

📌 Results & Insights

Best Performing Model: Gradient Boosting Classifier


Key Findings:

        Feature 22 in the brain matrices had the most impact on ADHD prediction.
        
        Gender classification improved significantly with Feature Y.
        
        Addressing class imbalance significantly enhanced model performance.

📌 How to Run the Project


        🔹 Clone the Repository

        git clone https://github.com/LABOSO123/adhd-prediction-model.git  
        cd adhd-prediction-model  

        🔹 Install Dependencies

        pip install -r requirements.txt  

        🔹 Run the Model

        python train_model.py  

        🔹 Make Predictions

python predict.py --input data/sample_input.csv  

📌 Repository Structure

        📂 adhd-prediction-model  
        ├── 📁 data/                   # Dataset (or instructions on where to get it)  
        ├── 📁 notebooks/               # Jupyter Notebooks for EDA and experimentation  
        ├── 📁 src/                     # Source code for data preprocessing & modeling  
        ├── 📁 models/                  # Trained model files  
        ├── train_model.py              # Script to train the model  
        ├── predict.py                  # Script to generate predictions  
        ├── requirements.txt            # Dependencies  
        ├── README.md                   # Project documentation  

📌 Future Work

    Incorporate deep learning techniques (CNNs, RNNs) for improved prediction.

    Deploy the model using Flask / FastAPI for real-world application.

    Implement feature importance analysis for better interpretability.

📌 Acknowledgments

WiDS Datathon organizers for the dataset and challenge.


📌 Let's Connect!

💼 LinkedIn: Faith Chemutai Laboso🐙 GitHub: LABOSO123

🚀 Feel free to fork this repository, contribute, or reach out with any questions!
