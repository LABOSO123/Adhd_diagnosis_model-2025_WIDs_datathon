ADHD Prediction and Gender Classification Model

📌 Overview

This repository contains a machine learning pipeline developed for the WiDS Datathon 2025, focusing on predicting ADHD and classifying gender based on neurophysiological data. The study explores feature engineering, model selection, and explainability techniques to enhance ADHD diagnosis, particularly in females.

🌟 Why This Matters

ADHD diagnosis in females is understudied and often misdiagnosed due to symptom variability. This model aims to leverage machine learning to identify distinct brain activity patterns, leading to personalized interventions and improved early diagnosis.


📊 Dataset

## Dataset Access
The dataset used in this project is part of the [WiDS Datathon 2025](https://www.kaggle.com/competitions/widsdatathon2025).  
To access the data, register and download it from the competition page.  


The dataset contains 2 folders - Train and Test and a data dictionary, all which can be accessed from kaggle. 

The dataset consists of neurophysiological features linked to ADHD diagnosis in the connectome matrices dataset.

Includes both quantitative measures (e.g., EEG signals, response times) and categorical factors (e.g., demographic attributes).

Preprocessing steps handled missing values, outliers, and feature transformations for robust modeling.


📌 Methodology & Approach

The model development follows a structured machine learning pipeline:

Exploratory Data Analysis (EDA) – Understanding feature distributions, correlations, and missing patterns.

Feature Engineering (FE) – Creating new meaningful features and reducing dimensionality.

Data Preprocessing – Encoding categorical variables, scaling numerical data, and handling imbalanced classes.



Model Selection – Training and comparing multiple models including:

Logistic Regression

Random Forest

HistoricalGradientBoost


Hyperparameter Tuning – Optimized models using GridSearchCV / Optuna.

Evaluation Metrics – Performance measured using:

Accuracy, Precision, Recall, F1-score (classification)

ROC-AUC Score (for ADHD prediction)

SHAP & LIME for interpretability


📌 Results & Insights

Best Performing Model: [HistoricalGradientBoost]

Key Findings:

Feature 22 in the brain matrices had the most impact on ADHD prediction.

Gender classification improved with Feature Y.

Addressing class imbalance significantly enhanced performance.

📌 How to Run the Project

🔹 Clone the Repository

git clone https://github.com/your-username/adhd-prediction-model.git  
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

🔹 Incorporate deep learning techniques (CNNs, RNNs) for improved prediction.🔹 Deploy the model using Flask / FastAPI for real-world application.🔹 Implement feature importance analysis for better interpretability.

📌 Acknowledgments

WiDS Datathon organizers for the dataset and challenge.

C4C Bootcamp for equipping women with valuable data analytics skills.

📌 Let's Connect!

💼 LinkedIn: (https://www.linkedin.com/in/faith-chemutai-laboso/)🐙 GitHub: (https://github.com/LABOSO123)

🚀 Feel free to fork this repository, contribute, or reach out with any questions!


<<<<<<< HEAD
📌 Overview

This repository contains a machine learning pipeline developed for the WiDS Datathon 2025, focusing on predicting ADHD and classifying gender based on neurophysiological data. The study explores feature engineering, model selection, and explainability techniques to enhance ADHD diagnosis, particularly in females.

🌟 Why This Matters

ADHD diagnosis in females is understudied and often misdiagnosed due to symptom variability. This model aims to leverage machine learning to identify distinct brain activity patterns, leading to personalized interventions and improved early diagnosis.


📊 Dataset
The dataset contains 2 folders - Train and Test and a data dictionary, all which can be accessed from kaggle. 

The dataset consists of neurophysiological features linked to ADHD diagnosis in the connectome matrices dataset.

Includes both quantitative measures (e.g., EEG signals, response times) and categorical factors (e.g., demographic attributes).

Preprocessing steps handled missing values, outliers, and feature transformations for robust modeling.


📌 Methodology & Approach

The model development follows a structured machine learning pipeline:

Exploratory Data Analysis (EDA) – Understanding feature distributions, correlations, and missing patterns.

Feature Engineering (FE) – Creating new meaningful features and reducing dimensionality.

Data Preprocessing – Encoding categorical variables, scaling numerical data, and handling imbalanced classes.



Model Selection – Training and comparing multiple models including:

Logistic Regression

Random Forest

HistoricalGradientBoost


Hyperparameter Tuning – Optimized models using GridSearchCV / Optuna.

Evaluation Metrics – Performance measured using:

Accuracy, Precision, Recall, F1-score (classification)

ROC-AUC Score (for ADHD prediction)

SHAP & LIME for interpretability


📌 Results & Insights

Best Performing Model: [HistoricalGradientBoost]

Key Findings:

Feature 22 in the brain matrices had the most impact on ADHD prediction.

Gender classification improved with Feature Y.

Addressing class imbalance significantly enhanced performance.

📌 How to Run the Project

🔹 Clone the Repository

git clone https://github.com/your-username/adhd-prediction-model.git  
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

🔹 Incorporate deep learning techniques (CNNs, RNNs) for improved prediction.🔹 Deploy the model using Flask / FastAPI for real-world application.🔹 Implement feature importance analysis for better interpretability.

📌 Acknowledgments

WiDS Datathon organizers for the dataset and challenge.

C4C Bootcamp for equipping women with valuable data analytics skills.

📌 Let's Connect!

💼 LinkedIn: (https://www.linkedin.com/in/faith-chemutai-laboso/)🐙 GitHub: (https://github.com/LABOSO123)

🚀 Feel free to fork this repository, contribute, or reach out with any questions!


=======
# WIDS-DATATHON_2025
 This is a repository exploring the dataset from WIDs datathon 2025 in collaboration with Kaggle. This model explores exploratory data analysis, feature engineering and modelling.
>>>>>>> c620527 (Initial commit)
=======
>>>>>>> f85aa1b (Update README.md)
