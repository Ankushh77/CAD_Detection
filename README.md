CAD Detection

Machine learning project that predicts the presence of coronary artery disease (CAD) from patient clinical data. It compares five classic classifiers, each combined with an ensemble method, and evaluates them with accuracy, precision, recall, F1 and 10-fold cross-validation.

Disclaimer: This is an educational project. It is not a medical tool and must not be used for diagnosis.

Overview
Loads a heart disease dataset (heart.csv) with clinical features such as age, sex, chest pain type (cp), resting blood pressure (trestbps) and thal.
Explores the data with scatter plots and histograms (age vs sex, thal distribution, trestbps distribution, cp vs trestbps).
Splits the data 70/30 with a stratified train/test split.
Trains five models and compares their performance.
Models
Base model	Ensemble method
Support Vector Classifier (linear kernel)	AdaBoost (150 estimators)
K-Nearest Neighbors (k = 1)	Bagging (300 estimators)
Decision Tree	AdaBoost (50 estimators)
Logistic Regression	AdaBoost (150 estimators)
Random Forest	AdaBoost (150 estimators)

For every model the code prints metrics before and after 10-fold cross-validation, plus a classification report. A final bar chart compares the accuracy of all classifiers.

Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn

Project Structure
CAD_Detection/
├── CAD_Detection.ipynb   # Notebook with outputs and explanations
├── cad_detection.py      # Same code exported as a Python script
└── README.md
How to Run
Clone the repository:
bash
   git clone https://github.com/Ankushh77/CAD_Detection.git
   cd CAD_Detection
Install the requirements:
bash
   pip install "scikit-learn<1.4" pandas numpy matplotlib seaborn

(Newer scikit-learn versions renamed base_estimator to estimator in AdaBoost and Bagging.) 3. Get a heart disease dataset as heart.csv and place it in the project folder. 4. The code was written in Google Colab and reads the file from Google Drive. To run it locally, remove the google.colab lines and set:

python
   path = "heart.csv"
Run the notebook, or run python cad_detection.py.
Known Limitations and Next Steps
The cross-validation loop trains on folds of the full dataset and then scores on the original test split, so the test rows are seen during training. This inflates the scores of high-capacity models (Decision Tree, Random Forest, KNN). Moving to a proper held-out evaluation (for example cross_val_score on the training set only) is the next improvement.
The accuracy values in the final comparison chart are typed in by hand, not computed from the runs.
Hyperparameters were not tuned.
Author

Ankush (@Ankushh77)
