# ScoreSense AI

ScoreSense AI is a machine learning based academic score prediction project. It uses student lifestyle and study behavior inputs to estimate an exam score, assign a grade category, and show practical recommendations through a Streamlit web interface.

The project includes the training notebook, extracted notebook code, trained model file, synthetic dataset, model comparison results, and two Streamlit app versions.

## Features

- Predicts exam score from student lifestyle inputs.
- Provides grade categories such as Excellent, Good, Average, and Below Average.
- Uses inputs such as study hours, sleep hours, screen time, active time, social time, attendance, stress level, and gender.
- Shows confidence range, score indicators, feature impact visuals, and improvement suggestions.
- Includes a premium Streamlit interface with an alternate visual design.
- Compares multiple regression models and stores the trained model as a pickle file.

## Project Structure

```text
.
+-- app.py                     # Main Streamlit web application
+-- premium_app.py             # Premium styled Streamlit version
+-- ScoreSense_AI.ipynb        # Main notebook for analysis, training, and evaluation
+-- extracted_notebook_code.py # Python code extracted from the notebook
+-- scoresense_model.pkl       # Trained model used by the Streamlit apps
+-- student_lifestyle_data.csv # Synthetic student lifestyle dataset
+-- model_results.txt          # Model comparison and evaluation results
+-- extract.py                 # Helper script for notebook extraction
+-- .gitignore                 # Ignored local/runtime files
```

## Dataset

The dataset contains synthetic student lifestyle records with columns such as:

- `sleep_hours`
- `study_hours`
- `mobile_usage_hours`
- `tv_hours`
- `exercise_hours`
- `extracurricular_hrs`
- `friends_time_hrs`
- `family_time_hrs`
- `attendance_pct`
- `stress_level`
- `gender`
- `exam_score`

The target variable is `exam_score`.

## Model Performance

The project compares several regression algorithms, including Linear Regression, Ridge Regression, Lasso Regression, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, and K-Nearest Neighbors.

Reported best tuned Gradient Boosting performance:

```text
R2   : 0.8523
MAE  : 2.924
RMSE : 3.703
MAPE : 4.77%
```

Linear Regression and Ridge Regression also performed strongly with an R2 score of approximately `0.8626`.

## Requirements

Install Python 3.10 or newer, then install the required packages:

```bash
pip install streamlit numpy pandas matplotlib seaborn scikit-learn
```

## How To Run

Clone the repository:

```bash
git clone https://github.com/sailingeshvva/ScoreSense_Al.git
cd ScoreSense_Al
```

Run the main app:

```bash
streamlit run app.py
```

Or run the premium version:

```bash
streamlit run premium_app.py
```

The app will open in your browser. Enter the student profile values and click the prediction button to generate the score estimate.

## Workflow

1. Load and explore the synthetic student lifestyle dataset.
2. Preprocess features and encode categorical values.
3. Train and compare multiple regression models.
4. Tune the best model.
5. Save the trained model as `scoresense_model.pkl`.
6. Use the saved model inside the Streamlit app for predictions.

## Notes

- Keep `scoresense_model.pkl` in the project root because both Streamlit apps load it directly.
- If the model file is missing, rerun the notebook to train and export the model.
- The dataset is synthetic and intended for academic project demonstration.

## Author

Sailingeshvva
