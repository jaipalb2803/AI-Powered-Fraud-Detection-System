# AI-Powered Fraud Detection System

This project demonstrates a complete workflow for building, evaluating, and simulating the deployment of a machine learning model to detect fraudulent transactions. The goal is to accurately identify fraud with minimal false positives using real-world transaction data.

## Project Structure
- `fraud_detection.ipynb`: Jupyter notebook containing all code, explanations, and results.
- `train_hsbc_df.csv`: Training dataset with labeled transactions.
- `test_hsbc_df.csv`: Test dataset for generating predictions.
- `fraud_predictions.csv`: Output file with predicted fraud labels and probabilities.
- `fraud_detection_model.joblib`: Saved trained model for deployment.

## Workflow Overview
1. **Import Required Libraries**: Load all necessary Python libraries for data analysis, visualization, and machine learning.
2. **Load and Explore the Dataset**: Read the transaction data, inspect its structure, and visualize class distributions.
3. **Preprocess Data**: Clean the data, handle missing values, encode categorical variables, and scale features.
4. **Split Data**: Divide the data into training and validation sets using stratified sampling.
5. **Train Model**: Train a classification model (e.g., RandomForest) to detect fraud.
6. **Evaluate Model**: Assess the model using precision, recall, F1-score, AUC-ROC, and confusion matrix.
7. **Tune Model**: Adjust thresholds and hyperparameters to minimize false positives.
8. **Export Model**: Save the trained model for future use.
9. **Deploy Model**: Load the saved model and generate predictions on new data.

## How to Use
1. Open `fraud_detection.ipynb` in Jupyter or VS Code.
2. Run each cell in order, following the explanations and outputs.
3. Review the generated `fraud_predictions.csv` for predicted results.

## Requirements
- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

Install requirements with:
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Next Steps
- Further tune model hyperparameters and thresholds.
- Explore advanced models (e.g., XGBoost, neural networks).
- Integrate the model into a real-time transaction monitoring system.

---

**Author:** Badavath Jaipal
**Date:** July 2025
