# Heart Disease Prediction Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-84.9%25-brightgreen.svg)](#evaluation-metrics)
[![Dataset](https://img.shields.io/badge/Dataset-Framingham%20Heart%20Study-orange.svg)](#dataset)

Predict the 10-year risk of coronary heart disease (CHD) using machine learning—because your heart deserves a heads-up.

## Project Goals

This machine learning project aims to:
- **Predict** 10-year coronary heart disease risk with high accuracy
- **Identify** key risk factors from clinical and lifestyle data
- **Provide** actionable insights for early intervention
- **Demonstrate** data science techniques in healthcare applications

## Overview

This project leverages the renowned **Framingham Heart Study** dataset to build a predictive model for coronary heart disease over a 10-year period. The Framingham study, one of the longest-running epidemiological studies, provides invaluable insights into cardiovascular risk factors. Our model combines demographic, behavioral, and clinical features to deliver data-driven cardiac risk assessments.

### Key Highlights
- **84.9% accuracy** on test data
- **3,751 clean samples** processed
- **13 clinical features** analyzed
- **Logistic regression** with feature standardization

## Dataset

The dataset `framingham.csv` contains comprehensive health data from the Framingham Heart Study participants:

### Feature Categories

**Demographics**
- `sex_male` - Gender (1 = male, 0 = female)
- `age` - Age in years

**Behavioral Risk Factors**
- `currentSmoker` - Current smoking status (1 = yes, 0 = no)
- `cigsPerDay` - Number of cigarettes smoked per day

**Medical History**
- `BPMeds` - Blood pressure medication usage (1 = yes, 0 = no)
- `prevalentStroke` - History of stroke (1 = yes, 0 = no)
- `prevalentHyp` - History of hypertension (1 = yes, 0 = no)
- `diabetes` - Diabetes status (1 = yes, 0 = no)

**Clinical Measurements**
- `totChol` - Total cholesterol level (mg/dL)
- `sysBP` - Systolic blood pressure (mmHg)
- `diaBP` - Diastolic blood pressure (mmHg)
- `BMI` - Body Mass Index (kg/m²)
- `heartRate` - Heart rate (beats per minute)
- `glucose` - Glucose level (mg/dL)

**Target Variable**
- `TenYearCHD` - 10-year CHD risk (1 = developed CHD, 0 = did not)

### Dataset Statistics
- **Total records**: 4,240 initial samples
- **Clean samples**: 3,751 (after preprocessing)
- **CHD cases**: 572 (15.2% - indicating class imbalance)
- **Missing data**: Handled through row removal (489 rows removed)

## Data Preprocessing Pipeline

Our preprocessing ensures data quality and model readiness:

### Data Cleaning
1. **Feature selection**: Dropped `education` column (low predictive value)
2. **Column renaming**: `male` → `sex_male` for semantic clarity
3. **Missing data**: Removed rows with null values (489 rows removed)

### Feature Engineering
4. **Key feature selection**: Focused on 6 high-impact features:
   - `age`, `sex_male`, `cigsPerDay`, `totChol`, `sysBP`, `glucose`
5. **Data splitting**: 70% training (2,625 samples) / 30% testing (1,126 samples)
6. **Standardization**: Applied `StandardScaler` for feature normalization

### Quality Assurance
- Verified data types and ranges
- Checked for outliers and anomalies
- Ensured balanced train-test distribution

## Exploratory Data Analysis

### Key Insights
- **Sample size**: 3,751 participants after cleaning
- **CHD prevalence**: 572 cases (15.2%) - reveals class imbalance challenge
- **Age distribution**: Wide range across adult population
- **Gender split**: Mixed male/female representation

### Visualizations Created
- **Target distribution**: Count plots showing CHD vs non-CHD cases with cubehelix palette
- **Trend analysis**: Line plots for CHD occurrence patterns
- **Class distribution**: Visual representation of the 85% healthy vs 15% CHD split

## Model Development

### Algorithm Selection
- **Primary model**: Logistic Regression
- **Rationale**: Interpretable, probabilistic output, suitable for binary classification
- **Implementation**: scikit-learn LogisticRegression with default hyperparameters

### Training Process
1. Feature standardization using training data statistics
2. Model fitting on scaled training features
3. Hyperparameter optimization (future enhancement)
4. Cross-validation for robust performance estimation

### Model Performance
- **Test accuracy**: 84.9%
- **Training time**: < 1 second
- **Model size**: Lightweight and deployable

## Evaluation Metrics

### Classification Report

| Metric     | Class 0 (No CHD) | Class 1 (CHD) | Weighted Avg |
|------------|------------------|---------------|--------------|
| **Precision**  | 0.85             | 0.61          | 0.82         |
| **Recall**     | 0.99             | 0.08          | 0.85         |
| **F1-score**   | 0.92             | 0.14          | 0.80         |
| **Support**    | 951              | 175           | 1,126        |

### Performance Analysis

**Strengths**
- **High overall accuracy** (84.9%)
- **Excellent at identifying healthy individuals** (99% recall for Class 0)
- **Low false positive rate** for CHD predictions

**Areas for Improvement**
- **Poor CHD detection** (8% recall for Class 1)
- **Class imbalance impact** - model bias toward majority class
- **Low F1-score for CHD cases** indicates room for improvement

### Confusion Matrix
```
Predicted:    No CHD    CHD
Actual:
No CHD         942       9
CHD            161      14
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pynthamil/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv heart-disease-env
   source heart-disease-env/bin/activate  # On Windows: heart-disease-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

4. **Run the analysis**
   ```bash
   jupyter notebook heart_disease_prediction.ipynb
   ```
   Or execute all cells programmatically:
   ```bash
   python heart_disease_prediction.py
   ```

5. **Make predictions**
   ```python
   # Example prediction
   from src.predict import predict_chd_risk
   
   patient_data = {
       'age': 45,
       'sex_male': 1,
       'cigsPerDay': 10,
       'totChol': 250,
       'sysBP': 140,
       'glucose': 90
   }
   
   risk_probability = predict_chd_risk(patient_data)
   print(f"10-year CHD risk: {risk_probability:.2%}")
   ```

## Project Structure

```
Heart-Disease-Prediction/
├── data/
│   └── framingham.csv          # Dataset
├── notebooks/
│   └── heart_disease_prediction.ipynb  # Main analysis
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning functions
│   ├── model_training.py       # Model building
│   ├── evaluation.py           # Metrics and visualization
│   └── predict.py              # Prediction utilities
├── models/
│   └── trained_model.pkl       # Saved model
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
└── LICENSE                     # MIT License
```

## Future Roadmap

### Model Improvements
- **Algorithm diversification**: Random Forest, XGBoost, Neural Networks
- **Ensemble methods**: Voting classifier, stacking
- **Hyperparameter tuning**: Grid search, Bayesian optimization
- **Feature engineering**: Polynomial features, interaction terms

### Class Imbalance Solutions
- **SMOTE**: Synthetic Minority Oversampling Technique
- **Class weighting**: Penalize majority class errors
- **Threshold tuning**: Optimize decision boundary
- **Cost-sensitive learning**: Asymmetric loss functions

### Technical Enhancements
- **Cross-validation**: K-fold validation for robust evaluation
- **Feature selection**: Recursive feature elimination, LASSO
- **Model interpretation**: SHAP values, feature importance
- **Performance monitoring**: MLflow integration

### Deployment & Applications
- **Web application**: Flask/FastAPI REST API
- **Interactive dashboard**: Streamlit/Dash interface
- **Mobile app**: Healthcare provider integration
- **Clinical decision support**: Hospital system integration

### Data & Research
- **External validation**: Test on different populations
- **Longitudinal analysis**: Time-series risk modeling
- **Multi-modal data**: Incorporate imaging, genetic data
- **Real-world validation**: Prospective clinical studies

## Clinical Significance

### Risk Factors Identified
Our model highlights key predictors of CHD risk:
- **Age**: Strong positive correlation with CHD risk
- **Gender**: Male gender increases risk
- **Smoking**: Cigarettes per day shows dose-response relationship
- **Cholesterol**: Total cholesterol levels impact prediction
- **Blood pressure**: Systolic BP is a significant predictor
- **Glucose**: Metabolic factors contribute to risk

### Clinical Applications
- **Preventive care**: Early identification of high-risk patients
- **Resource allocation**: Prioritize interventions for high-risk individuals
- **Patient counseling**: Data-driven risk communication
- **Population health**: Epidemiological insights for public health policy

## Dependencies

### Core Libraries
```
pandas>=1.3.0           # Data manipulation
numpy>=1.21.0           # Numerical computing
scikit-learn>=1.0.0     # Machine learning
matplotlib>=3.4.0       # Basic plotting
seaborn>=0.11.0         # Statistical visualization
```

### Optional Extensions
```
jupyter>=1.0.0          # Interactive notebooks
plotly>=5.0.0          # Interactive visualizations
shap>=0.41.0           # Model interpretability
mlflow>=1.20.0         # Experiment tracking
streamlit>=1.0.0       # Web app framework
```

### Development Tools
```
pytest>=6.0.0          # Testing framework
black>=21.0.0          # Code formatting
flake8>=3.9.0          # Linting
pre-commit>=2.15.0     # Git hooks
```

## Model Card

| Attribute | Value |
|-----------|-------|
| **Model Type** | Logistic Regression |
| **Training Data** | Framingham Heart Study (n=2,625) |
| **Test Performance** | 84.9% accuracy |
| **Input Features** | 6 clinical/demographic variables |
| **Output** | Binary CHD risk (0/1) + probability |
| **Bias Considerations** | Class imbalance toward healthy individuals |
| **Intended Use** | Research and educational purposes |
| **Limitations** | Not for clinical diagnosis |

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- Model improvements and new algorithms
- Data visualization enhancements
- Documentation improvements
- Web application development
- Clinical validation studies

## References

1. Kannel, W.B., et al. (1961). Factors of risk in the development of coronary heart disease. *Annals of Internal Medicine*, 55(1), 33-50.
2. D'Agostino, R.B., et al. (2008). General cardiovascular risk profile for use in primary care. *Circulation*, 117(6), 743-753.
3. Framingham Heart Study. (2021). Risk assessment tool. Retrieved from [framinghamheartstudy.org](https://www.framinghamheartstudy.org)

## Support

- **Issues**: [GitHub Issues](https://github.com/Pynthamil/Heart-Disease-Prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Pynthamil/Heart-Disease-Prediction/discussions)
- **Email**: pyndu15@gmail.com

---

⭐ **Star this repository** if you found it helpful!

**Disclaimer**: This model is for educational and research purposes only. It should not be used for clinical diagnosis or treatment decisions. Always consult healthcare professionals for medical advice.
