

# Network Intrusion Detection 

This project implements a robust machine learning system designed to detect malicious intrusions in network traffic. By analyzing the features of network connections, the system leverages various classification algorithms to accurately distinguish between normal, benign activity and potential cyber threats. The project follows a complete data science workflow, from data preprocessing and exploratory analysis to model comparison, hyperparameter optimization, and final evaluation, culminating in a highly effective model capable of identifying attacks with approximately 99% accuracy.



## 📊 Dataset

The project uses the **Network Intrusion Detection Dataset** from Kaggle:
- Source: [Network Intrusion Detection Dataset](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)
- Description: Contains network connection records with features extracted from network traffic
- Target: Binary classification (Normal vs Intrusion)

## 🎯 Project Objective

Develop and compare multiple machine learning models to effectively detect network intrusions and identify the best-performing algorithm for cybersecurity applications.

## 🛠️ Models Implemented

The following machine learning algorithms were implemented and evaluated:

| Model | Train Score | Test Score |
|-------|-------------|------------|
| Light GBM | 0.999943 | 0.995634 |
| XGBM | 0.999943 | 0.995501 |
| Random Forest | 0.999773 | 0.995898 |
| Decision Tree | 1.000000 | 0.992458 |
| CatBoost | 0.998469 | 0.994046 |
| GBM | 0.995236 | 0.992061 |
| KNN | 0.979415 | 0.979757 |
| Adaboost | 0.976239 | 0.970892 |
| Logistic Regression | 0.941704 | 0.938873 |
| Naive Bayes | 0.893785 | 0.894813 |
| Voting Classifier | 0.99983 | 0.995634 |

## 🏆 Best Performance

- Top Model: Light GBM & Voting Classifier
- Best Test Accuracy: 99.56%
- Training Accuracy: 99.99%

## 📈 Key Results

### Model Performance Highlights:
- Ensemble Methods (Random Forest, XGBoost, Light GBM) consistently outperformed individual classifiers
- Tree-based models showed excellent performance with minimal overfitting
- Traditional ML methods (Naive Bayes, Logistic Regression) showed lower but acceptable performance
- Voting Classifier achieved competitive results by combining multiple strong learners

### Performance Analysis:
- Random Forest: Highest test accuracy (99.59%) with good generalization
- XGBoost & Light GBM: Excellent performance (~99.55%) with faster training
- Decision Tree: Perfect training score but slightly lower test performance
- KNN: Moderate performance, suitable for baseline comparison

## 🔧 Technologies Used

- Python 3.x
- Scikit-learn - Machine Learning algorithms
- XGBoost - Gradient boosting framework
- LightGBM - Gradient boosting framework
- CatBoost - Gradient boosting framework
- Pandas - Data manipulation
- NumPy - Numerical computing
- Matplotlib/Seaborn - Data visualization



## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn
```

### Usage

1. Clone the repository:
```bash
https://github.com/Pavani1210/network-intrusion-detection.git
cd network-intrusion-detection
```

2. Download the dataset:
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)
   - Download and place in the `data/` directory



## 📊 Model Evaluation Metrics

The models were evaluated using:
- Accuracy Score
- Precision, Recall, and F1-Score
- Confusion Matrix
- ROC-AUC Score
- Cross-validation scores

## 🔍 Key Insights

1. Ensemble methods significantly outperform individual classifiers
2. Tree-based algorithms are particularly effective for this type of structured network data
3. Minimal overfitting observed in top-performing models
4. Feature engineering and hyperparameter tuning can further improve performance

## 🎓 Learning Outcomes

- Comprehensive comparison of multiple ML algorithms
- Understanding of ensemble methods effectiveness
- Practical experience with cybersecurity datasets
- Model evaluation and selection techniques


## 👨‍💻 Author

**Your Name**
- GitHub: [@Pavani1210]
- LinkedIn: [byreddy Reddy pavani]
- Email: pavanibyreddy128@gmail.com

## 🌟 Acknowledgments

- Kaggle community for the dataset and inspiration
- Open-source contributors of the ML libraries used
- Cybersecurity research community for domain insights

---

⭐ If you found this project helpful, please give it a star on GitHub!
