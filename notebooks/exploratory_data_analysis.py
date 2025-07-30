# -*- coding: utf-8 -*-
"""
Credit Card Fraud Detection - Refactored Implementation
Using organized project structure and clean code principles.
"""

# Import necessary libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('..')

# Import our custom modules
from src.data.data_loader import DataLoader
from src.models.model_factory import ModelFactory
from src.training.trainer import ModelTrainer
from src.training.evaluation import ModelEvaluator
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

print("Credit Card Fraud Detection - Refactored Implementation")
print("=" * 60)

# =============================================================================
# 1. DATA LOADING AND EXPLORATION
# =============================================================================

print("\n1. LOADING AND EXPLORING DATA")
print("-" * 40)

# Initialize data loader
data_loader = DataLoader()

# Load raw data
df_raw = data_loader.load_raw_data()

print(f"Dataset shape: {df_raw.shape}")
print(f"Dataset info:")
print(df_raw.info())

# Display first few rows
print("\nFirst 10 rows:")
print(df_raw.head(10))

# Check unique time values
print(f"\nUnique time values: {df_raw['Time'].nunique()}")

# Check for duplicates
duplicates = df_raw.duplicated().sum()
duplicate_percentage = (duplicates / len(df_raw)) * 100
print(f"Duplicates: {duplicates} ({duplicate_percentage:.2f}%)")

# Visualize duplicates
plt.figure(figsize=(10, 6))
vals = {"duplicates": duplicates, "total rows": len(df_raw)}
plt.bar(vals.keys(), vals.values())
plt.title("Duplicates vs Total Rows")
plt.ylabel("Count")
for i, (k, v) in enumerate(vals.items()):
    plt.text(i, v + 1000, str(v), ha='center')
plt.show()

# Check class distribution
fraud_count = df_raw["Class"].sum()
total_count = len(df_raw)
fraud_percentage = (fraud_count / total_count) * 100

print(f"\nClass Distribution:")
print(f"Fraud cases: {fraud_count}")
print(f"Normal cases: {total_count - fraud_count}")
print(f"Fraud percentage: {fraud_percentage:.3f}%")

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n2. EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# Distribution of key features
fig, ax = plt.subplots(1, 2, figsize=(18, 4))

# Amount distribution
amount_val = df_raw['Amount'].values
sns.distplot(amount_val, ax=ax[1], color='r')
ax[1].set_title('Distribution of Transaction Amount', fontsize=14)
ax[1].set_xlim([min(amount_val), max(amount_val)])

# Time distribution
time_val = df_raw['Time'].values
sns.distplot(time_val, ax=ax[0], color='b')
ax[0].set_title('Distribution of Transaction Time', fontsize=14)
ax[0].set_xlim([min(time_val), max(time_val)])

plt.tight_layout()
plt.show()

print("Observations:")
print("- Amount feature is right-skewed (many small transactions)")
print("- Time feature shows periodic patterns")

# Class distribution pie chart
d = df_raw['Class'].value_counts().reset_index(name='Count')
fig = px.pie(d, values='Count', names=['Not Fraud', 'Fraud'], hole=0.4, opacity=0.6,
            color_discrete_sequence=["#0B0A09", "#FF7676"])

fig.add_annotation(text='Highly Unbalanced Dataset<br>Requires Resampling',
                   x=1.2, y=0.9, showarrow=False, font_size=12, opacity=0.7)
fig.add_annotation(text='Class Distribution',
                   x=0.5, y=0.5, showarrow=False, font_size=14, opacity=0.7)

fig.update_layout(
    title=dict(text='Credit Card Fraud Detection - Class Distribution', x=0.47, y=0.98),
    legend=dict(x=0.37, y=-0.05, orientation='h'),
    hoverlabel=dict(bgcolor='white')
)

fig.update_traces(textposition='outside', textinfo='percent+label')
fig.show()

# Time vs Amount scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_raw, x='Time', y="Amount", hue="Class", palette="Set1")
plt.xlabel("Time (seconds)")
plt.title("Time vs Transaction Amount")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.show()

print("Observations from scatter plot:")
print("- Legitimate transactions dominate (Class=0)")
print("- Fraudulent transactions are scattered but may show patterns")
print("- No strong linear relationship between time and amount")

# Correlation heatmap
plt.figure(figsize=(20, 16))
correlation_matrix = df_raw.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
           square=True, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Features with strongest correlation to target
target_correlations = correlation_matrix['Class'].abs().sort_values(ascending=False)
print(f"\nTop 15 features correlated with Class:")
print(target_correlations.head(15))

# Visualize positive correlations
f, axes = plt.subplots(ncols=4, figsize=(20, 4))
positive_corr_features = ['V11', 'V8', 'V4', 'V2']

for i, feature in enumerate(positive_corr_features):
    sns.violinplot(x="Class", y=feature, data=df_raw, palette='Set1', ax=axes[i])
    axes[i].set_title(f'{feature} vs Class')

plt.suptitle('Features with Positive Correlation to Fraud', fontsize=16)
plt.tight_layout()
plt.show()

# Visualize negative correlations
f, axes = plt.subplots(ncols=4, figsize=(20, 4))
negative_corr_features = ['V17', 'V14', 'V12', 'V10']

for i, feature in enumerate(negative_corr_features):
    sns.violinplot(x="Class", y=feature, data=df_raw, palette='Set1', ax=axes[i])
    axes[i].set_title(f'{feature} vs Class')

plt.suptitle('Features with Negative Correlation to Fraud', fontsize=16)
plt.tight_layout()
plt.show()

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================

print("\n3. DATA PREPROCESSING")
print("-" * 40)

# Clean and preprocess data using our DataLoader
df_cleaned = data_loader.clean_data(df_raw.copy())
df_scaled = data_loader.scale_features(df_cleaned)

print("Data preprocessing completed:")
print(f"- Removed {duplicates} duplicate rows")
print(f"- Scaled Amount and Time features using RobustScaler")
print(f"- Final dataset shape: {df_scaled.shape}")

# Show scaled features
print("\nScaled Amount and Time features:")
print(df_scaled[["Amount", "Time"]].head())

# Check for outliers after scaling
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
df_scaled[['Amount']].plot(kind='box', ax=axes[0], color="#578BEC")
axes[0].set_title('Amount - After Scaling')

df_scaled[['Time']].plot(kind='box', ax=axes[1], color="#578BEC")
axes[1].set_title('Time - After Scaling')

plt.tight_layout()
plt.show()

# Prepare features and target
X, y = data_loader.prepare_features_target(df_scaled)
print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Balance the dataset
X_balanced, y_balanced = data_loader.balance_dataset(X, y)
print(f"\nAfter balancing:")
print(f"Features shape: {X_balanced.shape}")
print(f"Target shape: {y_balanced.shape}")
print(f"Class distribution: {y_balanced.value_counts().to_dict()}")

# Visualize balanced dataset
plt.figure(figsize=(8, 6))
sns.countplot(data=pd.DataFrame({'Class': y_balanced}), x='Class')
plt.title("Balanced Dataset - Class Distribution")
plt.show()

# Split the data
X_train, X_test, y_train, y_test = data_loader.split_data(X_balanced, y_balanced)
print(f"\nTrain/Test Split:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# =============================================================================
# 4. MODEL TRAINING AND EVALUATION
# =============================================================================

print("\n4. MODEL TRAINING AND EVALUATION")
print("-" * 40)

# Initialize trainer and evaluator
trainer = ModelTrainer()
evaluator = ModelEvaluator(trainer.config)

# Train individual models as shown in notebook
print("\n4.1 Training Individual Models")
print("-" * 30)

# Logistic Regression
print("Training Logistic Regression...")
lr_model, lr_metrics = trainer.train_single_model('logistic_regression', X_train, y_train, X_test, y_test)
print(f"Logistic Regression Results: {lr_metrics}")

# Plot confusion matrix for Logistic Regression
y_pred_lr = lr_model.predict(X_test)
evaluator.plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")
plt.show()

# Classification report
lr_report = evaluator.generate_classification_report(y_test, y_pred_lr, "Logistic Regression")

# MLP Neural Network
print("\nTraining MLP Neural Network...")
mlp_model, mlp_metrics = trainer.train_single_model('mlp', X_train, y_train, X_test, y_test)
print(f"MLP Results: {mlp_metrics}")

# Plot confusion matrix for MLP
y_pred_mlp = mlp_model.predict(X_test)
evaluator.plot_confusion_matrix(y_test, y_pred_mlp, "MLP Neural Network")
plt.show()

# XGBoost
print("\nTraining XGBoost...")
xgb_model, xgb_metrics = trainer.train_single_model('xgboost', X_train, y_train, X_test, y_test)
print(f"XGBoost Results: {xgb_metrics}")

# =============================================================================
# 5. CROSS-VALIDATION COMPARISON
# =============================================================================

print("\n5. CROSS-VALIDATION COMPARISON")
print("-" * 40)

# K-fold cross-validation as in notebook
cv_results = trainer.train_with_kfold_comparison(X_balanced, y_balanced)
print("\nK-Fold Cross-Validation Results:")
print(cv_results)

# Ensemble models comparison
print("\nEnsemble Models Comparison:")
ensemble_results = trainer.train_ensemble_models(X_balanced, y_balanced)
print(ensemble_results)

# =============================================================================
# 6. ROC CURVE COMPARISON
# =============================================================================

print("\n6. ROC CURVE COMPARISON")
print("-" * 40)

# Train multiple models for ROC comparison
models_for_roc = {
    'Logistic Regression': lr_model,
    'MLP': mlp_model,
    'XGBoost': xgb_model
}

# Add more models
try:
    rf_model, _ = trainer.train_single_model('random_forest', X_train, y_train, X_test, y_test)
    models_for_roc['Random Forest'] = rf_model
except:
    pass

try:
    gb_model, _ = trainer.train_single_model('gradient_boosting', X_train, y_train, X_test, y_test)
    models_for_roc['Gradient Boosting'] = gb_model
except:
    pass

# Plot ROC curves
evaluator.plot_roc_curve(models_for_roc, X_test, y_test)
plt.show()

# =============================================================================
# 7. HYPERPARAMETER TUNING
# =============================================================================

print("\n7. HYPERPARAMETER TUNING")
print("-" * 40)

# Perform grid search as in notebook
grid_search_results = trainer.perform_grid_search_comparison(X_train, y_train)
print("\nGrid Search Results:")
print(grid_search_results)

# =============================================================================
# 8. MODEL COMPARISON AND FINAL RESULTS
# =============================================================================

print("\n8. FINAL MODEL COMPARISON")
print("-" * 40)

# Train all major models
model_types = ['logistic_regression', 'xgboost', 'mlp', 'random_forest', 'svm', 'decision_tree']
all_results = trainer.train_multiple_models(model_types, X_train, y_train, X_test, y_test)

# Compare models
comparison_df = evaluator.compare_models(all_results)
print("\nFinal Model Comparison:")
print(comparison_df)

# Plot model comparison
evaluator.plot_model_comparison(comparison_df, 'f1_weighted')
plt.show()

evaluator.plot_model_comparison(comparison_df, 'accuracy')
plt.show()

# Get best model
best_model_name, best_model = trainer.get_best_model('f1_weighted')
print(f"\nBest Model: {best_model_name}")

# Feature importance for best model (if available)
try:
    feature_names = X_train.columns.tolist()
    importance_df = evaluator.evaluate_feature_importance(best_model, feature_names)
    
    if not importance_df.empty:
        evaluator.plot_feature_importance(importance_df, best_model_name)
        plt.show()
except:
    print("Feature importance not available for the best model")

# =============================================================================
# 9. SAVE RESULTS
# =============================================================================

print("\n9. SAVING RESULTS")
print("-" * 40)

# Save processed data
data_loader.save_processed_data(X_train, X_test, y_train, y_test)

# Save trained models
trainer.save_trained_models()

# Save evaluation results
evaluator.save_evaluation_results(all_results, 'evaluation_results.json')

print("\nProject completed successfully!")
print("All models trained, evaluated, and saved.")
print(f"Best performing model: {best_model_name}")
print(f"Best F1-weighted score: {all_results[best_model_name]['f1_weighted']:.4f}")

# =============================================================================
# 10. SUMMARY STATISTICS
# =============================================================================

print("\n10. SUMMARY STATISTICS")
print("-" * 40)

summary_stats = {
    'Original Dataset Size': len(df_raw),
    'After Cleaning': len(df_cleaned),
    'After Balancing': len(X_balanced),
    'Training Set Size': len(X_train),
    'Test Set Size': len(X_test),
    'Number of Features': X_train.shape[1],
    'Models Trained': len(all_results),
    'Best Model': best_model_name,
    'Best F1 Score': f"{all_results[best_model_name]['f1_weighted']:.4f}"
}

for key, value in summary_stats.items():
    print(f"{key}: {value}")

print("\n" + "="*60)
print("CREDIT CARD FRAUD DETECTION PROJECT COMPLETED")
print("="*60)