# -*- coding: utf-8 -*-
"""
SAE Diagnostic Model Development
================================
Manuscript: Automated Whole-Brain MRI Segmentation Combined with Clinical Scores
           for the Diagnosis of Sepsis-Associated Encephalopathy

This script performs:
1. Data loading and Z-score standardization
2. LASSO feature selection with 10-fold CV (Figure 2)
3. Training and evaluation of 8 ML models with stratified hold-out validation
4. Model comparison (ROC curves, confusion matrices, Table 2)
5. XGBoost feature importance and SHAP analysis (Figure 5)
6. Sensitivity analysis excluding APACHE II and SOFA scores (Figure 6)

Note: Due to patient privacy restrictions, the original dataset cannot be publicly
shared. A simulated data generator is included for demonstration purposes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LassoCV, LogisticRegression, lasso_path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, cohen_kappa_score)
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample

from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')

# ======================== 配置参数 ========================
RANDOM_STATE = 42
TEST_SIZE = 0.3
N_FOLDS_CV = 10
N_FOLDS_GRID = 3
N_BOOTSTRAP = 1000
DATA_FILE = "data.xlsx"
OUTPUT_DIR = "./"


# ======================== 模拟数据生成函数 ========================
def generate_simulated_data(n_samples=107, n_features=540, save_path=None):
    np.random.seed(RANDOM_STATE)

    n_control = 35
    n_nonsae = 38
    n_sae = 34
    y = np.array([0] * n_control + [1] * n_nonsae + [2] * n_sae)

    n_total = n_control + n_nonsae + n_sae
    X = np.random.randn(n_total, n_features) * 0.5

    X[y == 0, 0] += -1.0
    X[y == 1, 0] += 0.5
    X[y == 2, 0] += 1.5

    X[y == 0, 1] += -0.8
    X[y == 1, 1] += 0.3
    X[y == 2, 1] += 1.2

    for i in range(2, 6):
        X[y == 2, i] += -0.6

    X[y == 2, 6] += -0.7

    feature_names = []
    feature_names.append("APACHE_II")
    feature_names.append("SOFA")
    feature_names.append("Age")
    feature_names.append("Gender")

    brain_regions = [
        "Hippocampus_L", "Hippocampus_R", "Amygdala_L", "Amygdala_R",
        "Thalamus_L", "Thalamus_R", "Pallidum_L", "Pallidum_R",
        "Cerebellum_Cortex_L", "Cerebellum_Cortex_R", "Ventricle_Lat_L", "Ventricle_Lat_R",
        "Cingulum_Post_L", "Cingulum_Post_R", "Entorhinal_L", "Entorhinal_R",
        "Orbitofrontal_Lat_L", "Orbitofrontal_Lat_R", "Pars_orbitalis_L", "Pars_orbitalis_R",
        "Supramarginal_L", "Supramarginal_R"
    ]

    for i in range(n_features - 4):
        if i < len(brain_regions):
            feature_names.append(brain_regions[i])
        else:
            feature_names.append(f"FreeSurfer_Feature_{i + 1}")

    df = pd.DataFrame(X, columns=feature_names)
    df['Group'] = y

    if save_path:
        df.to_excel(save_path, index=False)
        print(f"模拟数据已保存至: {save_path}")

    return df


# ======================== 数据加载函数 ========================
def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"数据文件 '{filepath}' 未找到，正在生成模拟数据用于演示...")
        df = generate_simulated_data(save_path=filepath)
    else:
        print(f"加载数据文件: {filepath}")
        df = pd.read_excel(filepath, engine='openpyxl')

    print(f"数据维度: {df.shape}")
    print("类别分布:")
    print(df.iloc[:, -1].value_counts().sort_index())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()

    unique_classes = np.unique(y)
    if len(unique_classes) != 3:
        raise ValueError(f"必须是三分类数据！检测到类别数: {len(unique_classes)}")

    return X, y, unique_classes, feature_names, df


# ======================== 1. LASSO特征选择 ========================
def perform_lasso_feature_selection(X, y, feature_names, cv=10):
    print("\n" + "=" * 50)
    print("1. LASSO特征选择")
    print("=" * 50)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = np.logspace(-4, 1, 100)
    lasso_cv = LassoCV(alphas=alphas, cv=cv, max_iter=10000, n_jobs=-1, random_state=RANDOM_STATE)
    lasso_cv.fit(X_scaled, y)

    optimal_lambda = lasso_cv.alpha_
    selected_mask = lasso_cv.coef_ != 0
    selected_features = np.array(feature_names)[selected_mask].tolist()
    selected_idx = np.where(selected_mask)[0]

    print(f"最优 lambda: {optimal_lambda:.4f}")
    print(f"筛选出的特征数: {len(selected_features)}")

    mse_mean = np.mean(lasso_cv.mse_path_, axis=1)
    mse_std = np.std(lasso_cv.mse_path_, axis=1)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(alphas, mse_mean, 'b-', linewidth=1.5, label='Mean MSE')
    plt.fill_between(alphas, mse_mean - mse_std, mse_mean + mse_std,
                     alpha=0.2, color='b')
    plt.axvline(optimal_lambda, color='red', linestyle='--', linewidth=2,
                label=f'Optimal $\lambda$ ($\lambda_{{1se}}$): {optimal_lambda:.3f}')
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.title('LASSO Cross-Validation Error', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}Figure2A_lasso_cv.tiff", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: Figure2A_lasso_cv.tiff")

    _, coefs, _ = lasso_path(X_scaled, y, alphas=alphas)

    plt.figure(figsize=(12, 7), dpi=300)
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(selected_idx), 20)))
    for i, idx in enumerate(selected_idx):
        plt.plot(alphas, coefs[idx], color=colors[i % len(colors)],
                 linewidth=1.8, alpha=0.8)

    plt.axvline(optimal_lambda, color='red', linestyle='--', linewidth=2.5,
                label=f'Optimal $\lambda$: {optimal_lambda:.3f}')
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.title('LASSO Coefficient Paths (Non-zero Features)', fontsize=14)
    plt.grid(True, alpha=0.15)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}Figure2B_lasso_path.tiff", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: Figure2B_lasso_path.tiff")

    results_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': lasso_cv.coef_[selected_idx]
    })
    results_df['Absolute_Coefficient'] = np.abs(results_df['Coefficient'])
    results_df = results_df.sort_values('Absolute_Coefficient', ascending=False)

    results_df.to_excel(f"{OUTPUT_DIR}Lasso_Feature_Selection_Results.xlsx", index=False)
    print(f"已保存: Lasso_Feature_Selection_Results.xlsx")

    return selected_idx, selected_features, scaler


# ======================== 2. 模型定义 ========================
def get_models():
    models = {
        "Stepwise Logistic": {"type": "special"},
        "SVM": {"model": OneVsRestClassifier(SVC(probability=True, random_state=RANDOM_STATE))},
        "GBRT": {"model": GradientBoostingClassifier(random_state=RANDOM_STATE)},
        "XGBoost": {"model": XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')},
        "Random Forest": {"model": RandomForestClassifier(random_state=RANDOM_STATE)},
        "KNN": {"model": KNeighborsClassifier()},
        "Decision Tree": {"model": DecisionTreeClassifier(random_state=RANDOM_STATE)},
        "Naive Bayes": {"model": GaussianNB()}
    }

    param_grids = {
        "SVM": {'estimator__C': [0.1, 1, 10], 'estimator__gamma': ['scale', 'auto']},
        "GBRT": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
        "XGBoost": {'max_depth': [3, 5], 'subsample': [0.8, 1.0], 'reg_lambda': [0.1, 1.0]},
        "Random Forest": {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]},
        "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        "Decision Tree": {'max_depth': [3, 5, None], 'min_samples_split': [2, 5]}
    }

    return models, param_grids


def stepwise_logistic_selection(X_train, y_train, cv=3):
    lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=RANDOM_STATE)
    sfs = SequentialFeatureSelector(
        lr, n_features_to_select='auto', direction='forward',
        cv=cv, scoring='accuracy'
    )
    sfs.fit(X_train, y_train)
    lr.fit(X_train[:, sfs.get_support()], y_train)
    return {'model': lr, 'selected_features': sfs.get_support()}


# ======================== 3. 评估指标计算 ========================
def calculate_specificity(y_true, y_pred, classes):
    specificities = []
    for cls in classes:
        tn = np.sum((y_true != cls) & (y_pred != cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    return np.nanmean(specificities)


def calculate_ppv_npv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    ppvs, npvs = [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        ppvs.append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)
        npvs.append(tn / (tn + fn) if (tn + fn) > 0 else np.nan)

    return np.nanmean(ppvs), np.nanmean(npvs)


def bootstrap_auc_ci(model, X_test, y_test, classes, n_bootstrap=1000, alpha=0.95):
    if not hasattr(model, "predict_proba"):
        return np.nan, np.nan, np.nan

    y_test_bin = label_binarize(y_test, classes=classes)
    y_proba = model.predict_proba(X_test)

    aucs = []
    n = len(y_test)

    for _ in range(n_bootstrap):
        idx = resample(np.arange(n), replace=True, n_samples=n)
        if len(np.unique(y_test[idx])) < len(classes):
            continue
        try:
            auc = roc_auc_score(y_test_bin[idx], y_proba[idx],
                                multi_class='ovr', average='macro')
            aucs.append(auc)
        except:
            continue

    auc_mean = np.mean(aucs)
    ci_lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
    ci_upper = np.percentile(aucs, (1 + alpha) / 2 * 100)

    return auc_mean, ci_lower, ci_upper


def evaluate_model(model, X_test, y_test, classes, selected_features=None):
    if selected_features is not None:
        X_test = X_test[:, selected_features]

    y_pred = model.predict(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'Kappa': cohen_kappa_score(y_test, y_pred),
        'Specificity': calculate_specificity(y_test, y_pred, classes),
    }

    ppv, npv = calculate_ppv_npv(y_test, y_pred)
    metrics['PPV'] = ppv
    metrics['NPV'] = npv

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=classes)
            auc_point = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
            auc_mean, ci_lower, ci_upper = bootstrap_auc_ci(model, X_test, y_test, classes, n_bootstrap=N_BOOTSTRAP)

            metrics['AUC'] = auc_point
            metrics['AUC_CI_Lower'] = ci_lower
            metrics['AUC_CI_Upper'] = ci_upper
        except Exception as e:
            print(f"  AUC计算失败: {e}")
            metrics['AUC'] = np.nan
            metrics['AUC_CI_Lower'] = np.nan
            metrics['AUC_CI_Upper'] = np.nan
    else:
        metrics['AUC'] = np.nan
        metrics['AUC_CI_Lower'] = np.nan
        metrics['AUC_CI_Upper'] = np.nan

    return metrics, y_pred


# ======================== 4. 可视化函数 ========================
def plot_roc_curves(trained_models, X_test, y_test, classes, results_df, output_path):
    plt.figure(figsize=(6, 6), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, len(trained_models)))

    y_test_bin = label_binarize(y_test, classes=classes)

    for (name, model_data), color in zip(trained_models.items(), colors):
        if name not in results_df.index or pd.isna(results_df.loc[name, 'AUC']):
            continue

        if 'selected_features' in model_data:
            model = model_data['model']
            X = X_test[:, model_data['selected_features']]
        else:
            model = model_data['model']
            X = X_test

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
            plt.plot(fpr, tpr, color=color, lw=1.2,
                     label=f'{name} (AUC={results_df.loc[name, "AUC"]:.3f})',
                     alpha=0.85)

    plt.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('ROC Curves for Eight Candidate Models', fontsize=12)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_confusion_matrices(trained_models, X_test, y_test, classes, output_prefix):
    auc_scores = {}
    for name, model_data in trained_models.items():
        if 'selected_features' in model_data:
            model = model_data['model']
            X = X_test[:, model_data['selected_features']]
        else:
            model = model_data['model']
            X = X_test

        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
                y_test_bin = label_binarize(y_test, classes=classes)
                auc_scores[name] = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
            except:
                auc_scores[name] = 0
        else:
            auc_scores[name] = 0

    top_models = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_model_names = [m[0] for m in top_models]

    for name in top_model_names:
        model_data = trained_models[name]
        if 'selected_features' in model_data:
            model = model_data['model']
            X = X_test[:, model_data['selected_features']]
        else:
            model = model_data['model']
            X = X_test

        y_pred = model.predict(X)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(4, 4), dpi=150)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Control', 'non-SAE', 'SAE'],
                    yticklabels=['Control', 'non-SAE', 'SAE'],
                    cbar_kws={'shrink': 0.8})
        plt.title(f'{name} (AUC={auc_scores[name]:.3f})', fontsize=11)
        plt.xlabel('Predicted', fontsize=9)
        plt.ylabel('Actual', fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{name.lower().replace(' ', '_')}.tiff",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_prefix}_{name.lower().replace(' ', '_')}.tiff")


# ======================== 5. SHAP分析 ========================
def perform_shap_analysis(model, X_train, feature_names, output_prefix):
    print("\n" + "=" * 50)
    print("5. SHAP特征重要性分析")
    print("=" * 50)

    if not isinstance(feature_names, list):
        feature_names = list(feature_names)

    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[-15:][::-1]

    top_features = np.array(feature_names)[sorted_idx]
    top_importance = importance[sorted_idx]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.barh(range(len(top_features)), top_importance, color='#1f77b4')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance (Gain)', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.title('Top 15 Feature Importance (XGBoost)', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_feature_importance.tiff", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_prefix}_feature_importance.tiff")

    try:
        X_sample = X_train[:min(100, len(X_train))]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values_combined = np.zeros_like(shap_values[0])
            for sv in shap_values:
                shap_values_combined += np.abs(sv)
            shap_values = shap_values_combined / len(shap_values)

        plt.figure(figsize=(10, 8), dpi=300)
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_shap_summary.tiff", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_prefix}_shap_summary.tiff")

    except Exception as e:
        print(f"SHAP分析出错: {e}")
        print("跳过SHAP可视化，仅保留特征重要性图。")

    return top_features.tolist()


# ======================== 6. 敏感性分析 ========================
def sensitivity_analysis_without_clinical_scores(X_train, X_test, y_train, y_test,
                                                 feature_names, classes, output_path):
    print("\n" + "=" * 50)
    print("6. 敏感性分析（排除APACHE II和SOFA）")
    print("=" * 50)

    exclude_patterns = ['APACHE', 'SOFA']
    exclude_idx = []
    for i, name in enumerate(feature_names):
        if any(pattern in name.upper() for pattern in exclude_patterns):
            exclude_idx.append(i)

    print(f"排除的特征: {[feature_names[i] for i in exclude_idx]}")

    keep_idx = [i for i in range(len(feature_names)) if i not in exclude_idx]

    X_train_reduced = X_train[:, keep_idx]
    X_test_reduced = X_test[:, keep_idx]

    xgb_reduced = XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')
    xgb_reduced.fit(X_train_reduced, y_train)

    y_proba = xgb_reduced.predict_proba(X_test_reduced)
    y_test_bin = label_binarize(y_test, classes=classes)

    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    auc_score = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')

    plt.figure(figsize=(6, 6), dpi=300)

    plt.plot(fpr, tpr, 'b-', lw=1.5, label=f'Neuroimaging Only (AUC={auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('ROC Curve Excluding APACHE II and SOFA Scores', fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"仅神经影像特征的AUC: {auc_score:.3f}")
    print(f"已保存: {output_path}")

    return auc_score


# ======================== 主程序 ========================
def main():
    print("\n" + "=" * 60)
    print("SAE Diagnostic Model Development")
    print("Machine Learning Models for Sepsis-Associated Encephalopathy")
    print("=" * 60)

    X, y, classes, feature_names, df = load_data(DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")

    selected_idx, selected_features, lasso_scaler = perform_lasso_feature_selection(
        X_train, y_train, feature_names, cv=N_FOLDS_CV
    )

    X_train_selected = X_train[:, selected_idx]
    X_test_selected = X_test[:, selected_idx]
    selected_feature_names = [feature_names[i] for i in selected_idx]

    print(f"\n使用LASSO筛选后的 {len(selected_idx)} 个特征进行模型训练")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    print("\n" + "=" * 50)
    print("3. 多模型训练与评估")
    print("=" * 50)

    base_models, param_grids = get_models()
    results = {}
    trained_models = {}

    print("\n训练 Stepwise Logistic Regression...")
    try:
        lr_result = stepwise_logistic_selection(X_train_scaled, y_train, cv=3)
        trained_models["Stepwise Logistic"] = lr_result
        metrics, _ = evaluate_model(
            lr_result['model'], X_test_scaled, y_test, classes,
            lr_result['selected_features']
        )
        results["Stepwise Logistic"] = metrics
        print(
            f"  完成 - AUC: {metrics['AUC']:.3f} (95% CI: {metrics['AUC_CI_Lower']:.3f}-{metrics['AUC_CI_Upper']:.3f})")
    except Exception as e:
        print(f"  训练失败: {e}")

    for name, model_info in base_models.items():
        if name == "Stepwise Logistic":
            continue

        print(f"\n训练 {name}...")
        start_time = time.time()

        try:
            if name in param_grids:
                grid = GridSearchCV(
                    model_info['model'], param_grids[name],
                    cv=StratifiedKFold(N_FOLDS_GRID, shuffle=True, random_state=RANDOM_STATE),
                    scoring='accuracy', n_jobs=-1
                )
                grid.fit(X_train_scaled, y_train)
                best_model = grid.best_estimator_
                print(f"  最佳参数: {grid.best_params_}")
            else:
                best_model = model_info['model'].fit(X_train_scaled, y_train)

            trained_models[name] = {'model': best_model}
            metrics, _ = evaluate_model(best_model, X_test_scaled, y_test, classes)
            results[name] = metrics
            print(
                f"  完成 - AUC: {metrics['AUC']:.3f} (95% CI: {metrics['AUC_CI_Lower']:.3f}-{metrics['AUC_CI_Upper']:.3f})")

        except Exception as e:
            print(f"  训练失败: {e}")
            results[name] = {k: np.nan for k in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1',
                                                 'Kappa', 'Specificity', 'PPV', 'NPV',
                                                 'AUC_CI_Lower', 'AUC_CI_Upper']}

    results_df = pd.DataFrame(results).T
    col_order = ['AUC', 'AUC_CI_Lower', 'AUC_CI_Upper', 'Accuracy', 'Precision', 'Recall', 'F1',
                 'Kappa', 'Specificity', 'PPV', 'NPV']
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    print("\n" + "=" * 50)
    print("模型性能对比 (Table 2)")
    print("=" * 50)
    display_df = results_df.copy()
    display_df['AUC (95% CI)'] = display_df.apply(
        lambda x: f"{x['AUC']:.3f} ({x['AUC_CI_Lower']:.3f}-{x['AUC_CI_Upper']:.3f})", axis=1
    )
    display_cols = ['AUC (95% CI)', 'Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'Specificity', 'PPV', 'NPV']
    print(display_df[display_cols].round(3).sort_values('AUC (95% CI)', ascending=False))

    results_df.to_excel(f"{OUTPUT_DIR}Table2_model_performance.xlsx", float_format='%.3f')
    print(f"\n已保存: Table2_model_performance.xlsx")

    print("\n" + "=" * 50)
    print("4. 生成可视化图表")
    print("=" * 50)

    plot_roc_curves(trained_models, X_test_scaled, y_test, classes, results_df,
                    f"{OUTPUT_DIR}Figure3_roc_curves.tiff")

    plot_confusion_matrices(trained_models, X_test_scaled, y_test, classes,
                            f"{OUTPUT_DIR}Figure4_cm")

    if "XGBoost" in trained_models:
        xgb_model = trained_models["XGBoost"]['model']
        perform_shap_analysis(xgb_model, X_train_scaled, selected_feature_names,
                              f"{OUTPUT_DIR}Figure5")

    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train)
    X_test_full_scaled = scaler_full.transform(X_test)

    sensitivity_analysis_without_clinical_scores(
        X_train_full_scaled, X_test_full_scaled, y_train, y_test,
        feature_names, classes,
        f"{OUTPUT_DIR}Figure6_sensitivity_roc.tiff"
    )

    print("\n" + "=" * 50)
    print("分析完成")
    print("=" * 50)
    print("生成的文件:")
    print("  - Figure2A_lasso_cv.tiff")
    print("  - Figure2B_lasso_path.tiff")
    print("  - Lasso_Feature_Selection_Results.xlsx")
    print("  - Table2_model_performance.xlsx")
    print("  - Figure3_roc_curves.tiff")
    print("  - Figure4_cm_*.tiff")
    print("  - Figure5_feature_importance.tiff")
    print("  - Figure5_shap_summary.tiff")
    print("  - Figure6_sensitivity_roc.tiff")


if __name__ == "__main__":
    main()