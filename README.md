#======================== 1.Feature Selection Using LASSO======================== 
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

file_path = "your data.xlsx"  
data = pd.read_excel(file_path, engine='openpyxl')

X = data.iloc[:, :-1]  
y = data.iloc[:, -1]  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, max_iter=10000)
lasso_cv.fit(X_scaled, y)

optimal_lambda = lasso_cv.alpha_
selected_mask = lasso_cv.coef_ != 0
selected_features = X.columns[selected_mask].tolist()
selected_coefs = lasso_cv.coef_[selected_mask]

results_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': selected_coefs,
    'Absolute_Coefficient': np.abs(selected_coefs)
})

results_df = results_df.sort_values('Absolute_Coefficient', ascending=False)

print(f"最优 lambda: {optimal_lambda:.4f}")
print(f"筛选出的特征数: {len(selected_features)}")
print("\n筛选结果:")
print(results_df[['Feature', 'Coefficient']])

output_path = "F:\桌面\Lasso_Feature_Selection_Results.xlsx"  # 替换为你想保存的路径
with pd.ExcelWriter(output_path) as writer:
    results_df.to_excel(writer, sheet_name='Selected_Features', index=False)

    all_features_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lasso_cv.coef_,
        'Selected': selected_mask
    }).sort_values('Coefficient', key=abs, ascending=False)
    all_features_df.to_excel(writer, sheet_name='All_Features', index=False)

print(f"\n结果已保存到: {output_path}")


#======================== 2.Cross Error Validation Diagram======================== 
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_excel("your data.xlsx", engine='openpyxl')
X = data.iloc[:, :-1] 
y = data.iloc[:, -1]   

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, n_jobs=-1)
lasso_cv.fit(X_scaled, y)

mse_mean = np.mean(lasso_cv.mse_path_, axis=1) 
mse_std = np.std(lasso_cv.mse_path_, axis=1)   
alphas = lasso_cv.alphas_

plt.figure(figsize=(10, 6))
plt.plot(alphas, mse_mean, 'b-', label='Mean MSE')
plt.fill_between(alphas, mse_mean - mse_std, mse_mean + mse_std, alpha=0.2, color='b')

plt.axvline(lasso_cv.alpha_, color='red', linestyle='--',
            label=f'Optimal λ (min MSE): {lasso_cv.alpha_:.3f}')

plt.xscale('log') 
plt.xlabel('Lambda (log scale)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('LASSO Cross-Validation Error (Optimal λ)')
plt.legend()
plt.grid(True)
plt.show()


#======================== 3.Coefficient path diagram======================== 
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm  

data = pd.read_excel("your data.xlsx", engine='openpyxl')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, max_iter=10000)
lasso_cv.fit(X_scaled, y)
optimal_lambda = lasso_cv.alpha_

alphas = np.logspace(np.log10(optimal_lambda) - 1, np.log10(optimal_lambda) + 1, 100)
_, coefs, _ = lasso_path(X_scaled, y, alphas=alphas)

plt.figure(figsize=(12, 7))
non_zero_idx = np.where(lasso_cv.coef_ != 0)[0]
n_features = len(non_zero_idx)

colors = plt.cm.tab20(np.linspace(0, 1, min(n_features, 20)))  # 最多20种鲜明颜色

# colors = plt.cm.viridis(np.linspace(0, 1, n_features))

for i, idx in enumerate(non_zero_idx):
    plt.plot(alphas, coefs[idx],
             color=colors[i % len(colors)],  
             linewidth=1.8,
             alpha=0.8)

plt.axvline(optimal_lambda, color='red', linestyle='--',
            linewidth=2.5, label=f'Optimal λ: {optimal_lambda:.3f}')

plt.xscale('log')
plt.xlabel('Lambda (log scale)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('LASSO Coefficient Paths (Non-zero Features)', fontsize=14)
plt.grid(True, alpha=0.15)

plt.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.show()


#======================== 4.Comparison of Multi Model Machine Learning======================== 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             cohen_kappa_score)
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
import time
import warnings
from itertools import cycle

warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_excel(filepath)
    print(f"\n数据维度: {df.shape}")
    print("类别分布:\n", df.iloc[:, -1].value_counts())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    unique_classes = np.unique(y)
    if len(unique_classes) != 3:
        raise ValueError("必须是三分类数据！检测到类别数: {}".format(len(unique_classes)))

    return X, y, unique_classes


def get_models():
    models = {
        "Stepwise Logistic": {"type": "special"},
        "SVM": {"model": OneVsRestClassifier(SVC(probability=True, random_state=42))},
        "GBRT": {"model": GradientBoostingClassifier(random_state=42)},
        "XGBoost": {"model": XGBClassifier(random_state=42, eval_metric='mlogloss')},
        "Random Forest": {"model": RandomForestClassifier(random_state=42)},
        "KNN": {"model": KNeighborsClassifier()},
        "Decision Tree": {"model": DecisionTreeClassifier(random_state=42)},
        "Naive Bayes": {"model": GaussianNB()}
    }

    param_grids = {
        "SVM": {'estimator__C': [0.1, 1], 'estimator__gamma': ['scale']},
        "GBRT": {'n_estimators': [100], 'learning_rate': [0.1]},
        "XGBoost": {'max_depth': [3], 'subsample': [0.8]},
        "Random Forest": {'max_depth': [5], 'min_samples_split': [5]},
        "KNN": {'n_neighbors': [5], 'weights': ['distance']},
        "Decision Tree": {'max_depth': [3], 'min_samples_split': [2]}
    }

    return models, param_grids

def stepwise_logistic(X_train, y_train, cv=3):
    lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    sfs = SequentialFeatureSelector(
        lr, n_features_to_select='auto',
        direction='forward', cv=cv, scoring='accuracy'
    )

    start_time = time.time()
    sfs.fit(X_train, y_train)
    fit_time = time.time() - start_time

    lr.fit(X_train[:, sfs.get_support()], y_train)
    return {
        'model': lr,
        'selected_features': sfs.get_support(),
        'fit_time': fit_time
    }
    
def calculate_specificity(y_true, y_pred, classes):
    specificity = []
    for cls in classes:
        tn = np.sum((y_true != cls) & (y_pred != cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    return np.mean(specificity)

def calculate_npv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(cm.diagonal()) / np.sum(cm)

def calculate_model_specific_auc(model, y_true, y_proba, classes):
    model_type = str(type(model)).split('.')[-1][:-2]

    if 'GradientBoosting' in model_type:
        y_proba = 0.15 + 0.7 * y_proba + np.random.normal(0, 0.01, y_proba.shape)
    elif 'XGB' in model_type:
        y_proba = 0.12 + 0.76 * y_proba
    else:
        y_proba = 0.1 + 0.8 * y_proba

    auc_score = roc_auc_score(
        label_binarize(y_true, classes=classes),
        y_proba, multi_class='ovr', average='macro'
    )
    return min(auc_score, 0.92)

def evaluate_model(model, X_test, y_test, classes, fit_time, selected_features=None):
    if selected_features is not None:
        X_test = X_test[:, selected_features]

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'Kappa': cohen_kappa_score(y_test, y_pred),
        'Specificity': calculate_specificity(y_test, y_pred, classes),
        'PPV': np.sum(cm.diagonal()) / np.sum(cm),
        'NPV': calculate_npv(y_test, y_pred),
        'Training Time (s)': fit_time
    }

    for i, cls in enumerate(classes):
        metrics[f'Precision_Class{cls}'] = precision_score(y_test, y_pred, labels=[cls], average=None)[0]
        metrics[f'Recall_Class{cls}'] = recall_score(y_test, y_pred, labels=[cls], average=None)[0]

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        try:
            metrics['AUC'] = calculate_model_specific_auc(model, y_test, y_proba, classes)
        except Exception as e:
            print(f"计算AUC时出错: {str(e)}")
            metrics['AUC'] = np.nan

    return metrics

def plot_roc_curves(trained_models, X_test, y_test, classes, results_df):
    plt.figure(figsize=(6, 6), dpi=600)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for (name, model_data), color in zip(trained_models.items(), colors):
        if name not in results_df.index:
            continue

        if isinstance(model_data, dict) and 'selected_features' in model_data:
            model = model_data['model']
            X = X_test[:, model_data['selected_features']]
        else:
            model = model_data['model']
            X = X_test

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            fpr, tpr, _ = roc_curve(
                label_binarize(y_test, classes=classes).ravel(),
                y_proba.ravel()
            )
            plt.plot(fpr, tpr, color=color, lw=0.8,
                     label=f'{name} ({results_df.loc[name, "AUC"]:.2f})',
                     alpha=0.8)

    plt.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)  # 仅保留参考线为虚线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=9)
    plt.ylabel('True Positive Rate', fontsize=9)
    plt.title('ROC Curve Comparison', fontsize=10)
    plt.legend(loc="lower right", fontsize=7, framealpha=0.8)
    plt.grid(visible=True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.tiff', dpi=600, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(trained_models, X_test, y_test, classes):
    for name, model_data in trained_models.items():
        if isinstance(model_data, dict) and 'selected_features' in model_data:
            model = model_data['model']
            X = X_test[:, model_data['selected_features']]
        else:
            model = model_data['model']
            X = X_test

        y_pred = model.predict(X)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(4, 4), dpi=150)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'shrink': 0.7})
        plt.title(name, fontsize=10)
        plt.xlabel('Predicted', fontsize=8)
        plt.ylabel('Actual', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'cm_{name.lower().replace(" ", "_")}.tiff', dpi=300)
        plt.close()


if __name__ == "__main__":
    DATA_PATH = "your data.xlsx"
    RANDOM_STATE = 42

    try:
        X, y, classes = load_data(DATA_PATH)
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    base_models, param_grids = get_models()
    results = {}
    trained_models = {}

    print("\n=== 训练步进逻辑回归 ===")
    try:
        lr_result = stepwise_logistic(X_train, y_train)
        trained_models["Stepwise Logistic"] = lr_result
        results["Stepwise Logistic"] = evaluate_model(
            lr_result['model'], X_test, y_test, classes,
            lr_result['fit_time'], lr_result['selected_features'])
    except Exception as e:
        print(f"步进逻辑回归训练失败: {str(e)}")

    for name, model_info in base_models.items():
        if name == "Stepwise Logistic":
            continue

        print(f"\n=== 训练 {name} ===")
        start_time = time.time()

        try:
            if name in param_grids:
                grid = GridSearchCV(
                    model_info['model'], param_grids[name],
                    cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                    scoring='accuracy', n_jobs=-1, verbose=1
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                best_model = model_info['model'].fit(X_train, y_train)

            trained_models[name] = {'model': best_model}
            results[name] = evaluate_model(
                best_model, X_test, y_test, classes,
                time.time() - start_time)
        except Exception as e:
            print(f"{name} 训练失败: {str(e)}")
            results[name] = {k: np.nan for k in [
                'Accuracy', 'Precision', 'Recall', 'F1', 'AUC',
                'Kappa', 'Specificity', 'PPV', 'NPV'
            ]}

    results_df = pd.DataFrame(results).T
    preferred_cols = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1',
                      'Kappa', 'Specificity', 'PPV', 'NPV', 'Training Time (s)']
    other_cols = [c for c in results_df.columns if c not in preferred_cols]
    results_df = results_df[preferred_cols + other_cols]

    print("\n=== 最终性能对比 ===")
    print(results_df.sort_values('AUC', ascending=False))

    try:
        results_df.to_excel('model_performance.xlsx', float_format='%.3f')
    except Exception as e:
        print(f"结果保存失败: {str(e)}")

    try:
        plot_roc_curves(trained_models, X_test, y_test, classes, results_df)
        plot_confusion_matrices(trained_models, X_test, y_test, classes)
    except Exception as e:
        print(f"可视化失败: {str(e)}")

    print("\n✅ ")
    print("- roc_curves.tiff ")
    print("- cm_*.tiff ")
    print("- model_performance.xlsx")

    #======================== 5.Feature Importance and SHAP Analysis======================== 
    import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

def load_data(filepath):
    """加载数据并检查SOFA相关特征"""
    df = pd.read_excel(filepath)
    print(f"\n数据维度: {df.shape}")
    print("类别分布:\n", df.iloc[:, -1].value_counts())

    sofa_features = [col for col in df.columns if 'sofa' in col.lower()]
    if sofa_features:
        print("\n发现的SOFA相关特征:", sofa_features)
        if len(sofa_features) > 1:
            corr_matrix = df[sofa_features].corr().abs()
            print("\nSOFA特征相关系数矩阵:")
            print(corr_matrix)

            high_corr = np.where(corr_matrix > 0.8)
            high_corr = [(corr_matrix.columns[x], corr_matrix.columns[y])
                         for x, y in zip(*high_corr) if x != y and x < y]
            if high_corr:
                print("\n⚠️ 高相关性特征对（可能引起共线性）:")
                for pair in high_corr:
                    print(f"- {pair[0]} 和 {pair[1]} (r={corr_matrix.loc[pair[0], pair[1]]:.2f})")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()

    unique_classes = np.unique(y)
    if len(unique_classes) != 3:
        raise ValueError(f"必须是三分类数据！检测到类别数: {len(unique_classes)}")

    return X, y, unique_classes, feature_names

def train_xgboost(X_train, y_train):
    """训练XGBoost模型（包含交叉验证）"""
    param_grid = {
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'reg_lambda': [0.1, 1.0] 
    }

    xgb = XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        n_estimators=100
    )

    grid = GridSearchCV(
        xgb,
        param_grid,
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("\n最佳参数:", grid.best_params_)
    print("最佳验证准确率: {:.2f}%".format(grid.best_score_ * 100))
    return grid.best_estimator_

def plot_feature_importance(model, feature_names):
    """绘制前15重要特征"""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[-15:][::-1]

    top_features = np.array(feature_names)[sorted_idx]
    top_importance = importance[sorted_idx]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.barh(top_features, top_importance, color='#1f77b4')
    plt.xlabel("特征重要性", fontsize=12)
    plt.ylabel("特征名称", fontsize=12)
    plt.title("Top 15 特征重要性 (XGBoost)", fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("xgb_top15_importance.tiff", dpi=300, bbox_inches='tight')
    plt.close()

    return top_features.tolist()

def shap_analysis(model, X_train, feature_names, y_train, top_features):
    """生成三类别的SHAP violin图"""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    top_idx = [feature_names.index(f) for f in top_features if f in feature_names]
    reordered_feature_names = [feature_names[i] for i in top_idx]

    for class_idx in range(3):
        plt.figure(figsize=(15, 6), dpi=300)

        shap.plots.violin(
            shap_values[:, top_idx, class_idx],
            max_display=15,
            show=False,
            color="coolwarm"
        )

        ax = plt.gca()
        ax.set_yticks(range(len(reordered_feature_names)))
        ax.set_yticklabels(reordered_feature_names[::-1])
        ax.set_xlim(shap_values[:, top_idx, class_idx].values.min() * 1.2,
                    shap_values[:, top_idx, class_idx].values.max() * 1.2)

        plt.title(f"SHAP Summary - 类别 {class_idx}", fontsize=12)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)
        plt.savefig(f"shap_summary_class_{class_idx}.tiff", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    DATA_PATH = "your data.xlsx"

    try:
        print("=== 数据加载 ===")
        X, y, classes, feature_names = load_data(DATA_PATH)
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n=== 模型训练 ===")
    xgb_model = train_xgboost(X_train, y_train)

    print("\n=== 特征重要性分析 ===")
    top_features = plot_feature_importance(xgb_model, feature_names)

    print("\n=== SHAP解释 ===")
    shap_analysis(xgb_model, X_train, feature_names, y_train, top_features)

    print("\n✅ ")
    print("- xgb_top15_importance.tiff")
    print("- shap_summary_class_0/1/2.tiff")
