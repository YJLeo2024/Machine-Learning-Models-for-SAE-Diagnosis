# Machine Learning Models for SAE Diagnosis

This repository contains the source code for the manuscript:

**"Automated Whole-Brain MRI Segmentation Combined with Clinical Scores for the Diagnosis of Sepsis-Associated Encephalopathy: A Case-Control Study"**

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
Usage
Prepare Data
Place your data file as data.xlsx in the repository root directory. The file should contain features as columns and a group label column (0=Control, 1=non-SAE, 2=SAE) as the last column.
Note: Due to patient privacy restrictions, the original dataset cannot be publicly shared. If data.xlsx is not found, the script will automatically generate simulated data for demonstration.
Run Analysis
python SAE_Model_Development.py
Outputs
The script generates the following files:
File	Description
Figure2A_lasso_cv.tiff	LASSO cross-validation error curve
Figure2B_lasso_path.tiff	LASSO coefficient path plot
Figure3_roc_curves.tiff	ROC curves for 8 candidate models
Figure4_cm_*.tiff	Confusion matrices for top 3 models
Figure5_feature_importance.tiff	Top 15 feature importance (XGBoost)
Figure5_shap_summary.tiff	SHAP summary plot
Figure6_sensitivity_roc.tiff	ROC curve excluding APACHE II and SOFA
Table2_model_performance.xlsx	Performance metrics for all models
Lasso_Feature_Selection_Results.xlsx	LASSO-selected features and coefficients
License
This code is provided for research reproducibility purposes.
Contact
For questions regarding the code or data access, please contact the corresponding authors.
