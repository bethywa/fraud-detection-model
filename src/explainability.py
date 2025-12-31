import shap
import numpy as np
import pandas as pd

def compute_shap(model, X_background, X_sample):
    explainer = shap.Explainer(model, X_background)
    shap_values = explainer(X_sample)
    return explainer, shap_values

def shap_global_importance(shap_values, feature_names):
    # Handle extra bias column safely
    values = shap_values.values
    if values.shape[1] > len(feature_names):
        values = values[:, :len(feature_names), :]

    shap_vals_fraud = values[:, :, 1]
    importance = np.abs(shap_vals_fraud).mean(axis=0)

    return pd.DataFrame({
        "feature": feature_names,
        "shap_importance": importance
    }).sort_values(by="shap_importance", ascending=False)
