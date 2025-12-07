"""
interpret.py
- permutation_importance on held-out test set
- PDP + ICE via PartialDependenceDisplay.from_estimator for chosen features
Saves figures to ../figures
"""
import os
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def permutation_importance_plot(pipe, X_test, y_test, n_repeats=30, random_state=42):
    print('Computing permutation importance (may take a moment)...')
    res = permutation_importance(pipe, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    sorted_idx = res.importances_mean.argsort()[::-1]
    feature_names = X_test.columns
    means = res.importances_mean[sorted_idx]
    stds = res.importances_std[sorted_idx]

    top = min(len(feature_names), 10)
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names[sorted_idx][:top], means[:top], xerr=stds[:top])
    plt.xlabel('Permutation importance (mean decrease in score)')
    plt.title('Permutation importance (test set)')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'permutation_importance.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f'Saved permutation importance to {path}')
    return res

def pdp_ice_plots(pipe, X, features):
    # features: list of column names (must be numeric)
    for feat in features:
        print(f'Generating PDP+ICE for {feat}...')
        disp = PartialDependenceDisplay.from_estimator(pipe, X, [feat], kind='both', subsample=500, random_state=0)
        plt.suptitle(f'PDP + ICE: {feat}')
        plt.tight_layout()
        fname = f'pdp_ice_{feat.replace(" ", "_")}.png'
        path = os.path.join(FIG_DIR, fname)
        plt.savefig(path, dpi=200)
        plt.close()
        print(f'Saved {path}')
