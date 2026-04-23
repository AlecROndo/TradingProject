import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from models import build_dataset, data, btcdata
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

X, y = build_dataset(data, btcdata)

ALL_EVENTS = [
    "KXBTCD-26APR0216", "KXBTCD-26APR0321", "KXBTCD-26APR0421",
    "KXBTCD-26APR0521", "KXBTCD-26APR0621", "KXBTCD-26APR0716",
    "KXBTCD-26APR0816", "KXBTCD-26FEB1121", "KXBTCD-26FEB1316", "KXBTCD-26FEB1821",
    "KXBTCD-26FEB2016", "KXBTCD-26FEB2521", "KXBTCD-26FEB2716", "KXBTCD-26MAR0421",
    "KXBTCD-26MAR0616", "KXBTCD-26MAR1121", "KXBTCD-26MAR1321", "KXBTCD-26MAR1621", "KXBTCD-26MAR2021",
]

def LinearReg(X, y):
    kf = KFold(n_splits=7, shuffle=False)
    fold_rmses = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(ALL_EVENTS)):
        train_events = [ALL_EVENTS[i] for i in train_idx]
        test_events  = [ALL_EVENTS[i] for i in test_idx]

        X_train, y_train = build_dataset({e: data[e] for e in train_events}, btcdata)
        X_test,  y_test  = build_dataset({e: data[e] for e in test_events},  btcdata)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = Lasso(alpha=0.01)
        model.fit(X_train, y_train)

        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        test_rmse  = np.sqrt(mean_squared_error(y_test,  model.predict(X_test)))
        fold_rmses.append(test_rmse)


    print(f"\n  Mean Test RMSE across folds: {np.mean(fold_rmses):.4f}")
    return np.mean(fold_rmses)


def PolyReg(X, y, deg):
    kf = KFold(n_splits=7, shuffle=False)
    fold_rmses = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(ALL_EVENTS)):
        train_events = [ALL_EVENTS[i] for i in train_idx]
        test_events  = [ALL_EVENTS[i] for i in test_idx]

        X_train, y_train = build_dataset({e: data[e] for e in train_events}, btcdata)
        X_test,  y_test  = build_dataset({e: data[e] for e in test_events},  btcdata)

        model = Pipeline([
            ("poly",   PolynomialFeatures(degree=deg)),
            ("scaler", StandardScaler()),
            ("lasso",  Lasso(alpha=0.01))
        ])
        model.fit(X_train, y_train)

        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        test_rmse  = np.sqrt(mean_squared_error(y_test,  model.predict(X_test)))
        fold_rmses.append(test_rmse)



    return np.mean(fold_rmses)



def XGBReg(X, y):
    kf = KFold(n_splits=7, shuffle=False)
    fold_rmses = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(ALL_EVENTS)):
        train_events = [ALL_EVENTS[i] for i in train_idx]
        test_events  = [ALL_EVENTS[i] for i in test_idx]

        X_train, y_train = build_dataset({e: data[e] for e in train_events}, btcdata)
        X_test,  y_test  = build_dataset({e: data[e] for e in test_events},  btcdata)

        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        model.fit(X_train, y_train)

        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        test_rmse  = np.sqrt(mean_squared_error(y_test,  model.predict(X_test)))
        fold_rmses.append(test_rmse)



    return np.mean(fold_rmses)


def LGBMReg(X, y):
    kf = KFold(n_splits=7, shuffle=False)
    fold_rmses = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(ALL_EVENTS)):
        train_events = [ALL_EVENTS[i] for i in train_idx]
        test_events  = [ALL_EVENTS[i] for i in test_idx]

        X_train, y_train = build_dataset({e: data[e] for e in train_events}, btcdata)
        X_test,  y_test  = build_dataset({e: data[e] for e in test_events},  btcdata)

        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        test_rmse  = np.sqrt(mean_squared_error(y_test,  model.predict(X_test)))
        fold_rmses.append(test_rmse)
    return np.mean(fold_rmses)





lin = LinearReg(X, y)
poly = PolyReg(X, y, 3)
xgbr = XGBReg(X, y)
lgbm = LGBMReg(X, y)

results = {
    "Lasso (Linear)": lin,
    "Lasso (Poly-3)": poly,
    "XGBoost":        xgbr,
    "LightGBM":       lgbm,
}

fig, ax = plt.subplots(figsize=(8, 5))

colors = ["#378ADD", "#1D9E75", "#D85A30", "#BA7517"]
bars = ax.bar(results.keys(), results.values(), color=colors, width=0.5, zorder=2)

ax.set_ylabel("Mean Test RMSE", fontsize=12)
ax.set_title("Cross-validated RMSE by model", fontsize=14)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
ax.spines[["top", "right"]].set_visible(False)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.0005,
        f"{height:.4f}",
        ha="center", va="bottom", fontsize=10
    )

plt.tight_layout()
plt.savefig("rmse_by_model.png", dpi=150)
plt.show()