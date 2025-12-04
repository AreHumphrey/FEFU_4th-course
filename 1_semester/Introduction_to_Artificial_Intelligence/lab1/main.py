import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def solve_linear_regression_by_hand(features, targets):
    n_samples = features.shape[0]
    X_with_bias = np.hstack([np.ones((n_samples, 1)), features])
    weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ targets
    return weights


def predict_with_weights(features, weights):
    n_samples = features.shape[0]
    X_with_bias = np.hstack([np.ones((n_samples, 1)), features])
    return X_with_bias @ weights


def calculate_entropy(labels):
    if len(labels) == 0:
        return 0.0
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy


def find_best_threshold_for_split(feature_values, labels):
    best_threshold = None
    best_gain = -1

    for threshold in np.unique(feature_values):
        left_mask = feature_values <= threshold
        left_labels = labels[left_mask]
        right_labels = labels[~left_mask]

        gain = (
            calculate_entropy(labels)
            - (len(left_labels) / len(labels)) * calculate_entropy(left_labels)
            - (len(right_labels) / len(labels)) * calculate_entropy(right_labels)
        )

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def main():
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Линейная регрессия": LinearRegression(),
        "Дерево решений": DecisionTreeRegressor(random_state=42),
        "Случайный лес": RandomForestRegressor(n_estimators=50, random_state=42),
        "Ближайшие соседи": KNeighborsRegressor(n_neighbors=5)  # ← Чёткое название
    }

    results = {}
    for name, model in models.items():

        if name == "Ближайшие соседи":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2,
            "predictions": y_pred
        }

        print(f"{name:25} MAE: {mae:.3f},  MSE: {mse:.3f},  R^2: {r2:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        axes[idx].scatter(y_test, res["predictions"], alpha=0.6, edgecolors='k', linewidth=0.3)
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1.5)
        axes[idx].set_xlabel("Истинные значения")
        axes[idx].set_ylabel("Предсказанные значения")
        axes[idx].set_title(f"{name}\nR^2 = {res['R2']:.3f}")

    plt.tight_layout()
    plt.savefig("результаты_моделей.png", dpi=150)

    print("\nРучная линейная регрессия")
    X_train_small = X_train[:, :2]
    X_test_small = X_test[:, :2]

    weights = solve_linear_regression_by_hand(X_train_small, y_train)
    y_pred_manual = predict_with_weights(X_test_small, weights)
    mae_manual = mean_absolute_error(y_test, y_pred_manual)
    print(f"MAE ручной модели: {mae_manual:.3f}")

    print("\nПостроение разбиения по энтропии")
    np.random.seed(42)
    synthetic_feature = np.random.rand(20)
    synthetic_labels = (synthetic_feature > 0.5).astype(int)

    threshold, gain = find_best_threshold_for_split(synthetic_feature, synthetic_labels)
    print(f"Лучший порог: {threshold:.3f}, Информационный выигрыш: {gain:.3f}")


if __name__ == "__main__":
    main()
