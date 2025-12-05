import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples
)
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10})


def plot_clusters(ax, X_2d, labels, title, n_clusters=None):

    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        colors = ['black' if label == -1 else plt.cm.Spectral(label / len(unique_labels))
                  for label in labels]
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=15, alpha=0.7)
        n_noise = np.sum(labels == -1)
        title += f"\n(шум: {n_noise})"
    else:
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='Spectral', s=15, alpha=0.7)
        plt.colorbar(scatter, ax=ax, shrink=0.8)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xticks([])
    ax.set_yticks([])


def find_optimal_kmeans_k(X, k_range=range(2, 11)):
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k, silhouette_scores


def estimate_eps_dbscan(X, k=4, plot=False):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, k])[::-1]

    if plot:
        plt.figure(figsize=(6, 3))
        plt.plot(k_distances, linewidth=1)
        plt.title("k-NN расстояния ")
        plt.xlabel("Точки отсортированные")
        plt.ylabel(f"{k}-е расстояние")
        plt.tight_layout()
        plt.savefig("knn_distances.png", dpi=120)
        plt.close()

    eps = np.percentile(k_distances, 10)
    return eps


def safe_compute_metrics(X, labels, method_name):
    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if n_labels < 2:
        return {"Silhouette": "—", "CH": "—", "DB": "—", "Clusters": n_labels}

    try:
        sil = round(silhouette_score(X, labels), 3)
        ch = round(calinski_harabasz_score(X, labels), 1)
        db = round(davies_bouldin_score(X, labels), 3)
        return {"Silhouette": sil, "CH": ch, "DB": db, "Clusters": n_labels}
    except Exception as e:
        return {"Silhouette": "—", "CH": "—", "DB": "—", "Clusters": n_labels}


def plot_silhouette_analysis(X, k_opt, ax):
    kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(k_opt):
        ith_cluster_silhouette = sample_silhouette[labels == i]
        ith_cluster_silhouette.sort()
        size_cluster_i = ith_cluster_silhouette.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.Spectral(i / k_opt)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette, facecolor=color, edgecolor=color, alpha=0.7)
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Средний: {silhouette_avg:.3f}')
    ax.set_xlabel("Значения силуэта")
    ax.set_ylabel("Кластеры")
    ax.set_title(f"Анализ силуэта (k={k_opt})")
    ax.legend(loc="upper right")


def main():
    data = fetch_california_housing()
    X = data.data
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    print(f"Данные: {X.shape[0]} объектов, {X.shape[1]} признаков ")

    k_opt, sil_scores = find_optimal_kmeans_k(X_scaled)
    print(f"   Оптимальное k = {k_opt}")

    kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    agg = AgglomerativeClustering(n_clusters=k_opt)
    labels_agg = agg.fit_predict(X_scaled)

    X_sample = X_scaled[:1000]
    linkage_matrix = linkage(X_sample, method='ward')

    eps = estimate_eps_dbscan(X_scaled, k=4, plot=True)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_scaled)
    n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    print(f"   Найдено {n_clusters_db} кластеров, eps = {eps:.3f}")

    methods = {
        "K-средних": labels_kmeans,
        "Иерархическая": labels_agg,
        "DBSCAN": labels_dbscan
    }
    print(f"{'Метод':<18} | {'Кластеров':<10} | {'Silhouette ↑':<12} | {'CH ':<12} | {'DB ':<10}")
    print("-" * 78)
    for name, labels in methods.items():
        m = safe_compute_metrics(X_scaled, labels, name)
        clusters = m["Clusters"]
        sil = m["Silhouette"]
        ch = m["CH"]
        db = m["DB"]
        print(f"{name:<18} | {clusters:<10} | {sil:<12} | {ch:<12} | {db:<10}")

    fig = plt.figure(figsize=(16, 10))

    plot_clusters(fig.add_subplot(2, 3, 1), X_2d, labels_kmeans, "K-средних")
    plot_clusters(fig.add_subplot(2, 3, 2), X_2d, labels_agg, "Иерархическая")
    plot_clusters(fig.add_subplot(2, 3, 3), X_2d, labels_dbscan, "DBSCAN")

    ax_dendro = fig.add_subplot(2, 3, 4)
    dendrogram(linkage_matrix, ax=ax_dendro, no_labels=True, color_threshold=0)
    ax_dendro.set_title("Дендрограмма 1000 объектов")
    ax_dendro.set_xlabel("Объекты")
    ax_dendro.set_ylabel("Расстояние")

    plot_silhouette_analysis(X_scaled, k_opt, fig.add_subplot(2, 3, 5))

    ax_sil = fig.add_subplot(2, 3, 6)
    k_range = range(2, 11)
    ax_sil.plot(k_range, sil_scores, 'bo-')
    ax_sil.axvline(x=k_opt, color='red', linestyle='--', label=f'Оптимальное k={k_opt}')
    ax_sil.set_xlabel("Число кластеров (k)")
    ax_sil.set_ylabel("Silhouette Score")
    ax_sil.set_title("Подбор k для KMeans")
    ax_sil.legend()
    ax_sil.grid(True)

    plt.tight_layout()
    plt.savefig("кластеризация_улучшенная.png", dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main()
