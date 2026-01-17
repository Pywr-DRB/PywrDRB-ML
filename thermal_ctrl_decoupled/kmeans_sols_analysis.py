# Kmeans analysis
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans_representatives(
    df,
    k_range=range(2, 5),
    random_state=42,
    n_init="auto"
):
    """
    Run KMeans on df (rows = solutions, columns = variables), 
    automatically choose K using silhouette score,
    and return the row closest to the centroid in each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Each row is a solution; each column is a variable.
    k_range : range or list, optional
        Candidate numbers of clusters to try (default: 2..10).
    random_state : int, optional
        Random seed for reproducibility.
    n_init : int or "auto"
        Passed to sklearn KMeans.

    Returns
    -------
    best_k : int
        Selected number of clusters.
    sil_scores : dict
        Silhouette scores for each K tried.
    reps_df : pd.DataFrame
        Representative solutions: one row (original df row) per cluster.
    labels : pd.Series
        Cluster label for each row in df.
    """
    # 1. Standardize features (important if variables are on different scales)
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # 2. Search for best K using silhouette score
    sil_scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)
        # Silhouette score requires at least 2 clusters and less than n_samples
        if 1 < k < len(df):
            score = silhouette_score(X, labels)
            sil_scores[k] = score

    # Choose K with the highest silhouette score
    best_k = max(sil_scores, key=sil_scores.get)

    # 3. Fit final KMeans with best_k
    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    # 4. For each cluster, find the row closest to the centroid
    rep_indices = []
    for cluster_id in range(best_k):
        # indices of points in this cluster
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # data points in this cluster (in standardized space)
        X_cluster = X[cluster_mask]

        # corresponding centroid
        center = centers[cluster_id]

        # Euclidean distance to centroid
        dists = np.linalg.norm(X_cluster - center, axis=1)

        # index of closest point (in original df index space)
        closest_local_idx = np.argmin(dists)
        closest_global_idx = cluster_indices[closest_local_idx]
        rep_indices.append(df.index[closest_global_idx])

    reps_df = df.loc[rep_indices].copy()
    reps_df["cluster_id"] = range(best_k)

    return best_k, sil_scores, reps_df, pd.Series(labels, index=df.index, name="cluster_id")

#%%
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
import clt

policy ="GaussianRBFPolicy"
job_id = "143990"

# Load ref
toC = 103.16
df_ref = clt.borg.read_ref(pn.outputs.get(f"dps_{policy}_{job_id}/borg.ref"))
df_ref = df_ref.rename(columns={
    'obj3': 'Jtubr',
    'obj1': '-Jrel',
    'obj2': 'Jadd'
})
df_ref["Jadd"] /= 0.7984
df_ref["Jadd"] *= toC 
df_ref["Jtubr"] *= 3
# df_ref 95 153

df1 = df_ref[df_ref["Jtubr"] <=1 ] #53 153 10
df2 = df_ref[(df_ref["Jtubr"] > 1) & (df_ref["Jtubr"] <= 2)] #86, 143
df3 = df_ref[(df_ref["Jtubr"] > 2) & (df_ref["Jtubr"] <= 3)] #67 106 24

df23 = df_ref[df_ref["Jtubr"] >=1 ] #78 (Jtubr=2.2047), 46 (Jtubr=2.8788)

best_k, sil_scores, reps_df, labels = kmeans_representatives(df23.iloc[:, :-3])

print("Chosen K:", best_k)
print("Silhouette scores:", sil_scores)
print("Representative solutions (closest to each centroid):")
print(reps_df)









