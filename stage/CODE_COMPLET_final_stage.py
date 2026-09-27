# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:35:11 2026

@author: LAL MAZ
"""


import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
import pandas as pd
import seaborn as sns
from sklearn. manifold import TSNE
from scipy.stats import probplot
from scipy.stats import ks_2samp, mannwhitneyu, shapiro, ttest_ind 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
#import shap
from sklearn.datasets import make_classification
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV, GridSearchCV,RandomizedSearchCV
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import shutil
from sklearn.model_selection import StratifiedKFold


import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Concatenate, Input, concatenate, Activation, ELU, Add, Softmax
import random
from scipy import ndimage

matplotlib.use("QtAgg")

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

plt.close('all')


# ====================================== Extraction des régions et calcul des volumes ====================================== #



# -------------- Extraction des volumes -------------- #


# Chargement atlas
atlas = nib.load(r"/home/tonic-de76/Documents/code/stage_IM_M1/atlas_image.nii")

# Chargement image cible (à remplacer par le vrai nom)
target_img = nib.load(r"/home/tonic-de76/Documents/code/stage_IM_M1/atlas_image_target.nii")

# Resample atlas -> image cible
atlas_resampled = resample_to_img(
    source_img=atlas,
    target_img=target_img,
    interpolation='nearest'
)

# Données de l'atlas resamplé
data = atlas_resampled.get_fdata()

# Lecture du fichier labels
labels = pd.read_excel(r"/home/tonic-de76/Documents/code/stage_IM_M1/atlas_labels.xlsx")

# Labels présents hors fond
labels_presents = np.unique(data)
labels_presents = labels_presents[labels_presents != 0]

# Taille voxel et volume voxel
dx, dy, dz = atlas_resampled.header.get_zooms()[:3]
voxel_volume = dx*dy*dz

print("Résolution voxel :", (dx, dy, dz))
print("Volume voxel :", voxel_volume, "mm³")

# Création des masques
masks = {}
for j in labels_presents:
    masks[j] = (data == j).astype(np.uint8)

# Création objets NIfTI pour visualisation
masks_view = {}
for j in labels_presents:
    masks_view[j] = nib.Nifti1Image(
        masks[j],
        affine=atlas_resampled.affine,
        header=atlas_resampled.header
    )
 
mat = list(masks.values())[18]   # 11e matrice
plt.figure()
plt.imshow(mat[50,:,:],cmap='gray')
plt.title("Région ID : 17  -  50e coupe sur axe x")

plt.figure()
plt.imshow(data[50,:,:])
plt.title("50e coupe sur axe x")


# -------------- Exemple visualisation interface -------------- #


"""
region = 27
view = plotting.view_img(masks_view[region])
view.open_in_browser()
"""


# -------------- Calcul des volumes -------------- #


results = []

for j in labels_presents:                   # Parcourt chaque région
    n_voxels = np.sum(masks[j])             # Récupère le nombre de voxels : fais la somme de tous les pixels à 1 du masque
    volume_mm3 = n_voxels * voxel_volume    # Calcul le volume

    region_info = labels[labels["Region ID"] == j] # Récupère le nom de la région
    if len(region_info) > 0:
        region_name = region_info.iloc[0]["Region Name"] # On récupère le nom si c'est trouvée
    else:
        region_name = f"ROI_{int(j)}" # Sinon par défaut on met ROI_i

    # Stockage des résultat
    results.append({
        "Region ID": int(j),
        "Region Name": region_name,
        "Nb voxels": int(n_voxels),
        "Volume (mm3)": volume_mm3,
        "Volume (cm3)": volume_mm3 / 1000
    })

df_volumes = pd.DataFrame(results) # Transforme la liste results en tableau
print(df_volumes.head())

total_volume = df_volumes["Volume (mm3)"].sum() # Fais la somme de tous les volumes
df_volumes["Volume (%)"] = (df_volumes["Volume (mm3)"] / total_volume) * 100 # On convertit les volumes en pourcentages
print(df_volumes["Volume (%)"].sum()) # ça devrait faire environ 100% normalement
df_volumes.to_excel(r"/home/tonic-de76/Documents/code/stage_IM_M1/volumes_atlas_resampled.xlsx", index=False)




# ========================== Préparation des données ========================== #



"""
dt_atlas_resample = pd.read_excel(r"G:/Data Stage IM M1/data_volumetry_bilateral.xlsx")
 
X = dt_atlas_resample.drop(columns=["label"])
y = dt_atlas_resample['label'].values
 
X_train_subject, X_test_subject, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50, stratify=y  # stratify=y : conserve la proportion contrôle/patient
)
 
X_train = X_train_subject.drop(columns=["Subject"])
X_test = X_test_subject.drop(columns=["Subject"])

scaler = StandardScaler()
 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

split_name = "80/20"
"""

dt_atlas_resample = pd.read_excel(r"/home/tonic-de76/Documents/code/stage_IM_M1/data_volumetry_bilateral.xlsx")
X = dt_atlas_resample.drop(columns=["label"])
y = dt_atlas_resample['label'].values

X_train_subject, X_test_subject, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50, stratify=y  # stratify=y : conserve la proportion contrôle/patient
)

X_train = X_train_subject.drop(columns=["Subject"])
X_test = X_test_subject.drop(columns=["Subject"])
 
scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
 
split_name = "80/20"
 
#cv = StratifiedKFold(n_splits=5, shuffle=None, random_state=42)
#folds = list(cv.split(X_train, y_train))
#X_train_FINAL = X_train(X_train['indice',:] == folds[4][0])
 
# Sur 80 % train uniquement on refait un split en 80/20
X_train_subject_cnn, X_val_subject_cnn, y_train_cnn, y_val_cnn = train_test_split(
    X_train_subject,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=50
)
 
X_train_cnn = X_train_subject_cnn.drop(columns=["Subject"])
X_val_cnn = X_val_subject_cnn.drop(columns=["Subject"])
 
scaler = StandardScaler()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

folds = []
folds_score = []
folds_nom = []
 
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_subject, y_train)):
 
    train_subjects = X_train_subject.iloc[train_idx]["Subject"].tolist()
    val_subjects = X_train_subject.iloc[val_idx]["Subject"].tolist()
 
    # Pour le CNN
    folds.append({
        "train_subjects": train_subjects,
        "val_subjects": val_subjects
    })
 
    # Pour les classifieurs sklearn
    folds_score.append((train_idx, val_idx))
    folds_nom.append((train_subjects,val_subjects))


print("\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
print(folds_score)


# ====================================== Classifieurs ====================================== #



# =================================  SVM  ================================= #



# Pas de fit ici, sinon fuite d'information (data leakage)

kernels = ['linear', 'poly', 'rbf']   # linear : hyperplan linéaire, interprétable, adapté si les classes sont linéairement séparables
                                      # poly : frontière polynomiale, capture des interactions non-linéaires modérées
                                      # rbf : frontière radiale gaussienne, capture des non-linéarités complexes


# Boucle sur deux stratégies de split pour comparer leur impact sur la performance
# 70/30 = plus de données test => estimation plus fiable de la généralisation
# 80/20 = plus de données train => meilleur apprentissage si échantillon petit



# --------- Affichage --------- #


print(f"\n{'='*50}")
print(f"Split {split_name} — Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"{'='*50}")

for k in kernels:   # Boucle sur chaque type de kernel SVM

    # Création du modèle SVM avec le kernel courant
    # random_state pour reproductibilité (certains kernels ont une composante stochastique)
    svm = SVC(C=10, kernel=k, random_state=50)

    # Entraînement du SVM sur les données train standardisées
    # Le modèle apprend la frontière de décision séparant contrôles et patients
    svm.fit(X_train_scaled, y_train)
    
    
    # ------------------ SVM Performances ------------------ #
    

    # Accuracy sur le train : mesure l'ajustement du modèle aux données d'entraînement
    # Si très élevée et test faible → overfitting (le modèle mémorise au lieu de généraliser)
    acc_train = accuracy_score(y_train, svm.predict(X_train_scaled))

    # Accuracy sur le test : mesure la capacité de généralisation à des données jamais vues
    # C'est la métrique qui compte réellement
    acc_test = accuracy_score(y_test, svm.predict(X_test_scaled))

    # Validation croisée 5-fold sur le train uniquement
    # Découpe le train en 5 plis, entraîne sur 4, évalue sur 1, répète 5 fois
    # Donne une estimation plus robuste que le seul acc_train
    # Détecte l'overfitting : si cv << acc_train, le modèle ne généralise pas
    cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv = folds_score)


    # Affichage des résultats pour ce kernel
    print(f"\n  Kernel: {k}")
    print(f"  Acc train: {acc_train:.3f}")
    print(f"  Acc test:  {acc_test:.3f}")
    print(f"  CV 5-fold: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    # Moyenne ± écart-type du CV : l'écart-type indique la stabilité du modèle
    # Un grand écart-type signifie que la performance varie beaucoup selon le pli

    # Classification report : détaille précision, rappel, f1 par classe
    # Précision = parmi les prédits positifs, combien sont vrais positifs
    # Rappel = parmi les vrais positifs, combien sont détectés
    # F1 = moyenne harmonique des deux, utile si classes déséquilibrées
    print(classification_report(y_test, svm.predict(X_test_scaled)))
    

# ------------------ SVM visualisation sous PCA ------------------ #


# Réduction à 2 composantes principales pour pouvoir tracer en 2D
# On perd de l'information mais on gagne en interprétabilité visuelle
# La PCA est entraînée sur le train scalé uniquement (même logique que le scaler)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Affiche la part de variance expliquée par chaque composante
# Si la somme est faible (ex: < 0.7), la projection 2D est une approximation grossière
# Les frontières visibles ne reflètent pas exactement celles de l'espace 3D original
print(f"\nVariance expliquée PCA ({split_name}): {pca.explained_variance_ratio_}")

# Création de 3 sous-graphiques côte à côte, un par kernel
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Titre global indiquant quel split est visualisé
fig.suptitle(f"Frontières de décision — Split {split_name}", fontsize=14)
    


# Boucle sur chaque kernel pour tracer sa frontière de décision en 2D
for ax, k in zip(axes, kernels):

    # On ré-entraîne un SVM sur les 2 composantes PCA (pas sur les 3 features originales)
    # Ce SVM 2D sert UNIQUEMENT à la visualisation, pas à l'évaluation
    # Les performances reportées plus haut viennent du SVM entraîné sur les 3 features
    svm_2d = SVC(kernel=k, random_state=50)
    svm_2d.fit(X_train_pca, y_train)

    # Calcul des bornes de la grille avec une marge de 1 unité
    # Pour que les points ne soient pas collés aux bords du graphique
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

    # Création d'une grille de 300x300 points couvrant l'espace 2D
    # Chaque point de la grille sera classifié par le SVM pour colorier les zones
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))

    # Prédiction du SVM 2D sur chaque point de la grille
    # np.c_ concatène les coordonnées x et y en une matrice (n_points, 2)
    # ravel() aplatit la grille 2D en vecteur 1D
    # reshape remet le résultat en forme 2D pour le contour plot
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Colorie les zones de décision : bleu pour classe 0, rouge pour classe 1
    # alpha=0.3 : transparence pour que les points restent visibles
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    # Affichage des points d'entraînement contrôles (classe 0) en bleu
    ax.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1],
               c='blue', label='Contrôle', edgecolors='k', s=40)

    # Affichage des points d'entraînement patients (classe 1) en rouge
    ax.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1],
               c='red', label='Patient', edgecolors='k', s=40)

    # Titre du sous-graphique = nom du kernel
    ax.set_title(f'Kernel: {k}')
    # Axes = composantes principales, pas les variables originales
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    # Légende pour distinguer les deux classes
    ax.legend()

# Ajuste l'espacement entre sous-graphiques pour éviter les chevauchements
plt.tight_layout()
# Sauvegarde avec un nom de fichier unique par split
# replace('/', '_') car '/' est interdit dans les noms de fichiers
plt.savefig(f"svm_decision_boundaries_{split_name.replace('/', '_')}.png", dpi=150)
# Affichage à l'écran
#plt.show()


"""
# ------------------  SVM  -  Explicabilité : Shap values  ------------------ #



# ------ lineaire ------ #


plt.title("Modèle explicabilité SVM linéaire")
svm_lin = SVC(kernel='linear', random_state=50)
svm_lin.fit(X_train_scaled, y_train)
explainer_svm_lin = shap.LinearExplainer(svm_lin, X_train_scaled, feature_perturbation="correlation_dependent")
shap_values_svm_lin = explainer_svm_lin.shap_values(X_test_scaled)
shap.summary_plot(shap_values_svm_lin, X_test_scaled)


# ------ rbf ------ #


plt.suptitle("Modèle explicabilité SVM RBF")
svm_rbf = SVC(C=10, kernel='rbf', random_state=50, probability=True)
svm_rbf.fit(X_train_scaled, y_train)
def f(x):
    return svm_rbf.predict_proba(x)[:, 1]
explainer_svm_rbf = shap.KernelExplainer(f, X_train_scaled, feature_perturbation="correlation_dependent")
shap_values_svm_rbf = explainer_svm_rbf.shap_values(X_test_scaled, nsamples=100)
shap.summary_plot(shap_values_svm_rbf, X_test_scaled)


# ------ poly ------ #


plt.suptitle("Modèle explicabilité SVM polynomial")
svm_pol = SVC(kernel='poly', random_state=50, probability=True)
svm_pol.fit(X_train_scaled, y_train)
def f2(x):
    return svm_pol.predict_proba(x)[:, 1]
explainer_svm_pol = shap.KernelExplainer(f2, X_train_scaled, feature_perturbation="correlation_dependent")
shap_values_svm_pol = explainer_svm_pol.shap_values(X_test_scaled)
shap.summary_plot(shap_values_svm_pol, X_test_scaled)
"""


# =================================  KMeans  ================================= #



print("\n=============== Kmeans ===============\n")


kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_train_scaled, y_train) # entraînement des kmeans
cluster_labels = kmeans.labels_ # récupère les labels des classes prédites
acc_kmeans_train = accuracy_score(y_train,kmeans.predict(X_train_scaled)) # accuracy sur la base de train
acc_kmeans_test = accuracy_score(y_test,kmeans.predict(X_test_scaled)) # accuracy sur la base de test

cv_kmeans = cross_val_score(kmeans, X_train_scaled, y_train, scoring='accuracy', cv=5)
cv_kmeans_2 = cross_validate(kmeans, X_train_scaled, y_train, scoring=('accuracy','f1','recall','precision'), cv=5)


print(f"Acc train: {acc_kmeans_train:.3f}\nAcc test: {acc_kmeans_test:.3f}")
print(f"CV 5-fold 2 acc mean: {cv_kmeans_2['test_accuracy'].mean():.3f} +/- {cv_kmeans_2['test_accuracy'].std():.3f}")
print(f"CV 5-fold acc mean: {cv_kmeans.mean():.3f} +/- {cv_kmeans.std():.3f}\n")
print(classification_report(y_test, kmeans.predict(X_test_scaled)))


plt.scatter(X_train_pca[cluster_labels==0,0], X_train_pca[cluster_labels==0,1], c='lightskyblue', label='Contrôle KMeans', s=20)
plt.scatter(X_train_pca[cluster_labels==1,0], X_train_pca[cluster_labels==1,1], c='lightpink', label='Patient KMeans', s=20)
plt.scatter(X_train_pca[y_train==0,0], X_train_pca[y_train==0,1], facecolors='none', edgecolors='mediumblue', label='Contrôle', linewidths=1.5)
plt.scatter(X_train_pca[y_train==1,0], X_train_pca[y_train==1,1], facecolors='none', edgecolors='crimson', label='Patient', linewidths=1.5)
plt.title("K-means avec n = 2 - Partie droite")
plt.ylabel("Volume (%)")
plt.legend(loc='lower right')
#plt.show()



# =================================  KNN  ================================= #
 


knn = KNeighborsClassifier(metric='cosine', n_neighbors=11, weights='distance')

# X_pca les valeurs standardisés
knn.fit(X_train_scaled, y_train)
 
y_pred_knn = knn.predict(X_test_scaled)

 
# ------------------ KNN Visualisation ------------------ #


print("\n=============== KNN ===============\n") 

# Obligation de passer par un knn 2d sinon faut faire une acp ?
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
 
 
knn_2d =  KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X_train_pca, y_train)
 
 
# Création de la grille
x_min, x_max = X_train_pca[:,0].min()-1, X_train_pca[:,0].max()+1
y_min, y_max = X_train_pca[:,1].min()-1, X_train_pca[:,1].max()+1
 
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
 
# Prédiction sur toute la grille
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
 
# Affichage des zones
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
 
# Points des sujets
plt.scatter(
    X_train_pca[y_train==0,0],
    X_train_pca[y_train==0,1],
    c="blue",
    label="Contrôle",
    edgecolors="k"
)
 
plt.scatter(
    X_train_pca[y_train==1,0],
    X_train_pca[y_train==1,1],
    c="red",
    label="Patient",
    edgecolors="k"
)
 
 
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KNN visualisé dans l'espace PCA ")
plt.legend()
#plt.show()


# ------------------ KNN Performances ------------------ #

 
acc_train = accuracy_score(y_train, knn.predict(X_train_scaled))
acc_test = accuracy_score(y_test, knn.predict(X_test_scaled))
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv = folds_score)

print(f"\n Performances KNN")
print(f"  Acc train knn: {acc_train:.3f}")
print(f"  Acc test knn:  {acc_test:.3f}")
print(f"  CV 5-fold KNN: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(classification_report(y_test, knn.predict(X_test_scaled)))
 
 
X_test_pca = pca.transform(X_test_scaled)
 
 
plt.figure()
plt.scatter(X_test_pca[y_pred_knn==0,0], X_test_pca[y_pred_knn==0,1], c='lightskyblue', label='Contrôle KNN', s=20)
plt.scatter(X_test_pca[y_pred_knn==1,0], X_test_pca[y_pred_knn==1,1], c='lightpink', label='Patient KNN', s=20)
plt.scatter(X_test_pca[y_test==0,0], X_test_pca[y_test==0,1], facecolors='none', edgecolors='mediumblue', label='Contrôle', linewidths=1.5)
plt.scatter(X_test_pca[y_test==1,0], X_test_pca[y_test==1,1], facecolors='none', edgecolors='crimson', label='Patient', linewidths=1.5)
plt.title("KNN prédit vs réel")
plt.ylabel("Volume (%)")
plt.legend(loc='lower right')
#plt.show()
 

"""
# ------------------  KNN  -  Explicabilité : Shap values  ------------------ #



shap.initjs()
 
def f(x):
    return knn.predict_proba(x)[:, 1] #oblige de mettre une fonction pba pour que explain knn fonctionne 
#☻ c'st la probabilité d'appartenir a la classe 1 celle des patients 
explain_knn = shap.KernelExplainer(f, X_train_scaled)
shap_values_knn = explain_knn.shap_values(X_test_scaled)
 
plt.title("Résumé SHAP values — KNN")
shap.summary_plot(shap_values_knn, X_test_scaled)
 
 
#shap.plots.beeswarm(explain_knn)
#en gros shap.summary utilise deja un beeswarm plot il met une erreur car il croit qu'il faut faire 2 API 
'''
shap_explanation = shap.Explanation(
    values=shap_values_knn,
    data=X_test_scaled,
    feature_names=X_train_scaled.columns
)
 
shap.plots.beeswarm(shap_explanation)

# Plus les valeus sont etale plus elles impacte et plus les valeurs sont décle vers la droite
'''
"""



# =================================  Random Forest  ================================= #



print("\n=============== RF ===============\n")

model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)

y_pred_logistic = model_logistic.predict(X_test)


model_rf = RandomForestClassifier(
    n_estimators=400,      # nombre d’arbres
    max_depth=30, 
    min_samples_leaf=3,
    max_features='sqrt',       # profondeur max; None = laisser l’algorithme décider
    random_state=50        # reproductibilité
)


model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)

cv_scores_rf = cross_val_score(model_rf, X_train_scaled, y_train, cv = folds_score )


# ------------------ RF Performances ------------------ #
  

print("Logistic accuracy:", accuracy_score(y_test, y_pred_logistic))
print("RF accuracy:", accuracy_score(y_test, y_pred_rf))
print(f"  CV 5-fold RF: {cv_scores_rf.mean():.3f} ± {cv_scores_rf.std():.3f}")
print("Random Forest report:")
print(classification_report(y_test, y_pred_rf))
print(f"\nVariance expliquée PCA ({split_name}): {pca.explained_variance_ratio_}")
print(f"Variance expliquée cumulée PCA ({split_name}): {pca.explained_variance_ratio_.sum():.4f}")


# ------------------ RF Visualisation ------------------ #

 
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train_scaled)
 
rf_full = RandomForestClassifier(n_estimators=200, random_state=42)
rf_full.fit(X_train, y_train)          # variables originales, PAS PCA

"""
# ------------------  RF  -  Explicabilité : Shap values  ------------------ #

 
import shap
# Construction de l'explainer
explainer = shap.TreeExplainer(rf_full)
# Calcul des valeurs de Shapley sur les données à expliquer
shap_values = explainer.shap_values(X_test)
explanation = explainer(X_test)         # X_test 133 colonnes , objet Explanation
# explanation.values : dimensions (n_samples, n_features, n_classes)
 
plt.title("explicabilité RF local")
#Explication locale — un individu
shap.plots.waterfall(explanation[0, :, 1])
 
plt.title("explicabilité RF : global")
#Explication globale — force du modèle
shap.plots.beeswarm(explanation[:, :, 1])
 
plt.title("explicabilité RF : importance globale")
shap.plots.bar(explanation[:, :, 1])
"""


# =================================  MLP  ================================= #



# ============================ Optimisation HP MLP ============================ #


print("\n\n ============ Optimisation HP MLP ============\n\n")

param_mlp = {'hidden_layer_sizes': [50,100,150],
             'activation': ('identity','logistic','tanh','relu'),
             'solver': ('lbfgs','sgd','adam'),
             'alpha': [1e-3,1e-4,1e-5]}

mlp_param = MLPClassifier()


# -------------- MLP  -  Base train -------------- #


print("\n\n ================== MLP  - RandomizedSearchCV -  Base train ==================\n\n")


evaluator_mlp_train = RandomizedSearchCV(
    mlp_param, param_distributions=param_mlp,
    n_iter=10, cv=5, scoring='accuracy',
    n_jobs=-1, random_state=42).fit(X_train_scaled, y_train)

df_mlp_train = pd.DataFrame(evaluator_mlp_train.cv_results_)


# ------------ ETAPE 1 : diagnostic du bruit ------------ #


top = df_mlp_train.sort_values('rank_test_score').head(15).copy()
top['params_str'] = top['params'].astype(str)


print(top[['params_str', 'mean_test_score', 'std_test_score']].to_string(index=False))


best_mean = df_mlp_train.loc[df_mlp_train['rank_test_score'] == 1, 'mean_test_score'].iloc[0]
best_std  = df_mlp_train.loc[df_mlp_train['rank_test_score'] == 1, 'std_test_score'].iloc[0]

meilleur_hp_mlp_train = evaluator_mlp_train.best_params_

print("\n\n ========== MLP  -  RandomziedSearchCV  -  Base de train ==========\n\n")
print("Meilleurs paramètres :", evaluator_mlp_train.best_params_)
print("Meilleur score CV :", evaluator_mlp_train.best_score_)
print(f"\nMeilleur : {best_mean:.4f} ± {best_std:.4f}")
print(f"Seuil de significativité (1 std) : [{best_mean-best_std:.4f}, {best_mean+best_std:.4f}]")


# ------------ Intervalle de confiance ------------ #


dans_bruit = (df_mlp_train['mean_test_score'] >= best_mean - best_std).sum()
print(f"Configurations indistinguables du meilleur : {dans_bruit} / {len(df_mlp_train)}")


# ------------ ETAPE 2 : Visualisation qui préserve l'incertitude ------------ #


plt.figure(figsize=(10, 6))
x = np.arange(len(top))

plt.errorbar(x, top['mean_test_score'], yerr=top['std_test_score'], fmt='o', capsize=4, color='black', ecolor='gray')
plt.axhline(best_mean - best_std, ls='--', color='red', label=f'seuil bruit ({best_mean-best_std:.3f})')
plt.xticks(x, [f"cfg {i}" for i in top.index], rotation=90)
plt.ylabel("balanced accuracy CV (moyenne ± std)")
plt.title("Top 15 configurations — barres = écart-type CV - MLP - Base train")
plt.legend()
plt.tight_layout()
#plt.show()

"""
# -------------- MLP  -  Base test -------------- #


print("\n\n ================== MLP  - RandomizedSearchCV -  Base test ==================\n\n")


evaluator_mlp_test = RandomizedSearchCV(
    mlp_param, param_distributions=param_mlp,
    n_iter=10, cv=5, scoring='accuracy',
    n_jobs=-1, random_state=42).fit(X_test_scaled, y_test)

df_mlp_test = pd.DataFrame(evaluator_mlp_test.cv_results_)


# ------------ ETAPE 1 : diagnostic du bruit ------------ #


top = df_mlp_test.sort_values('rank_test_score').head(15).copy()
top['params_str'] = top['params'].astype(str)


print(top[['params_str', 'mean_test_score', 'std_test_score']].to_string(index=False))


best_mean = df_mlp_test.loc[df_mlp_test['rank_test_score'] == 1, 'mean_test_score'].iloc[0]
best_std  = df_mlp_test.loc[df_mlp_test['rank_test_score'] == 1, 'std_test_score'].iloc[0]

meilleur_hp_mlp_test = evaluator_mlp_test.best_params_

print("Meilleurs paramètres :", evaluator_mlp_test.best_params_)
print("Meilleur score CV :", evaluator_mlp_test.best_score_)
print(f"\nMeilleur : {best_mean:.4f} ± {best_std:.4f}")
print(f"Seuil de significativité (1 std) : [{best_mean-best_std:.4f}, {best_mean+best_std:.4f}]")


# ------------ Intervalle de confiance ------------ #


dans_bruit = (df_mlp_test['mean_test_score'] >= best_mean - best_std).sum()
print(f"Configurations indistinguables du meilleur : {dans_bruit} / {len(df_mlp_test)}")


# ------------ ETAPE 2 : Visualisation qui préserve l'incertitude ------------ #


plt.figure(figsize=(10, 6))
x = np.arange(len(top))

plt.errorbar(x, top['mean_test_score'], yerr=top['std_test_score'], fmt='o', capsize=4, color='black', ecolor='gray')
plt.axhline(best_mean - best_std, ls='--', color='red', label=f'seuil bruit ({best_mean-best_std:.3f})')
plt.xticks(x, [f"cfg {i}" for i in top.index], rotation=90)
plt.ylabel("balanced accuracy CV (moyenne ± std)")
plt.title("Top 15 configurations — barres = écart-type CV - MLP - Base train")
plt.legend()
plt.tight_layout()
plt.show()
"""


# ================= MLP avec bon HP ================= #



# ------------------ MLP Matrice de confusion ------------------ #


mlp = MLPClassifier(solver=meilleur_hp_mlp_train['solver'], hidden_layer_sizes=(meilleur_hp_mlp_train['hidden_layer_sizes'],),
                    alpha=meilleur_hp_mlp_train['alpha'], activation=meilleur_hp_mlp_train['activation']).fit(X_train_scaled, y_train)
 
y_pred_mlp = mlp.predict(X_test_scaled)


print("\n\n ------------ Matrice de confusion MLP ------------\n\n")
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

print(cm_mlp)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp)
disp.plot()
plt.title("MLP - Matrice de confusion")
#plt.show()

tn, fp, fn, tp = cm_mlp.ravel()
print(f"Matrice de confusion des MLP: (tn, fp, fn, tp) = ({tn}, {fp}, {fn}, {tp})")
print("\n\n")

df_erreurs_mlp = X_test.copy()  
df_erreurs_mlp["Classe_reelle"] = y_test
df_erreurs_mlp["Classe_predite"] = y_pred_mlp
df_erreurs_mlp["Subject"] = X_test_subject["Subject"]
df_erreurs_mlp = df_erreurs_mlp[df_erreurs_mlp["Classe_reelle"] != df_erreurs_mlp["Classe_predite"]]

#classe réelle = 0, prédite = 1 => FP
#classe réelle = 1, prédite = 0 => FN

df_erreurs_mlp["Type Erreur"] = np.where(
    (df_erreurs_mlp["Classe_reelle"] != df_erreurs_mlp["Classe_predite"]) &
    (df_erreurs_mlp["Classe_predite"] == 1),
    "FP",
    "FN"
)


# ------------------ MLP Performances ------------------ #

 
acc_train = accuracy_score(y_train, mlp.predict(X_train_scaled))
acc_test = accuracy_score(y_test, mlp.predict(X_test_scaled))
cv_scores = cross_val_score(mlp, X_train_scaled, y_train, cv = folds_score )

 
print(f"\n --------------- Performances MLP ---------------")
print(f"  Acc train MLP: {acc_train:.3f}")
print(f"  Acc test MLP:  {acc_test:.3f}")
print(f"  CV 5-fold MLP: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(classification_report(y_test, mlp.predict(X_test_scaled)))
 
 
X_test_pca = pca.transform(X_test_scaled)
 
 
plt.figure()
plt.scatter(X_test_pca[y_pred_mlp==0,0], X_test_pca[y_pred_mlp==0,1], c='lightskyblue', label='Contrôle MLP', s=20)
plt.scatter(X_test_pca[y_pred_mlp==1,0], X_test_pca[y_pred_mlp==1,1], c='lightpink', label='Patient MLP', s=20)
plt.scatter(X_test_pca[y_test==0,0], X_test_pca[y_test==0,1], facecolors='none', edgecolors='mediumblue', label='Contrôle', linewidths=1.5)
plt.scatter(X_test_pca[y_test==1,0], X_test_pca[y_test==1,1], facecolors='none', edgecolors='crimson', label='Patient', linewidths=1.5)
plt.title("MLP prédit vs réel")
plt.ylabel("Volume (%)")
plt.legend(loc='lower right')
#plt.show()


# ------------------ MLP Shap Values ------------------ #

"""
plt.suptitle("Shap Values MLP")
def f2(x):
    return mlp.predict_proba(x)[:, 1]
explainer_mlp = shap.KernelExplainer(f2, X_train_scaled, feature_perturbation="correlation_dependent")
shap_values_mlp = explainer_mlp.shap_values(X_test_scaled)
shap.summary_plot(shap_values_mlp, X_test_scaled)
"""


# ========================== Structure CNN ========================== #



X_train_cnn_scaled = pd.DataFrame(scaler.fit_transform(X_train_cnn), columns=X_train_cnn.columns, index=X_train_cnn.index)
X_val_cnn_scaled = pd.DataFrame(scaler.transform(X_val_cnn), columns=X_val_cnn.columns, index=X_val_cnn.index)

X_train_cnn_scaled.insert(0,'Subject',X_train_subject_cnn['Subject'])
X_val_cnn_scaled.insert(0,'Subject',X_val_subject_cnn['Subject'])
X_test_scaled.insert(0,'Subject',X_test_subject['Subject'])


# ---------------------- Séparation des folds ---------------------- #


"""
print("\n\n---------------- Séparation des folds images ----------------\n\n")

hc_list = os.listdir("/home/tonic-de76/Documents/code/stage_IM_M1/images/HC")
msa_list = os.listdir("/home/tonic-de76/Documents/code/stage_IM_M1/images/MSA")

f1 = folds_nom[0]
f2 = folds_nom[1]
f3 = folds_nom[2]
f4 = folds_nom[3]
f5 = folds_nom[4]

cpt = 0
for j in  range(5):
    for i in range(len(X_train_subject['Subject'])):

        img_comp = X_train_subject['Subject'].values[i] + '_T1_in_MNI222_HC.nii'

        if (img_comp in hc_list):

            # ---- HC ---- #

            if (X_train_subject['Subject'].values[i] in folds_nom[j][0]): #1e colonne/liste = train /// 2e colonne/liste = val

                # ---- Base de train ---- #

                shutil.copy2(f"/home/tonic-de76/Documents/code/stage_IM_M1/images/HC/{img_comp}",f"/home/tonic-de76/Documents/code/stage_IM_M1/images/folds/fold_{j+1}/f{j+1}_train")
                print(f"\nImage récupérée  :  {img_comp}  ->  HC  ->  f{j+1} train")

            elif (X_train_subject['Subject'].values[i] in folds_nom[j][1]):

                # ---- Base de val ---- #

                shutil.copy2(f"/home/tonic-de76/Documents/code/stage_IM_M1/images/HC/{img_comp}",f"/home/tonic-de76/Documents/code/stage_IM_M1/images/folds/fold_{j+1}/f{j+1}_val")
                print(f"\nImage récupérée  :  {img_comp}  ->  HC  ->  f{j+1} val")


        else:

            # ---- MSA ---- #

            img_comp = X_train_subject['Subject'].values[i] + '_T1_in_MNI222_MSA.nii'

            if (X_train_subject['Subject'].values[i] in folds_nom[j][0]): #1e colonne/liste = train /// 2e colonne/liste = val
            
                # ---- Base de train ---- #

                shutil.copy2(f"/home/tonic-de76/Documents/code/stage_IM_M1/images/MSA/{img_comp}",f"/home/tonic-de76/Documents/code/stage_IM_M1/images/folds/fold_{j+1}/f{j+1}_train")
                print(f"\nImage récupérée  :  {img_comp}  ->  MSA  ->  f{j+1} train")

            elif (X_train_subject['Subject'].values[i] in folds_nom[j][1]):

                # ---- Base de val ---- #

                shutil.copy2(f"/home/tonic-de76/Documents/code/stage_IM_M1/images/MSA/{img_comp}",f"/home/tonic-de76/Documents/code/stage_IM_M1/images/folds/fold_{j+1}/f{j+1}_val")
                print(f"\nImage récupérée  :  {img_comp}  ->  MSA  ->  f{j+1} val")
            

        cpt = cpt + 1
"""



# ----------------------------- FONCTIONS ----------------------------- #

resultats_fold = []

def read_nifti_file(filepath):
    #Read and load volume
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    #Normalise le volume avec ses propres bornes min/max

    vmin = np.min(volume)
    vmax = np.max(volume)
    volume = (volume - vmin) / (vmax - vmin + 1e-8)  # +epsilon pour éviter la division par 0
    volume = volume.astype("float32")
    """
    plt.hist(volume.ravel())
    plt.show()
    """
    return volume

#(tf_cuda) tonic-de76@TONIC-DE76:~/Documents/code/stage_IM_M1
# python ./partie_cnn_keras.py

def process_scan(path):
    #Read and resize volume
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    # volume = resize_volume(volume)
    return volume


def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        #angles = [-20, -10, -5, 5, 10, 20]
        #angles = [-5, -3, 3, 5]
        angles = [-3, 3]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume.astype(np.float32)

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def load_images_and_labels(folder):

    cpt = 0
    print("\n-------------------------------- Fonction load_images_and_labels --------------------------------\n")
    X = []
    y = []

    print("\n-------------------------------- On rentre dans la boucle --------------------------------\n")
    for file in os.listdir(folder):

        print(f"\n Itération {cpt+1}")
        if not (file.endswith(".nii") or file.endswith(".nii.gz")):
            continue
 
        filepath = os.path.join(folder, file)

        print("\nChargement img\n")
        # Chargement de l'image
        img = process_scan(filepath)
        X.append(img)
        print("\nChargement img FAIT\n")
 
        # Label selon le nom
        if "_HC" in file:
            y.append(0)
            print("\nlabel HC\n")
        elif "_MSA" in file:
            y.append(1)
            print("\nlabel MSA\n")
        else:
            raise ValueError(f"Impossible de déterminer le label pour {file}")
        cpt = cpt + 1
    print("\n-------------------------------- FIN FONCTION --------------------------------\n")
    return np.array(X), np.array(y)


# ------------- STRUCTURE CNN ------------- #


def get_model(width=91, height=109, depth=91):
    #Build a 3D convolutional neural network model

    inputs = keras.Input((width,height,depth,1))

    x = keras.layers.Conv3D(filters=32, kernel_size=3,padding="same", activation=None)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU(alpha=1.0)(x)

    x = keras.layers.AveragePooling3D(pool_size=2)(x)
  
    #1e Identity block
    x1 = keras.layers.Conv3D(filters=64,kernel_size=1,padding="same",activation=None)(x)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = keras.layers.ELU(alpha=1.0)(x1)

    x1 = keras.layers.Conv3D(filters=64,kernel_size=1,padding="same",activation=None)(x1)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = keras.layers.ELU(alpha=1.0)(x1)

    x1 = keras.layers.Conv3D(filters=32,kernel_size=1,padding="same",activation=None)(x1)
    x1 = keras.layers.BatchNormalization()(x1)
        
    x = keras.layers.Add()([x,x1])
    x = keras.layers.ELU(alpha=1.0)(x)
    # Fin Block Id 1


    x = keras.layers.Conv3D(filters=64, kernel_size=3,padding="same", activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU(alpha=1.0)(x)

    x = keras.layers.AveragePooling3D(pool_size=2)(x)

    #2e Identity block
    x2 = keras.layers.Conv3D(filters=128,kernel_size=1,padding="same",activation=None)(x)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ELU(alpha=1.0)(x2)

    x2 = keras.layers.Conv3D(filters=128,kernel_size=1,padding="same",activation=None)(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ELU(alpha=1.0)(x2)

    x2 = keras.layers.Conv3D(filters=64,kernel_size=1,padding="same",activation=None)(x2)
    x2 = keras.layers.BatchNormalization()(x2)

    x = keras.layers.Add()([x,x2])
    x = keras.layers.ELU(alpha=1.0)(x)
    # Fin Block Id 2


    x = keras.layers.Conv3D(filters=128,kernel_size=3,padding="same",activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU(alpha=1.0)(x)

    x = keras.layers.Conv3D(filters=128,kernel_size=3,padding="same",activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU(alpha=1.0)(x)

    x = keras.layers.Conv3D(filters=128,kernel_size=3,padding="same",activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU(alpha=1.0)(x)

    x = keras.layers.AveragePooling3D(pool_size=2)(x)


    #3e Identity block
    x3 = keras.layers.Conv3D(filters=256,kernel_size=1,padding="same",activation=None)(x)
    x3 = keras.layers.BatchNormalization()(x3)
    x3 = keras.layers.ELU(alpha=1.0)(x3)

    x3 = keras.layers.Conv3D(filters=256,kernel_size=1,padding="same",activation=None)(x3)
    x3 = keras.layers.BatchNormalization()(x3)
    x3 = keras.layers.ELU(alpha=1.0)(x3)

    x3 = keras.layers.Conv3D(filters=128,kernel_size=1,padding="same",activation=None)(x3)
    x3 = keras.layers.BatchNormalization()(x3)

    x = keras.layers.Add()([x,x3])
    x = keras.layers.ELU(alpha=1.0)(x)
    # Fin Block Id 3

    x = keras.layers.AveragePooling3D(pool_size=2)(x)
    x = keras.layers.Flatten()(x)


    #Dense block
    x = keras.layers.Dense(units=512, activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(units=512, activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Dense(units=512, activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU(alpha=1.0)(x)

    outputs = keras.layers.Dense(units=2, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# --------------------------------------------------------------------------------------- #



# ------------- Création base de test ------------- #

print("\n-------------------------------- Creation base test --------------------------------n")

batch_size = 5
hc_test_paths = [
    os.path.join(os.getcwd(), "/home/tonic-de76/Documents/code/stage_IM_M1/images/HC/base_test", x)
    for x in os.listdir("/home/tonic-de76/Documents/code/stage_IM_M1/images/HC/base_test")
]
print("Images contrôles test (HC): " + str(len(hc_test_paths)))
hc_test_img = np.array([process_scan(path) for path in hc_test_paths])
hc_labels = np.array([0 for _ in range(len(hc_test_img))])


msa_test_paths = [
    os.path.join(os.getcwd(), "/home/tonic-de76/Documents/code/stage_IM_M1/images/MSA/base_test", x)
    for x in os.listdir("/home/tonic-de76/Documents/code/stage_IM_M1/images/MSA/base_test")
]
print("Images patients test (MSA): " + str(len(msa_test_paths)))
msa_test_img = np.array([process_scan(path) for path in msa_test_paths])
msa_labels = np.array([1 for _ in range(len(msa_test_img))])


x_test = np.concatenate((hc_test_img, msa_test_img), axis=0)
y_test = np.concatenate((hc_labels, msa_labels), axis=0)

test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

test_dataset = (
    test_loader.shuffle(len(x_test))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

print("\n-------------------------------- Entraîenement du modèle --------------------------------n")

# -------------------- Entraînement modèle -------------------- #

val_accu = []
test_accu = []
train_accu = []

for i in range(5):

    # ------------ Création base train ------------ #
    print("\n-------------------------------- Creation base train --------------------------------n")

    x_train, y_train = load_images_and_labels(f"/home/tonic-de76/Documents/code/stage_IM_M1/images/folds/fold_{i+1}/f{i+1}_train")

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )

    # ------------ Création base val ------------ #
    print("\n-------------------------------- Creation base val --------------------------------n")

    x_val, y_val = load_images_and_labels(f"/home/tonic-de76/Documents/code/stage_IM_M1/images/folds/fold_{i+1}/f{i+1}_val")

    val_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = (
        val_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )

    # ----------------------- Récupération du modèle ----------------------- #

    print("\n-------------------------------- Récupération modèle --------------------------------n")
    cnn_model = get_model(width=91, height=109, depth=91)
    cnn_model.summary() # OK


    # ----------------------- Compilation du modèle ----------------------- #

    print("\n-------------------------------- Compilation--------------------------------n")
    # Compile model
    initial_learning_rate = 0.00001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    cnn_model.compile(
        loss="sparse_categorical_crossentropy",
        #loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
        run_eagerly=True,
    )

    # Define callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3d_image_classification.tensorflow.keras", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=30)

    print("\n-------------------------------- modèle avec le fit --------------------------------n")

    # Train the model, doing validation at the end of each epoch
    epochs = 100
    history=cnn_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    # ------------ Affichage courbes ------------ #

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for j, metric in enumerate(["acc", "loss"]):
        ax[j].plot(cnn_model.history.history[metric])
        ax[j].plot(cnn_model.history.history["val_" + metric])
        ax[j].set_title(f"Model fold {i+1}".format(metric))
        ax[j].set_xlabel("epochs")
        ax[j].set_ylabel(metric)
        ax[j].legend(["train", "val"])


    # ----------------------- Prédiction ----------------------- #

    print("\n-------------------------------- Prédiction --------------------------------n")

    train_loss, train_acc = cnn_model.evaluate(train_dataset)
    test_loss, test_acc = cnn_model.evaluate(test_dataset)
    val_loss, val_acc = cnn_model.evaluate(val_dataset)

    resultats_fold.append({
    "Fold": i + 1,
    "Train loss": train_loss,
    "Train accuracy": train_acc,
    "Validation loss": val_loss,
    "Validation accuracy": val_acc,
    "Test loss": test_loss,
    "Test accuracy": test_acc,
    "Nb epochs": len(history.history["loss"]),
    "Best val accuracy": max(history.history["val_acc"]),
    "Best val loss": min(history.history["val_loss"])
})

    print(f"\n--------------- FOLD {i+1} ---------------n")
    print('Test accuracy:', test_acc)
    print('Train accuracy', train_acc)
    print(history.history.keys())
    plt.plot(history.history['loss'])
    #plt.show()

    val_accu.append(val_acc)
    test_accu.append(test_acc)
    train_accu.append(train_acc)


# ----------------------- Evaluation ----------------------- #

df = pd.DataFrame(resultats_fold)



moy_val_acc = np.mean(val_accu)
std_cv_val_acc = np.std(val_accu)

moy_test_acc = np.mean(test_accu)
std_cv_test_acc = np.std(test_accu)

moy_train_acc = np.mean(train_accu)
std_cv_train_acc = np.std(train_accu)

df.loc[len(df)] = {
    "Fold": "Moyenne",
    "Train accuracy": moy_train_acc,
    "Validation accuracy": moy_val_acc,
    "Test accuracy": moy_test_acc
}

print(df)

print(f"\n --------------- Performances CNN ---------------")
print(f"  Moyenne train acc cnn: {moy_train_acc:.3f} ± {std_cv_train_acc:.3f}")
print(f"  Moyenne test acc cnn:  {moy_test_acc:.3f} ± {std_cv_test_acc:.3f}")
print(f"  Score cv = Moyenne val acc {moy_val_acc:.3f} ± {std_cv_val_acc:.3f}")
y_pred = np.argmax(cnn_model.predict(test_dataset), axis=1)
print(classification_report(y_test, y_pred))


chemin = "/home/tonic-de76/Documents/code/stage_IM_M1/resultats_cross_validation.xlsx"

df.to_excel(chemin, index=False)

print(f"Fichier enregistré dans : {chemin}")

plt.show()