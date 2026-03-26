---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"deletable": false, "editable": false, "nbgrader": {"checksum": "56952047c4310bf0c313ca44bb5af720", "grade": false, "grade_id": "cell-883bbb5e1919ca1e", "locked": true, "schema_version": 3, "solution": false, "task": false}, "slideshow": {"slide_type": ""}, "tags": []}

# Analyse de données

:::{admonition} Consignes
:class: dropdown

Vous documenterez votre analyse de données dans cette feuille. Nous
vous fournissons seulement l'ossature. À vous de piocher dans les
feuilles et TPs précédents les éléments que vous souhaiterez
réutiliser et adapter pour composer votre propre analyse!

:::

+++ {"deletable": false, "editable": false, "nbgrader": {"checksum": "07b6f7ec89dbe99e092902adb7ff9cc8", "grade": false, "grade_id": "cell-e15fdd0116a9b9e2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Import des bibliothèques

+++ {"deletable": false, "editable": false, "nbgrader": {"checksum": "96489bd4f670f5daa152f95fa18ce301", "grade": false, "grade_id": "cell-406ad2a159064cfa", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On commence par importer les bibliothèques dont nous aurons besoin.

:::{admonition} Consignes
:class: dropdown

Inspirez vous des précédentes feuilles. N'oubliez pas d'importer
`utilities` et `intro_science_donnees`.

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  checksum: 8e6a7da13b6e1d41caa1c33a8c0104e5
  grade: false
  grade_id: cell-76e64ee7bcb96bb5
  locked: false
  schema_version: 3
  solution: true
  task: false
---
%load_ext autoreload
%autoreload 2
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
%matplotlib inline
import scipy
from scipy import signal
import pandas as pd
import numpy as np
import seaborn as sns
from glob import glob as ls
import sys
from utilities import *
from intro_science_donnees import data
from intro_science_donnees import *
```

+++ {"deletable": false, "editable": false, "nbgrader": {"checksum": "e319f7fbbc65f7368fd94b546edb9e34", "grade": false, "grade_id": "cell-6ec03d05a4e1c693", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Chargement des images

:::{admonition} Consignes
:class: dropdown

- Chargez vos images recentrées et réduites.
- Affichez les.

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  checksum: a842cb19c2f315d9a183cf8a17bd295f
  grade: false
  grade_id: cell-211e275ccc954714
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# Chargement de votre propre jeu de donnée s
dataset_dir = 'data'
images = load_images(dataset_dir, "*.png")

# Chargement du jeu de données ApplesAndBananas
# dataset_dir = os.path.join(data.dir, 'ApplesAndBananas')
# images = load_images(dataset_dir, "?[012]?.png")
from intro_science_donnees import *  #les librairies classiques pour ISD (numpy, PIL...) y sont incluses
# Configuration intégration dans Jupyter
%matplotlib inline
extension = 'jpg' 
dataset_dir = 'data'
images_a = load_images(dataset_dir, f"a*.{extension}")
images_b = load_images(dataset_dir, f"b*.{extension}")
images = pd.concat([images_a, images_b])
image_grid(images, titles=images.index)
```

```{code-cell} ipython3
assert len(images) == 20
```

```{code-cell} ipython3
show_source(my_preprocessing)
```

```{code-cell} ipython3
show_source(my_foreground_filter)
```

```{code-cell} ipython3
plt.imshow(my_preprocessing(images.iloc[0]))
```

```{code-cell} ipython3
plt.imshow(my_preprocessing(images.iloc[19]))
```

```{code-cell} ipython3
clean_images = {}
raw_files = glob.glob('data/**/*.jpg', recursive=True)

for file_path in raw_files:
    name = os.path.basename(file_path)
    img = Image.open(file_path).convert('RGB')
    clean_images[name] = my_preprocessing(img)

os.makedirs('clean_data', exist_ok=True)
for name, img in clean_images.items():
    img.save(os.path.join('clean_data', os.path.splitext(name)[0]+".png"))
```

```{code-cell} ipython3
dataset_dir = 'clean_data'
clean_images = load_images(dataset_dir, "*.png")
image_grid(clean_images, titles=clean_images.index)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"checksum": "90ece8faea4fbed29ee2f9101b82061d", "grade": false, "grade_id": "cell-e723e3fb5001bd97", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Prétraitement : extraction des attributs

:::{admonition} Consignes
:class: dropdown

- Choisissez entre des attributs ad-hoc, les votres ou une combinaison de certains attributs.
- N'oubliez pas de normaliser votre table une fois les traitements
  effectués.
  
:::

```{code-cell} ipython3
show_source(area)
show_source(elongation)
show_source(ratio)
show_source(std_brightness)
show_source(texture_ratio)
show_source(ratio_over_std)
```

```{code-cell} ipython3
df_features = pd.DataFrame({
    'area': clean_images.apply(area),
    'elongation': clean_images.apply(elongation),
    'surface_perimeter_ratio': clean_images.apply(ratio),
    'std_brightness': clean_images.apply(std_brightness),
    'texture_ratio': clean_images.apply(texture_ratio),
    'ratio_over_std': clean_images.apply(ratio_over_std),
    'class': clean_images.index.map(lambda name: 1 if name[0] == 'a' else -1),
})
df_features
```

```{code-cell} ipython3
epsilon = sys.float_info.epsilon
df_class = df_features['class']
df_features = (df_features - df_features.mean())/(df_features.std() + epsilon)
```

```{code-cell} ipython3
df_features.describe()
```

```{code-cell} ipython3
df_features["class"] = df_class
df_features.describe()
```

```{code-cell} ipython3
corr = df_features.corr()
corr.style.format(precision=2).background_gradient(cmap='coolwarm')
```

```{code-cell} ipython3
df_features_reduced = df_features[[
    'surface_perimeter_ratio',
    'ratio_over_std',
    'elongation',
    'class'
]]

display(df_features_reduced.head())
```

+++ {"deletable": false, "editable": false, "nbgrader": {"checksum": "0b04fc60580d53165e039ab88751a746", "grade": false, "grade_id": "cell-a69bb602ef2221a4", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Sauvegarde intermédiaire

+++ {"deletable": false, "editable": false, "nbgrader": {"checksum": "35a89b8dc927e0dd763e9cdf592c4bdf", "grade": false, "grade_id": "cell-8cb1846130b9cfe2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Consignes
:class: dropdown

Une fois que vous êtes satisfaits des attributs obtenus, faites en une
sauvegarde intermédiaire.

:::

```{code-cell} ipython3
df_features.to_csv("features_data_basic.csv")
```

+++ {"deletable": false, "editable": false, "nbgrader": {"checksum": "b11c92757f734a72bf65f47b9513ca5f", "grade": false, "grade_id": "cell-4576e2c2723b724d", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Visualisation des données

+++ {"deletable": false, "editable": false, "nbgrader": {"checksum": "ba2beca9ed69fdecbf145787bb70f7af", "grade": false, "grade_id": "cell-2a67648ea9c07906", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Consignes
:class: dropdown

Visualisez les attributs et leur corrélation. Mettez en œuvre les
formes que vous jugerez les plus pertinentes (carte thermique, ...).

:::

```{code-cell} ipython3
df_features_reduced.style.background_gradient(cmap='coolwarm')
```

```{code-cell} ipython3
corr_matrix = df_features_reduced.drop(columns='class').corr()
corr_matrix
```

```{code-cell} ipython3
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.show()
```

```{code-cell} ipython3
sns.pairplot(df_features, hue='class', palette='Set1', diag_kind='kde')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"checksum": "b6afee00ce7826717b483c73488161ff", "grade": false, "grade_id": "cell-c103279002f68467", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Performance 

:::{admonition} Consignes
:class: dropdown

- Choisissez un ou plusieurs classificateurs et calculez leur
  performances.
- Comparez les performances selon les attributs utilisés.
- Si vous le souhaitez, vous pouvez aussi:
    - comparer un ou plusieurs classificateurs
	- comparer les performances selon le nombre de colonnes (pixels ou
      attributs) considérées.

:::

```{code-cell} ipython3
from sklearn.metrics import balanced_accuracy_score as sklearn_metric
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
```

```{code-cell} ipython3
model_name = [
    "Nearest Neighbors", "Parzen Windows", "Linear SVM", "RBF SVM", 
    "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", 
    "AdaBoost", "Naive Bayes"
]

model_list = [
    KNeighborsClassifier(3),
    RadiusNeighborsClassifier(radius=12.0),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=0.1, C=14, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=200, max_features=1),
    MLPClassifier(shuffle=True, max_iter=2000, random_state=36, tol=1e-4),
    AdaBoostClassifier(),
    GaussianNB()
]
```

```{code-cell} ipython3
compar_results = systematic_model_experiment(
    df_features_reduced,
    model_name,
    model_list,
    sklearn_metric
)

display(
    compar_results.sort_values(by='perf_te', ascending=False)
    .style.format(precision=3)
    .background_gradient(cmap='Blues')
)
```

```{code-cell} ipython3
df_features_large = df_features_reduced.drop("class", axis = 1)
df_features_large = pd.concat([df_features_large, clean_images.apply(get_colors)], axis=1)
```

```{code-cell} ipython3
epsilon = sys.float_info.epsilon 
df_features_large = (df_features_large - df_features_large.mean()) / (df_features_large.std() + epsilon)
```

```{code-cell} ipython3
df_features_large = df_features_large.fillna(0)
df_features_large['class'] = df_features_large.index.map(lambda name: 1 if name.startswith('a') else -1)
df_features_large.head()
```

```{code-cell} ipython3
compar_results_large = systematic_model_experiment(
    df_features_large, 
    model_name, 
    model_list, 
    sklearn_metric
)

display(compar_results_large.sort_values(by='perf_te', ascending=False).style.format(precision=3).background_gradient(cmap='Blues'))
```

```{code-cell} ipython3
sklearn_model = GaussianNB()
```

```{code-cell} ipython3
performances = pd.DataFrame(columns = ['Traitement', 'perf_tr', 'std_tr', 'perf_te', 'std_te'])
```

```{code-cell} ipython3
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features, sklearn_model, sklearn_metric)
performances.loc[0] = ["3 attributs ad-hoc", p_tr, s_tr, p_te, s_te]
```

```{code-cell} ipython3
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features_large, sklearn_model, sklearn_metric)
performances.loc[1] = ["23 attributs ad-hoc", p_tr, s_tr, p_te, s_te]
performances
```

```{code-cell} ipython3
corr_large = df_features_large.corr()
sval = corr_large['class'].drop('class').abs().sort_values(ascending=False)
ranked_columns = sval.index.values
print(ranked_columns)
```

```{code-cell} ipython3
col_selected = ranked_columns[0:3]
df_features_final = pd.DataFrame.copy(df_features_large)
df_features_final = df_features_final[col_selected]
df_features_final['class'] = df_features_large["class"]
g = sns.pairplot(df_features_final, hue="class", markers=["o", "s"], diag_kind="hist")
```

```{code-cell} ipython3
feat_lc_df, ranked_columns = feature_learning_curve(df_features_large, sklearn_model, sklearn_metric)
```

```{code-cell} ipython3
plt.errorbar(feat_lc_df.index+1, feat_lc_df['perf_tr'], yerr=feat_lc_df['std_tr'], label='Training set')
plt.errorbar(feat_lc_df.index+1, feat_lc_df['perf_te'], yerr=feat_lc_df['std_te'], label='Test set')
plt.xticks(np.arange(1, len(feat_lc_df)+1, 1))
plt.xlabel('Number of features')
plt.ylabel(sklearn_metric.__name__)
plt.legend(loc='lower right');
```

```{code-cell} ipython3
compar_results_final = systematic_model_experiment(
    df_features_final, 
    model_name, 
    model_list, 
    sklearn_metric
)
display(compar_results_final.sort_values(by='perf_te', ascending=False).style.format(precision=3).background_gradient(cmap='Blues'))
```

```{code-cell} ipython3
df_features_final.to_csv('features_data.csv')
```

```{code-cell} ipython3
sklearn_model = GaussianNB()
```

```{code-cell} ipython3
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features_final, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
```

```{code-cell} ipython3
performances.loc[2] = ["Variance univariée", p_tr, s_tr, p_te, s_te]
performances
```

```{code-cell} ipython3
sklearn_model = GaussianNB()
```

```{code-cell} ipython3
best_perf = -1
std_perf = -1
best_i = 0
best_j = 0
nattributs = len(ranked_columns)
for i in np.arange(nattributs):     
    for j in np.arange(i+1,nattributs): 
        df = df_features_large[[ranked_columns[i], ranked_columns[j], 'class']]
        p_tr, s_tr, p_te, s_te = df_cross_validate(df, sklearn_model, sklearn_metric)        
        if p_te > best_perf: 
            best_perf = p_te
            std_perf = s_te
            tr_best = p_tr
            tr_std = s_tr
            best_i = i
            best_j = j            
metric_name = sklearn_metric.__name__.upper()
print('BEST PAIR: {}, {}'.format(ranked_columns [best_i], ranked_columns[best_j]))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, best_perf, std_perf))
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, tr_best, tr_std))
```

```{code-cell} ipython3

```

+++ {"deletable": false, "nbgrader": {"checksum": "ae7f04d9aea4689c8f25f33fd8cddf30", "grade": true, "grade_id": "cell-c5a8e3d90bb35375", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

VOTRE RÉPONSE ICI

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
