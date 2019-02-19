import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# k-means + ROC http://www.philipkalinda.com/ds3.html
# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

def plot_clusters(X, labels, cax = None):
    ### tSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=80, n_iter=300)
    tsne_results = tsne.fit_transform(X=X, y=labels)

    # pca = PCA(n_components=10)
    # tsne_results = pca.fit_transform(q_psies.T)

    tsne_df = pd.DataFrame()

    tsne_df['xt'] = tsne_results[:, 0]
    tsne_df['yt'] = tsne_results[:, 1]

    # fg = sns.FacetGrid(data=tsne_df, aspect=1.61)
    # fg.map(plt.scatter, 'xt', 'yt', c=labels, s=50, cmap='viridis').add_legend()

    if cax == None:
        fig, cax = plt.subplots(1,1)

    cax.scatter(tsne_df['xt'], tsne_df['yt'], c=labels, s=50, cmap='viridis')


cnames_groups = {
    'Participant': ['Gender', 'Age', 'Education'],
    'Robot': ['Rationality', 'Color', 'Side'],
    'NARS': ['Interactions', 'Social_Influence', 'Emotions'],
    'BFI': ['Extroversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness'],
    'GODSPEED': ['Anthropomorphism', 'Animacy', 'Likability', 'Intelligence', 'Safety'],
    'Roles': ['Investments', 'Analyst', 'Jury', 'Bartender'],
    'Choices': ['agree2rational', 'prefer']}

df_dir = 'data/dataframes/'
save_dir = 'data/paper/'

raw_df = pd.read_csv(df_dir + '/new_raw_df.csv', index_col=0)

columns_of_intrest = cnames_groups['BFI'] + cnames_groups['NARS']
data = raw_df[columns_of_intrest + ['id']]

### renaming the index to be the users id
data = data.set_index('id')

### noraml trasformation
data = data.fillna(0)
scaler = MinMaxScaler(copy=True, feature_range=(0,1))
scaler.fit(data)
data[columns_of_intrest] = scaler.transform(data)


### clustering using kmeans
cq = cnames_groups['BFI']
# cq = cnames_groups['NARS']

X = data[cq].copy()
# X = data.copy()

# do_pca = False
do_pca = True

if do_pca:
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(X)
    X = pca_results[:,:2]

km = KMeans(3, random_state=0).fit(X)
labels = km.predict(X)

dbs = DBSCAN(eps = 0.001).fit(X)
labels1 = dbs.labels_

fig, cax = plt.subplots(1,1)

if do_pca:
    cax.scatter(X[:,0], X[:,1],  c=labels, s=50, cmap='viridis', alpha = .5)
    cax.scatter(X[:,0], X[:,1],  c=labels1, s=50, cmap='hot', marker = '+')
else:
    plot_clusters(X, labels, cax)
    # plot_clusters(X, labels1, cax)


### Run the Kmeans algorithm and get the index of data points clusters
sse = []
list_k = list(range(1, 50))
#
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)
#
# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');


# ### k fold cross validation of results
# list_k = list(range(1, 20))
# kf = KFold(n_splits=10, shuffle=True, random_state=123)
# colors = ['m', 'y', 'k', '#9500d8', '#a6ff9b', '#7f3a18', 'b', 'g', 'r', 'c']
# k_models = {};
# k_probabilities = {};
# k_accuracies = {};
# k_pred_spec = {};
# k_roc = {};
# k_auc = {}
# for k in list_k:
#     for mdl_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
#         train_data = data.iloc[train_idx]
#         test_data = data.iloc[test_idx]
#         models = {}
#
#         km = KMeans(n_clusters=k)
#         km.fit(train_data)
#         predicted_labels = km.predict(test_data)
#
#         print()
        # for classer in classifiers:
        #     model = LogisticRegression()
        #     model.fit(train_data.iloc[:, [0, 1, 2, 3]], train_data['actual_{}'.format(classer)])
        #     models[classer] = model
        # k_models[mdl_idx] = models
        # temp_probabilities = pd.DataFrame(columns=classifiers)
        #
        # for mdl_key, mdl_model in k_models[mdl_idx].items():
        #     temp_probabilities[mdl_key] = mdl_model.predict_proba(test_data.iloc[:, [0, 1, 2, 3]])[:, 1]
        #     k_probabilities[mdl_idx] = temp_probabilities
        #
        # for mdl_key, mdl_probs in k_probabilities[mdl_idx].items():
        #     predicted_species = temp_probabilities.idxmax(axis=1)
        #     pred_spec = temp_probabilities.max(axis=1)
        #     lgr_accuracy = len(test_data[test_data['species'] == predicted_species]) / len(test_data)
        #     k_accuracies[mdl_idx] = lgr_accuracy
        #     k_pred_spec[mdl_idx] = pred_spec
        #
        # roc = {}
        # for classer in classifiers:
        #     fpr, tpr, thresholds = roc_curve(test_data['actual_{}'.format(classer)], k_probabilities[mdl_idx][classer])
        #     roc[classer] = (fpr, tpr, thresholds)
        # k_roc[mdl_idx] = roc
        #
        # auc = {}
        # for classer in classifiers:
        #     auc_score = roc_auc_score(test_data['actual_{}'.format(classer)], k_probabilities[mdl_idx][classer])
        #     auc[classer] = auc_score
        # k_auc[mdl_idx] = auc
plt.show()
