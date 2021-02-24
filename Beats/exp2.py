# Utility Imports
import os.path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import table
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning

# Model Imports
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

'''
# Helper Methods
'''


def pca_transform(train_x_p, n_comp = 1):
    shift = train_x_p - np.mean(train_x_p, axis=0)

    u, s, v = np.linalg.svd(shift)
    v_T = np.transpose(v)

    v_slice = v_T[:, : n_comp]
    transformed_data = np.dot(train_x_p, v_slice)
    return transformed_data


def plot_2D(train_x_p, labels=None):
    fig = plt.figure()
    projection_2D = pca_transform(train_x_p, 2)
    fig.gca().scatter(projection_2D[:, 0], projection_2D[:, 1], c=labels)
    fig.gca().set_title('PCA 2D Transformation 2')
    plt.savefig("Plot 2D 2")
    plt.clf()


def plot_3D(train_x_p, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    projection_3D = pca_transform(train_x_p, 3)
    ax.scatter(projection_3D[:,0],projection_3D[:,1],projection_3D[:,2],c=labels)
    fig.gca().set_title('PCA 3D Transformation 2')
    plt.savefig("Plot 3D 2")
    plt.clf()


def generate_table(df_p):
    gen_table = pd.concat([pd.DataFrame(df_p["params"]),
                           pd.DataFrame(df_p["mean_train_score"], columns=["Mean Train Score"]),
                           pd.DataFrame(df_p["mean_test_score"], columns=["Mean Test Score"]),
                           pd.DataFrame(df_p["rank_test_score"], columns=["Rank"])], axis=1)
    return gen_table

'''
# Data Loading & Pre-processing
'''
print("================== Data Loading & Preprocessing ==================")
df = pd.read_csv('data/beatsdataset.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df.drop(df.columns[[0]], axis=1)
df['class'].value_counts()
df['class'] = df['class'].map({'BigRoom': 0, 'Breaks': 1, 'Dance': 2, 'DeepHouse': 3, 'DrumAndBass': 4, 'Dubstep': 5,
                               'ElectroHouse': 6, 'ElectronicaDowntempo': 7, 'FunkRAndB': 8, 'FutureHouse': 9,
                               'GlitchHop': 10, 'HardcoreHardTechno': 11, 'HardDance': 12, 'HipHop': 13, 'House': 14,
                               'IndieDanceNuDisco': 15, 'Minimal': 16, 'ProgressiveHouse': 17, 'PsyTrance': 18,
                               'ReggaeDub': 19, 'TechHouse': 20, 'Techno': 21, 'Trance': 22})
print("==== dataframe head ====")
print(df.head())

print("\n==== Removing Columns That Don't Aid Classification ====")
print("\n==== dataframe head ====")
print(df.head())

X = df.drop(['class'], axis=1)
print("\n==== X head ====")
print(X.head())

y = df['class']
print("\n==== y head ====")
print(y.head())

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)


print("\n================== Dataset Analysis ==================")
print("Instances: ", len(df.index) + 1)
print("Attributes: ", len(df.columns))

_, data_labels = np.unique(train_y, return_inverse=True)

print("Generating Plots Of Dataset 2")
plot_2D(train_X, data_labels)

plot_3D(train_X, data_labels)


print("\n================== Tuning Hyperparameters ==================")
print("==== Tuning RFC with Bagging ====")
nasa_rfc = RandomForestClassifier(bootstrap=True)
rfc_parameters = {
    'n_estimators': [10, 100, 1000, 10000],
    'max_depth': [100, 1000, 10000]
}
rfc_start = time.time()
rfc_clf = GridSearchCV(nasa_rfc, rfc_parameters, return_train_score=True, n_jobs=-1)
rfc_clf.fit(train_X, train_y)
rfc_clf_score = rfc_clf.score(test_X, test_y)
print("Test Score: ", rfc_clf_score, " with parameters: ", rfc_clf.best_params_)
print("Total time to run Grid Search: ", time.time() - rfc_start, " seconds")

print("Generating Table")
rfc_table = generate_table(rfc_clf.cv_results_)
print(rfc_table)


print("\n==== Tuning SVM ====")
nasa_svc = SVC()
svc_parameters = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.1, 0.001, 0.0001],
    'kernel': ['poly', 'rbf', 'sigmoid']
}

svc_start = time.time()
svc_clf = GridSearchCV(nasa_svc, svc_parameters, return_train_score=True, n_jobs=-1)
svc_clf.fit(train_X, train_y)
svc_clf_score = svc_clf.score(test_X, test_y)
print("Test Score: ", svc_clf_score, " with parameters: ", svc_clf.best_params_)
print("Total time to run Grid Search: ", time.time() - svc_start, " seconds")

print("Generating Table")
svc_table = generate_table(svc_clf.cv_results_)
print(svc_table)


print("\n==== Tuning NN ====")
nasa_nn = MLPClassifier()
nn_parameters = {
    'hidden_layer_sizes': [(5, 5), (5, 5, 5), (6, 6), (6, 6, 6), (7, 7), (7, 7, 7), (8, 8), (8, 8, 8)],
    'learning_rate': ["constant", "invscaling", "adaptive"],
}
nn_start = time.time()
nn_clf = GridSearchCV(nasa_nn, nn_parameters, return_train_score=True, n_jobs=-1)
nn_clf.fit(train_X, train_y)
nn_clf_score = nn_clf.score(test_X, test_y)
print("Test Score: ", nn_clf_score, " with parameters: ", nn_clf.best_params_)
print("Total time to run Grid Search: ", time.time() - nn_start, " seconds")

print("Generating Table")
nn_table = generate_table(nn_clf.cv_results_)
print(nn_table)

print("\n================== Optimal Model Comparison ==================")
