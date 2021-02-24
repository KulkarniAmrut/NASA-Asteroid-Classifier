# Utility Imports
import os.path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

# Model Imports
from sklearn.ensemble import RandomForestClassifier
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
    plt.clf()

    print("Generating 2D Plot")

    fig = plt.figure()
    projection_2D = pca_transform(train_x_p, 2)
    fig.gca().scatter(projection_2D[:, 0], projection_2D[:, 1], c=labels)
    fig.gca().set_title('PCA 2D Transformation')
    plt.savefig("Plot 2D")
    plt.clf()


def plot_3D(train_x_p, labels=None):
    plt.clf()

    print("Generating 3D Plot")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    projection_3D = pca_transform(train_x_p, 3)
    ax.scatter(projection_3D[:,0],projection_3D[:,1],projection_3D[:,2],c=labels)
    fig.gca().set_title('PCA 3D Transformation')
    plt.savefig("Plot 3D")
    plt.clf()

def generate_table(df_p):
    print("Generating Table")

    gen_table = pd.concat([pd.DataFrame(df_p["params"]),
                           pd.DataFrame(df_p["mean_train_score"], columns=["Mean Train Score"]),
                           pd.DataFrame(df_p["mean_test_score"], columns=["Mean Test Score"]),
                           pd.DataFrame(df_p["rank_test_score"], columns=["Rank"])], axis=1)
    return gen_table


def learning_curve_plotter(estimator, X_p, y_p, name):
    # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    from sklearn.model_selection import ShuffleSplit

    print("Generating Learning Curves")

    plt.clf()

    cvp = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator=estimator,
                                                                          X=X_p,
                                                                          y=y_p,
                                                                          cv=cvp,
                                                                          train_sizes=np.linspace(.1, 1.0, 5),
                                                                          n_jobs=-1,
                                                                          return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Testing Score")
    plt.legend(loc="best")
    plt.savefig(name)
    plt.clf()
    print("Finished Generating Learning Curves\n")

'''
# Data Loading & Pre-processing
'''
print("================== Data Loading & Preprocessing ==================")
df = pd.read_csv('data/nasa.csv')
df['Hazardous'].value_counts()
df['Hazardous'] = df['Hazardous'].map({True: 1, False: 0})
print("==== dataframe head ====")
print(df.head())

print("\n==== Removing Columns That Don't Aid Classification ====")
df_delt = df[['Absolute Magnitude', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Epoch Date Close Approach',
        'Relative Velocity km per sec', 'Miss Dist.(kilometers)', 'Orbit Uncertainity', 'Minimum Orbit Intersection',
        'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
        'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg', 'Aphelion Dist',
        'Perihelion Time', 'Mean Anomaly', 'Mean Motion', 'Hazardous']]
print("\n==== dataframe head ====")
print(df_delt.head())

X = df_delt.drop(['Hazardous'], axis=1)
print("\n==== X head ====")
print(X.head())

y = df_delt['Hazardous']
print("\n==== y head ====")
print(y.head())

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)


print("\n================== Dataset Analysis ==================")
print("Instances: ", len(df.index) + 1)
print("Attributes: ", len(df.columns))

_, data_labels = np.unique(train_y, return_inverse=True)

print("Generating Plots Of Dataset")
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
print("Total time to run Grid Search: ", time.time() - rfc_start, " seconds\n")
rfc_table = generate_table(rfc_clf.cv_results_)
print(rfc_table)


print("\n==== Tuning SVM ====")
nasa_svc = SVC()
svc_parameters = {
    'C': [0.001, 0.1, 1, 10, 100, 1000],
    'kernel': ['poly', 'rbf', 'sigmoid']
}

svc_start = time.time()
svc_clf = GridSearchCV(nasa_svc, svc_parameters, return_train_score=True, n_jobs=-1)
svc_clf.fit(train_X, train_y)
svc_clf_score = svc_clf.score(test_X, test_y)
print("Test Score: ", svc_clf_score, " with parameters: ", svc_clf.best_params_)
print("Total time to run Grid Search: ", time.time() - svc_start, " seconds\n")
svc_table = generate_table(svc_clf.cv_results_)
print(svc_table)


print("\n==== Tuning NN ====")
nasa_nn = MLPClassifier()
nn_parameters = {
    'hidden_layer_sizes': [(3, 4), (4, 3), (5, 5), (5, 5, 5), (6, 6), (6, 6, 6), (7, 7), (7, 7, 7), (8, 8), (8, 8, 8)],
    'learning_rate': ["constant", "invscaling", "adaptive"]
}
nn_start = time.time()
nn_clf = GridSearchCV(nasa_nn, nn_parameters, return_train_score=True, n_jobs=-1)
nn_clf.fit(train_X, train_y)
nn_clf_score = nn_clf.score(test_X, test_y)
print("Test Score: ", nn_clf_score, " with parameters: ", nn_clf.best_params_)
print("Total time to run Grid Search: ", time.time() - nn_start, " seconds\n")
nn_table = generate_table(nn_clf.cv_results_)
print(nn_table)

print("\n================== Optimal Model Comparison ==================")
rfc_opt = RandomForestClassifier(max_depth=rfc_clf.best_params_['max_depth'],
                                 n_estimators=rfc_clf.best_params_['n_estimators'],
                                 bootstrap=True)
print("=== Optimal RFC Params ===")
print(rfc_opt.get_params())
learning_curve_plotter(rfc_opt, X, y, 'Optimal RFC Learning Curve')


svc_opt = SVC(C=svc_clf.best_params_['C'],
              kernel=svc_clf.best_params_['kernel'])
print("\n=== Optimal SVC Params ===")
print(svc_opt.get_params())
learning_curve_plotter(svc_opt, X, y, 'Optimal SVC Learning Curve')


nn_opt = MLPClassifier(hidden_layer_sizes=nn_clf.best_params_['hidden_layer_sizes'],
                       learning_rate=nn_clf.best_params_['learning_rate'])
print("\n=== Optimal NN Params ===")
print(nn_opt.get_params())
learning_curve_plotter(nn_opt, X, y, 'Optimal NN Learning Curve')

print("\n====== Test Score Comparisons ======")
print("Generating Bar Graph")
plt.clf()
labels = ('Random Forest Test Score', 'SVM Test Score', 'Neural Net Test Score')
rfc_opt.fit(train_X, train_y)
svc_opt.fit(train_X, train_y)
nn_opt.fit(train_X, train_y)
bar_scores = [rfc_opt.score(test_X, test_y), svc_opt.score(test_X, test_y), nn_opt.score(test_X, test_y)]
y_pos = np.arange(len(labels))
plt.bar(y_pos, bar_scores)
plt.xticks(y_pos, labels)
plt.savefig('Optimized Model Test Scores')
print("Finished Generating Bar Graph")