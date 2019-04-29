"""
Use t-sne to visualize the distributions of new feature vectors X' = X * projection_matrix
with different metric learning methods.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import metric_learn as ml
from sklearn.model_selection import KFold

METRIC_LEARNING_METHODS = [('CM', ml.covariance.Covariance, 'unsupervised'),
                           ('LMNN', ml.lmnn.LMNN, 'supervised', {'k': None, 'regularization': 0.5}),
                           ('LFDA', ml.lfda.LFDA, 'supervised'),
                           ('LSML_Supervised', ml.lsml.LSML_Supervised, 'supervised', {'tol': 0.001}),
                           ('SDML_Supervised', ml.sdml.SDML_Supervised, 'supervised', {'balance_param': 0.5, 'sparsity_param': 0.
                           ('RCA_Supervised', ml.rca.RCA_Supervised, 'supervised'),
                           ('MMC_Supervised', ml.mmc.MMC_Supervised, 'supervised')]


class MetricLearningMethod:
    def __init__(self, name, call, supervised, extra_parameters={}):
        self.name = name
        self.call = call
        self.supervised = supervised
        self.extra_parameters = extra_parameters

    def instance(self):
        return self.call(**self.extra_parameters)


class MetricLearningInfoError(Exception):

    def __init__(self):
        pass



if __name__ == '__main__':

    # Load data files or generate them.
    npy_dir = "dataset_npy_file/"
    if not os.path.exists(npy_dir):
        print('No .npy files, generate from .txt files.')
        os.makedirs(npy_dir)
        train_data = np.genfromtxt('data_process(float)/data_train.txt', dtype=np.float)
        train_label = np.genfromtxt('data_process(float)/label_train.txt', dtype=np.float)
        print('Train data loaded.')
        test_data = np.genfromtxt('data_process (float)/data_test.txt', dtype=np.float)
        test_label = np.genfromtxt('data_process(float)/label_test.txt', dtype=np.float)
        print('Test data loaded.')

        print('Saving .npy files.')
        np.save(npy_dir + 'train_data.npy', train_data)
        np.save(npy_dir + 'train_label.npy', train_label)
        np.save(npy_dir + 'test_data.npy', test_data)
        np.save(npy_dir + 'test_label.npy', test_label)
        print('.npy files saved.')

    else:
        print('.npy files found. Loading...')

        train_data = np.load(npy_dir + 'train_data.npy')
        test_data = np.load(npy_dir + 'test_data.npy')
        train_label = np.load(npy_dir + 'train_label.npy')
        test_label = np.load(npy_dir + 'test_label.npy')
        print('.npy files loaded.')

    fold_num = 2
    kf = KFold(n_splits=fold_num, random_state=1, shuffle=True)
    kf.get_n_splits(train_data)

    # Control group
    for train_index, validation_index in kf.split(train_data):
        print(len(train_index))
        print(len(validation_index))
        X_train, y_train = train_data[train_index], train_label[train_index]

        print('Control group')
        plt.cla()
        tsne = TSNE(n_components=2)
        tsne_data = tsne.fit_transform(X_train)
        print('tsne_data')
        tsne_min, tsne_max = tsne_data.min(0), tsne_data.max(0)
        tsne_data = (tsne_data - tsne_min) / (tsne_max - tsne_min)

        plt.figure(figsize=(6, 6))
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=y_train)
        plt.savefig('figures/original.jpg', dpi=500)

        k_neighbors = 3

        for metric_learning_method in METRIC_LEARNING_METHODS:

            ml_wrapper = MetricLearningMethod(*metric_learning_method)
            print(ml_wrapper.name)

            if 'k' in ml_wrapper.extra_parameters:
                ml_wrapper.extra_parameters['k'] = k_neighbors

            ml_instance = ml_wrapper.instance()

            try:
                if ml_wrapper.supervised == 'supervised':
                    ml_data = ml_instance.fit_transform(X_train, y_train)
                elif ml_wrapper.supervised == 'weakly_supervised' or ml_wrapper.supervised == 'unsupervised':
                    ml_data = ml_instance.fit_transform(X_train)
                else:
                    raise MetricLearningInfoError
                transformer = ml_instance.transformer()
                metric = ml_instance.metric()

            except MetricLearningInfoError as e:
                print('Metric learning info error')
            print(ml_wrapper.name + ' over')
            plt.cla()
            tsne = TSNE(n_components=2)
            tsne_data = tsne.fit_transform(ml_data)
            print('tsne_data')
            tsne_min, tsne_max = tsne_data.min(0), tsne_data.max(0)
            tsne_data = (tsne_data - tsne_min) / (tsne_max - tsne_min)

            plt.figure(figsize=(6, 6))
            plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=y_train)
            plt.savefig('figures/' + ml_wrapper.name + '.jpg', dpi=500)

        break
