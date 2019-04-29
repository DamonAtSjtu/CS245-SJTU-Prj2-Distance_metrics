"""
Main function to compute kNN together with different distance metrics and metric learning methods.
"""
import os
import logging
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from distance_metrics import cosine_distance, earth_mover_distance
from utils import save_obj
import metric_learn as ml


USE_PCA = False
PCA_DIMENSION = 50

USE_T_SNE_VISUALIZATION = True

BUILTIN_DISTANCE_METRICS = ['euclidean', 'manhattan', 'chebyshev']
CUSTOM_DISTANCE_METRICS = {'cosine': cosine_distance}

METRIC_LEARNING_METHODS = [('CM', ml.covariance.Covariance, 'unsupervised'),
                           ('LMNN', ml.lmnn.LMNN, 'supervised', {'k': None, 'regularization': 0.5}),
                           ('LFDA', ml.lfda.LFDA, 'supervised'),
                           ('LSML_Supervised', ml.lsml.LSML_Supervised, 'supervised', {'tol': 0.001}),
                           ('SDML_Supervised', ml.sdml.SDML_Supervised, 'supervised', {'balance_param': 0.5, 'sparsity_param': 0.01}),
                           ('RCA_Supervised', ml.rca.RCA_Supervised, 'supervised'),
                           ('MMC_Supervised', ml.mmc.MMC_Supervised, 'supervised')]


# Generate k values
def k_neighbors_iterator():
    return [2,3,4,5,6,7,8]


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


if __name__ == "__main__":

    # Register logger.
    logger = logging.getLogger('knn')
    file_log_handler = logging.FileHandler('knn.log')
    stdout_log_handler = logging.StreamHandler()

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_log_handler.setFormatter(formatter)
    stdout_log_handler.setFormatter(formatter)

    logger.addHandler(file_log_handler)
    logger.addHandler(stdout_log_handler)
    logger.setLevel(logging.INFO)

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

        if USE_PCA:
            pca = PCA(n_components=PCA_DIMENSION)
            train_data = pca.fit_transform(train_data)
            test_data = pca.fit_transform(test_data)
            np.save(npy_dir + 'train_data_pca_' + str(PCA_DIMENSION) + '.npy', train_data)
            np.save(npy_dir + 'test_data_pca_' + str(PCA_DIMENSION) + '.npy', test_data)

    else:
        print('.npy files found. Loading...')
        if USE_PCA:
            if os.path.isfile(npy_dir + 'train_data_pca_' + str(PCA_DIMENSION) + '.npy') and \
                 os.path.isfile(npy_dir + 'test_data_pca_' + str(PCA_DIMENSION) + '.npy'):

                train_data = np.load(npy_dir + 'train_data_pca_' + str(PCA_DIMENSION) + '.npy')
                test_data = np.load(npy_dir + 'test_data_pca_' + str(PCA_DIMENSION) + '.npy')
            else:
                train_data = np.load(npy_dir + 'train_data.npy')
                test_data = np.load(npy_dir + 'test_data.npy')

                pca = PCA(n_components=PCA_DIMENSION)
                train_data = pca.fit_transform(train_data)
                test_data = pca.fit_transform(test_data)
                np.save(npy_dir + 'train_data_pca_' + str(PCA_DIMENSION) + '.npy', train_data)
                np.save(npy_dir + 'test_data_pca_' + str(PCA_DIMENSION) + '.npy', test_data)
        else:
            train_data = np.load(npy_dir + 'train_data.npy')
            test_data = np.load(npy_dir + 'test_data.npy')

        pca = PCA(n_components=PCA_DIMENSION)
        train_data = pca.fit_transform(train_data)
        test_data = pca.fit_transform(test_data)
        np.save(npy_dir + 'train_data_pca_' + str(PCA_DIMENSION) + '.npy', train_data)
        np.save(npy_dir + 'test_data_pca_' + str(PCA_DIMENSION) + '.npy', test_data)

        train_label = np.load(npy_dir + 'train_label.npy')
        test_label = np.load(npy_dir + 'test_label.npy')
        print('.npy files loaded.')

    # K folds cross validation.
    fold_num = 5
    kf = KFold(n_splits=fold_num, random_state=1, shuffle=True)
    kf.get_n_splits(train_data)

    # 1. Distance metric
    acc_dict = {}
    top_acc_dict = {}

    custom_distance_metrics_list = [cdm for cdm in CUSTOM_DISTANCE_METRICS]
    all_distance_metrics = BUILTIN_DISTANCE_METRICS + custom_distance_metrics_list
    logger.info('Distance metrics: ' + str(all_distance_metrics))
    if USE_PCA:
        logger.info('PCA: ' + str(PCA_DIMENSION))

    for distance_metric in BUILTIN_DISTANCE_METRICS:

        acc_dict.update({distance_metric: {}})
        k_neighbors_list = k_neighbors_iterator()
        logger.info('Distance metric: ' + distance_metric)
        top_acc = 0
        top_k = 0

        for k_neighbors in k_neighbors_list:

            if distance_metric in custom_distance_metrics_list:
                distance_metric_func = CUSTOM_DISTANCE_METRICS[distance_metric]
                neighbor = KNeighborsClassifier(n_neighbors=k_neighbors,
                                                metric='pyfunc',
                                                metric_params={'func': distance_metric_func}
                                                )
            else:
                neighbor = KNeighborsClassifier(n_neighbors=k_neighbors,
                                                metric=distance_metric
                                                )

            fold_acc = []
            fold_index = 0
            for train_index, validation_index in kf.split(train_data):
                X_train, y_train = train_data[train_index], train_label[train_index]
                X_validation, y_validation = train_data[validation_index], train_label[validation_index]

                neighbor.fit(X_train, y_train)
                y_pred = neighbor.predict(X_validation)

                correct_count = (y_pred == y_validation).sum()
                total_count = np.shape(y_validation)[0]
                fold_acc.append(float(correct_count) / float(total_count))
                print('Fold', fold_index, 'fold acc:', float(correct_count) / float(total_count))
                fold_index += 1

            acc = np.mean(fold_acc)
            if top_acc < acc:
                top_acc = acc
                top_k = k_neighbors

            acc_dict[distance_metric].update({k_neighbors: acc})
            logger.info("k: " + str(k_neighbors) + "\tAcc: " + str(acc))

        top_acc_dict.update({distance_metric: [top_k, top_acc]})

        if USE_PCA:
            save_obj(top_acc, 'top_acc_PCA_' + str(PCA_DIMENSION) + '_dict')
            save_obj(acc_dict, 'acc_PCA_' + str(PCA_DIMENSION) + '_dict')
        else:
            save_obj(top_acc, 'top_acc_dict')
            save_obj(acc_dict, 'acc_dict')

    logger.info('Best result:')
    logger.info('Metric  -  k  -  Accuracy')
    for key, value in top_acc_dict.items():
        logger.info(key + '  -  ' + str(value[0]) + '  -  ' + str(value[1]))
    logger.info('\n\n')

    # 2. Metric learn
    acc_dict = {}
    top_acc_dict = {}

    for metric_learning_method in METRIC_LEARNING_METHODS:

        ml_wrapper = MetricLearningMethod(*metric_learning_method)

        acc_dict.update({ml_wrapper.name: {}})
        k_neighbors_list = k_neighbors_iterator()
        logger.info('Metric learning method: ' + ml_wrapper.name)
        top_acc = 0
        top_k = 0

        for k_neighbors in k_neighbors_list:

            neighbor = KNeighborsClassifier(n_neighbors=k_neighbors,
                                            metric='euclidean'
                                            )

            if 'k' in ml_wrapper.extra_parameters:
                ml_wrapper.extra_parameters['k'] = k_neighbors

            ml_instance = ml_wrapper.instance()

            fold_acc = []
            fold_index = 0
            for train_index, validation_index in kf.split(train_data):
                X_train, y_train = train_data[train_index], train_label[train_index]
                X_validation, y_validation = train_data[validation_index], train_label[validation_index]

                try:
                    if ml_wrapper.supervised == 'supervised':
                        X_train = ml_instance.fit_transform(X_train, y_train)
                        X_validation = ml_instance.transform(X_validation)
                    elif ml_wrapper.supervised == 'weakly_supervised' or ml_wrapper.supervised == 'unsupervised':
                        X_train = ml_instance.fit_transform(X_train)
                        X_validation = ml_instance.transform(X_validation)
                    else:
                        raise MetricLearningInfoError
                    transformer = ml_instance.transformer()
                    metric = ml_instance.metric()

                except MetricLearningInfoError as e:
                    print('Metric learning info error')

                neighbor.fit(X_train, y_train)
                y_pred = neighbor.predict(X_validation)

                correct_count = (y_pred == y_validation).sum()
                total_count = np.shape(y_validation)[0]
                fold_acc.append(float(correct_count) / float(total_count))
                print('Fold', fold_index, 'fold acc:', float(correct_count) / float(total_count))
                fold_index += 1

            acc = np.mean(fold_acc)
            if top_acc < acc:
                top_acc = acc
                top_k = k_neighbors

            acc_dict[ml_wrapper.name].update({k_neighbors: acc})
            logger.info("k: " + str(k_neighbors) + "\tAcc: " + str(acc))

        top_acc_dict.update({ml_wrapper.name: [top_k, top_acc]})

        if USE_PCA:
            save_obj(top_acc, 'ml_top_acc_PCA_' + str(PCA_DIMENSION) + '_dict')
            save_obj(acc_dict, 'ml_acc_PCA_' + str(PCA_DIMENSION) + '_dict')
        else:
            save_obj(top_acc, 'ml_top_acc_dict')
            save_obj(acc_dict, 'ml_acc_dict')

    logger.info('Best result:')
    logger.info('ML method  -  k  -  Accuracy')
    for key, value in top_acc_dict.items():
        logger.info(key + '  -  ' + str(value[0]) + '  -  ' + str(value[1]))
    logger.info('\n\n')


