#!/usr/bin/python3

import os
import os.path
import numpy as np
import scipy.io
import xlrd

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

ONE_TIME_FALL_DATASET = 'one'  # Value to pass to load the one time fallers + controls data set
MULTI_TIME_FALL_DATASET = 'multi'  # Value to pass to load the multi time fallers + controls data set


class DataLoader:
    base_path = './Data espen/'
    train_fraction = 0.80  # fraction of subjects used for train set. number of segments per subject is variable.
    validation_fraction = 0.10  # fraction of subjects used for validation. number of segments per subject is variable.

    CONTROL_LABEL = 0
    ONE_TIME_FALLER_LABEL = 1
    MULTI_TIME_FALLER_LABEL = 2

    def load(self, dataset_selection=ONE_TIME_FALL_DATASET):
        """
        Gets subject ids from excel file and loads acc and vel data from mat file for each subject. Return a train,
        validation and test set. Each set consists of the data X and the label y.
        :param dataset_selection:
        Determines whether datasets from the single time fallers and their controls or the multi time fallers with
        their controls should be loaded. Can be either 'one' or 'multi'.
        :return: train_X, train_y, validation_X,
        validation_y, test_X, test_y
        """
        multi_time_fallers, multi_time_fallers_controls, one_time_fallers, one_time_fallers_controls = self.read_ids_from_excel()
        logger.debug('')

        if dataset_selection == ONE_TIME_FALL_DATASET:
            train_X, train_y, validation_X, validation_y, test_X, test_y = self.get_split_shuffled_data_set(
                self.ONE_TIME_FALLER_LABEL, self.CONTROL_LABEL, one_time_fallers, one_time_fallers_controls)
        elif dataset_selection == MULTI_TIME_FALL_DATASET:
            train_X, train_y, validation_X, validation_y, test_X, test_y = self.get_split_shuffled_data_set(
                self.MULTI_TIME_FALLER_LABEL, self.CONTROL_LABEL, multi_time_fallers, multi_time_fallers_controls)

        logger.info('Loaded train samples with shape {} and train labels with shape {}.'
                    .format(train_X.shape, train_y.shape))
        logger.info('Loaded validation samples with shape {} and test labels with shape {}.'
                    .format(validation_X.shape, validation_y.shape))
        logger.info('Loaded test samples with shape {} and test labels with shape {}.'
                    .format(test_X.shape, test_y.shape))
        logger.info('Of {} instances loaded, {}% is used for training, {}% for validation, {}% for testing.'
                    .format(len(train_y) + len(test_y) + len(validation_y),
                            np.round(100.0 * len(train_y) / (len(train_y) + len(test_y) + len(validation_y)), 1),
                            np.round(100.0 * len(validation_y) / (len(train_y) + len(test_y) + len(validation_y)), 1),
                            np.round(100.0 * len(test_y) / (len(train_y) + len(test_y) + len(validation_y)), 1)))

        return train_X, train_y, validation_X, validation_y, test_X, test_y

    def read_ids_from_excel(self):
        sheet = xlrd.open_workbook(os.path.join(self.base_path, 'File_number_Fall_class.xlsx')).sheet_by_index(0)
        one_time_fallers = self.get_ids_from_column(1, sheet)
        one_time_fallers_controls = self.get_ids_from_column(3, sheet)
        multi_time_fallers = self.get_ids_from_column(6, sheet)
        multi_time_fallers_controls = self.get_ids_from_column(8, sheet)
        return multi_time_fallers, multi_time_fallers_controls, one_time_fallers, one_time_fallers_controls

    def get_ids_from_column(self, column, sheet):
        return list(
            [int(sheet.cell_value(i, column)) for i in range(2, sheet.nrows) if sheet.cell_value(i, column) != ''])

    def get_split_shuffled_data_set(self, label, control_label, fallers, controls):
        indices = list(range(len(fallers)))
        np.random.shuffle(indices)

        n_train_instances = int(self.train_fraction * len(indices))
        n_validation_instances = int(self.validation_fraction * len(indices))
        logger.info('Loading training data.')
        train_X, train_y = self.get_data_set(fallers,
                                             controls,
                                             indices[:n_train_instances],
                                             label,
                                             control_label)
        logger.info('Loading validation data.')
        validation_X, validation_y = self.get_data_set(fallers,
                                                       controls,
                                                       indices[
                                                       n_train_instances:n_train_instances + n_validation_instances],
                                                       label,
                                                       control_label)
        logger.info('Loading test data.')
        test_X, test_y = self.get_data_set(fallers,
                                           controls,
                                           indices[n_train_instances + n_validation_instances:],
                                           label,
                                           control_label)
        return train_X, train_y, validation_X, validation_y, test_X, test_y

    def get_data_set(self, fallers, controls, indices, label, control_label):
        train_instance_sets = []
        train_label_sets = []
        for index in indices:
            fall_id = fallers[index]
            fall_X, fall_y = self.get_user_data_and_labels_for_id(fall_id, label)
            train_instance_sets.append(fall_X)
            train_label_sets.append(fall_y)

            control_id = controls[index]
            control_X, control_y = self.get_user_data_and_labels_for_id(control_id, control_label)
            train_instance_sets.append(control_X)
            train_label_sets.append(control_y)
        train_set = np.concatenate(train_instance_sets, axis=0)
        train_labels = np.concatenate(train_label_sets)
        return train_set, train_labels

    def get_user_data_and_labels_for_id(self, id, label):
        filename = 'Acc_Vel_gait_30sec_{}.mat'.format(id)
        logger.info('Processing file {}'.format(filename))
        user_data = self.load_user_data(filename)
        user_labels = [label for _ in user_data]
        return user_data, user_labels

    def load_user_data(self, filename):
        path = os.path.join(self.base_path, filename)
        data = scipy.io.loadmat(path)
        acc = np.array([data['Acc_gait_30sec'][0][i] for i in range(len(data['Acc_gait_30sec'][0]))])
        vel = np.array([data['Vel_gait_30sec'][0][i] for i in range(len(data['Vel_gait_30sec'][0]))])
        userdata = np.concatenate((acc, vel), axis=2)
        return userdata


def load_one_time_fall_dataset():
    """
    Loads a dataset containing the one time fallers and there matched controls. Fallers are distributed over train,
    validation and test set. Controls are kept in the same set as their matched subjects. All segments of a specific
    subject, control of faller, end up in the same set. Gets subject ids from excel file and loads acc and vel data
    from mat file for each subject. Return a train, validation and test set. Each set consists of the data X and the
    label y.
    :return: train_X, train_y, validation_X, validation_y, test_X, test_y
    """
    return DataLoader().load(dataset_selection=ONE_TIME_FALL_DATASET)


def load_multi_time_fall_dataset():
    """
    Loads a dataset containing the multiple time fallers and there matched controls. Fallers are distributed over
    train, validation and test set. Controls are kept in the same set as their matched subjects. All segments of a
    specific subject, control of faller, end up in the same set. Gets subject ids from excel file and loads acc and
    vel data from mat file for each subject. Return a train, validation and test set. Each set consists of the data X
    and the label y.
    :return: train_X, train_y, validation_X, validation_y, test_X, test_y
    """
    return DataLoader().load(dataset_selection=MULTI_TIME_FALL_DATASET)
