import pandas as pd
import numpy as np



np.random.seed(0)

class Data_manager(object):
    def __init__(self):
        self.self = self
        self.debug_counter = 0

    def load_data(self, config, shuffle = True):
        if self.debug_counter == 0:
            self.debug_counter = +1
            data = pd.read_csv(config.data["dir"])
            data_ctnuous = data[config.data["continuous_vars"]]
            data_ctnuous = data_ctnuous/data_ctnuous.max(axis=0)
            data_nominal = data[config.data["nominal_vars"]]
            data_ordinal = data[config.data["ordinal_vars"]]/data[config.data["ordinal_vars"]].max()

            # concatenate continuous, nominal, ordinal variables
            data_inputs = pd.concat([data_ctnuous, data_nominal, data_ordinal], axis = 1)

            # Variables of data_labels = [status, Stage]
            data_labels = data[config.data["labels"]]

            if shuffle is True:
                shuffle_idx = np.random.permutation(len(data_inputs))
                data_inputs = data_inputs.iloc[shuffle_idx]
                data_labels = data_labels.iloc[shuffle_idx]

            ### code for def cross_valid_info
            self.input_shape = np.shape(data_inputs)
            self.labels_shape = np.shape(data_labels)

        elif self.debug_counter == 1:
            print("You already call load_data method!")
            raise
        else:
            print("Something was wrong!")
            raise

        self.n_abnormal = len([x for x in data_labels["status"] if x == 1])
        self.n_normal = len([x for x in data_labels["status"] if x == 0])

        return data_inputs, data_labels

    def split_data4crossvalid(self, inputs, labels, n_fold):
        """
        :param n_fold: number of fold for cross validation, dtpye = integer
        :return: dictionary for cross validation.
        """
        if self.debug_counter == 1:
            self.debug_counter += 1
            self.n_fold = n_fold
            self.n_data = len(labels)
            self.n_ele_fold = []

            n_element_fold = self.n_data//n_fold
            self.cv_dict = {}

            for i in range(self.n_fold):
                if i != (self.n_fold - 1):
                    self.cv_dict[str(i) + "_inputs"] = inputs[n_element_fold*i: n_element_fold*(i+1)]
                    self.cv_dict[str(i) + "_labels"] = labels[n_element_fold*i: n_element_fold*(i+1)]
                    self.n_ele_fold.append(n_element_fold)
                elif i == (self.n_fold - 1):
                    self.cv_dict[str(i) + "_inputs"] = inputs[n_element_fold*i: ]
                    self.cv_dict[str(i) + "_labels"] = labels[n_element_fold*i: ]
                    self.n_ele_fold.append(len(self.cv_dict[str(i) + "_labels"]))

        else:
            print("You should load your data first using load_data() method!")
            raise

        return self.cv_dict

    def cross_valid_info(self):
        if self.debug_counter == 2:
            print("_"*80)
            print("Inputs shape: ", self.input_shape)
            print("Labels shape: ", self.labels_shape)
            print("")
            print("N_normal: ", self.n_normal)
            print("N_abnormal: ", self.n_abnormal)
            print("R(Abnormal/Total) = %.2f" % (self.n_abnormal/self.input_shape[0]))
            print("")
            print("Number of validation fold: ", self.n_fold)
            print("Number of element per fold: ", self.n_ele_fold)
            print("_" * 80)
        else:
            print("Now debug counter value: %d plz load or split data first." % (self.debug_counter))
