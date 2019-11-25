import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import dataloader as dl
from utils import yaml_utils
from utils.EDA import *
from utils.utils import *
from models import *


torch.manual_seed(777)
torch.cuda.manual_seed_all(777)

class main():
    def __init__(self, args):
        self.args = args

    def load_data4cv(self):
        print_arg(args)
        self.config = yaml_utils.Config(yaml.load(open(self.args.config_path)))

        parkinson_dm = dl.Data_manager()
        self.parkinson_inputs, self.parkinson_labels = parkinson_dm.load_data(self.config, shuffle = True)
        self.pks_dict = parkinson_dm.split_data4crossvalid(self.parkinson_inputs, self.parkinson_labels, self.args.n_cv_folds)
        parkinson_dm.cross_valid_info()
        self.n_inst = np.shape(self.parkinson_inputs)[0]
        self.n_vars = np.shape(self.parkinson_inputs)[1]

    def perform_EDA(self):
        ### Exploratory Data Analysis
        df4corr = pd.concat([self.parkinson_inputs, self.parkinson_labels['status']], axis = 1)
        chi_square_normality_test(self.parkinson_inputs)
        chi_square_normality_test(self.parkinson_inputs, pca = True)
        plot_corr_heatmap(self.config, df4corr)
        plot_corr_heatmap_after_pca(self.config, df4corr)
        pca_biplot(self.config, df4corr)

    def Naive_logistic_regression(self):
        ### naive logistic regression.
        print("_"*80)
        print("Apply naive logistic regression")
        log_accuracy_holder = []
        log_auroc_holder = []
        for i in range(self.args.n_cv_folds):
            train_fold_list = list(range(self.args.n_cv_folds))
            train_fold_list = [x for x in train_fold_list if x != i]

            temp = 0
            for fold in train_fold_list:
                if temp == 0:
                    train_inputs = self.pks_dict[str(fold) + "_inputs"]
                    train_labels = self.pks_dict[str(fold) + "_labels"]
                    temp = 1
                else:
                    train_inputs = pd.concat([train_inputs, self.pks_dict[str(fold) + "_inputs"]], axis = 0)
                    train_labels = pd.concat([train_labels, self.pks_dict[str(fold) + "_labels"]], axis = 0)

            test_inputs = self.pks_dict[str(i) + "_inputs"]
            test_labels = self.pks_dict[str(i) + "_labels"]

            logistic_acc, logistic_auroc = train_eval_logistic_regression(self.args,
                                                                          self.config,
                                                                          train_inputs,
                                                                          train_labels["status"],
                                                                          test_inputs,
                                                                          test_labels["status"],
                                                                          penalty = 'none')
            log_accuracy_holder.append(logistic_acc)
            log_auroc_holder.append(logistic_auroc)

        print("CV Accuracy(Naive Logistic Regression): %f " % np.mean(log_accuracy_holder))
        print("CV Auroc(Naive Logistic Regression: %f" % np.mean(log_auroc_holder))

    def PCA_logistic_regression(self):
        ### PCA logistic regression
        print("_"*80)
        print("Apply logistic regression after applying PCA")
        ls_pca_log_accuracy_holder = {}
        ls_pca_log_auroc_holder = {}
        for reduced_dim in range(1,(self.n_vars + 1)):
            pca_log_accuracy_holder = []
            pca_log_auroc_holder = []
            for i in range(self.args.n_cv_folds):
                train_fold_list = list(range(self.args.n_cv_folds))
                train_fold_list = [x for x in train_fold_list if x != i]

                temp = 0
                for fold in train_fold_list:
                    if temp == 0:
                        train_inputs = self.pks_dict[str(fold) + "_inputs"]
                        train_labels = self.pks_dict[str(fold) + "_labels"]
                        temp = 1
                    else:
                        train_inputs = pd.concat([train_inputs, self.pks_dict[str(fold) + "_inputs"]], axis = 0)
                        train_labels = pd.concat([train_labels, self.pks_dict[str(fold) + "_labels"]], axis = 0)

                test_inputs = self.pks_dict[str(i) + "_inputs"]
                test_labels = self.pks_dict[str(i) + "_labels"]

                good_of_fit, pca_logistic_acc, pca_logistic_auroc = train_eval_pca_logistic_regression(self.args,
                                                                                                       self.config,
                                                                                                       train_inputs,
                                                                                                       train_labels["status"],
                                                                                                       test_inputs,
                                                                                                       test_labels["status"],
                                                                                                       reduced_dim,
                                                                                                       penalty = 'none')
                pca_log_accuracy_holder.append(pca_logistic_acc)
                pca_log_auroc_holder.append(pca_logistic_auroc)

            ls_pca_log_accuracy_holder[str(reduced_dim)] = np.mean(pca_log_accuracy_holder)
            ls_pca_log_auroc_holder[str(reduced_dim)] = np.mean(pca_log_auroc_holder)

        plot_scree_plot(self.config, good_of_fit, num_vars = 6)
        plot_1d_ls_acc_plot(self.config, ls_pca_log_accuracy_holder, "Reduced Dimension", "Accuracy of Logistic Regression using PCA")
        plot_1d_ls_auroc_plot(self.config, ls_pca_log_accuracy_holder, "Reduced Dimension", "Auroc of Logistic Regression using PCA")
        ls_pca_log_acc_maxkey = max(ls_pca_log_accuracy_holder, key=ls_pca_log_accuracy_holder.get)
        ls_pca_log_maxacc = ls_pca_log_accuracy_holder[ls_pca_log_acc_maxkey]
        ls_pca_log_auroc_maxkey = max(ls_pca_log_auroc_holder, key=ls_pca_log_auroc_holder.get)
        ls_pca_log_maxauroc = ls_pca_log_auroc_holder[ls_pca_log_auroc_maxkey]

        print("CV Accuracy by input dimension(PCA Logistic Regression): ", ls_pca_log_accuracy_holder)
        print("Max Accuracy:\t key %s\t Value\t %f" %(ls_pca_log_acc_maxkey, ls_pca_log_maxacc))
        print("")
        print("CV Auroc by input dimension(PCA Logistic Regression): ", ls_pca_log_auroc_holder)
        print("Max Auroc:\t key %s\t Value\t %f" %(ls_pca_log_auroc_maxkey, ls_pca_log_maxauroc))


    def LDA(self):
        ### Linear Discriminant Analysis.
        print("_"*80)
        print("Apply Linear Discriminant Analysis")
        lda_accuracy_holder = []
        lda_auroc_holder = []
        for i in range(self.args.n_cv_folds):
            train_fold_list = list(range(self.args.n_cv_folds))
            train_fold_list = [x for x in train_fold_list if x != i]

            temp = 0
            for fold in train_fold_list:
                if temp == 0:
                    train_inputs = self.pks_dict[str(fold) + "_inputs"]
                    train_labels = self.pks_dict[str(fold) + "_labels"]
                    temp = 1
                else:
                    train_inputs = pd.concat([train_inputs, self.pks_dict[str(fold) + "_inputs"]], axis = 0)
                    train_labels = pd.concat([train_labels, self.pks_dict[str(fold) + "_labels"]], axis = 0)

            test_inputs = self.pks_dict[str(i) + "_inputs"]
            test_labels = self.pks_dict[str(i) + "_labels"]

            lda_acc, lda_auroc = train_eval_LDA(self.args,
                                                self.config,
                                                train_inputs,
                                                train_labels["status"],
                                                test_inputs,
                                                test_labels["status"])
            lda_accuracy_holder.append(lda_acc)
            lda_auroc_holder.append(lda_auroc)

        print("CV Accuracy(Linear Discriminant Analysis): %f " % np.mean(lda_accuracy_holder))
        print("CV Auroc(Linear Discriminant Analysis): %f " % np.mean(lda_auroc_holder))

    def PCA_LDA(self):
        ### PCA Linear Discriminant Analysis.
        print("_"*80)
        print("Apply Linear Discriminant Analysis after applying PCA")
        ls_pca_lda_accuracy_holder = {}
        ls_pca_lda_auroc_holder = {}
        for reduced_dim in range(1, (self.n_vars + 1)):
            pca_lda_accuracy_holder = []
            pca_lda_auroc_holder = []
            for i in range(self.args.n_cv_folds):
                train_fold_list = list(range(self.args.n_cv_folds))
                train_fold_list = [x for x in train_fold_list if x != i]

                temp = 0
                for fold in train_fold_list:
                    if temp == 0:
                        train_inputs = self.pks_dict[str(fold) + "_inputs"]
                        train_labels = self.pks_dict[str(fold) + "_labels"]
                        temp = 1
                    else:
                        train_inputs = pd.concat([train_inputs, self.pks_dict[str(fold) + "_inputs"]], axis=0)
                        train_labels = pd.concat([train_labels, self.pks_dict[str(fold) + "_labels"]], axis=0)

                test_inputs = self.pks_dict[str(i) + "_inputs"]
                test_labels = self.pks_dict[str(i) + "_labels"]

                good_of_fit, pca_lda_acc, pca_lda_auroc = train_eval_pca_LDA(self.args,
                                                                             self.config,
                                                                             train_inputs,
                                                                             train_labels["status"],
                                                                             test_inputs,
                                                                             test_labels["status"],
                                                                             reduced_dim)
                pca_lda_accuracy_holder.append(pca_lda_acc)
                pca_lda_auroc_holder.append(pca_lda_auroc)

            ls_pca_lda_accuracy_holder[str(reduced_dim)] = np.mean(pca_lda_accuracy_holder)
            ls_pca_lda_auroc_holder[str(reduced_dim)] = np.mean(pca_lda_auroc_holder)

        plot_1d_ls_acc_plot(self.config, ls_pca_lda_accuracy_holder, "Reduced Dimension", "Accuracy of Linear Discriminant Analysis using PCA")
        plot_1d_ls_auroc_plot(self.config, ls_pca_lda_auroc_holder, "Reduced Dimension", "Auroc of Linear Discriminant Analysis using PCA")
        ls_pca_lda_acc_maxkey = max(ls_pca_lda_accuracy_holder, key=ls_pca_lda_accuracy_holder.get)
        ls_pca_lda_maxacc = ls_pca_lda_accuracy_holder[ls_pca_lda_acc_maxkey]
        ls_pca_lda_auroc_maxkey = max(ls_pca_lda_auroc_holder, key=ls_pca_lda_auroc_holder.get)
        ls_pca_lda_maxauroc = ls_pca_lda_auroc_holder[ls_pca_lda_auroc_maxkey]
        print("CV Accuracy by input dimension(PCA LDA): ", ls_pca_lda_accuracy_holder)
        print("Max Accuracy:\t key %s\t Value\t %f" % (ls_pca_lda_acc_maxkey, ls_pca_lda_maxacc))
        print("")
        print("CV Auroc by input dimension(PCA LDA): ", ls_pca_lda_auroc_holder)
        print("Max Auroc:\t key %s\t Value\t %f" % (ls_pca_lda_auroc_maxkey, ls_pca_lda_maxauroc))

    def KSVM(self):
        ### Support Vector Machine using kernel trick.
        print("_"*80)
        print("Apply Kernel Support Vector Machine using RBF kernel")
        ls_svm_accuracy_holder = {}
        ls_svm_auroc_holder = {}
        for slack_var in np.linspace(0.1, 10, num = self.n_vars):
            svm_accuracy_holder = []
            svm_auroc_holder = []
            for i in range(self.args.n_cv_folds):
                train_fold_list = list(range(self.args.n_cv_folds))
                train_fold_list = [x for x in train_fold_list if x != i]

                temp = 0
                for fold in train_fold_list:
                    if temp == 0:
                        train_inputs = self.pks_dict[str(fold) + "_inputs"]
                        train_labels = self.pks_dict[str(fold) + "_labels"]
                        temp = 1
                    else:
                        train_inputs = pd.concat([train_inputs, self.pks_dict[str(fold) + "_inputs"]], axis = 0)
                        train_labels = pd.concat([train_labels, self.pks_dict[str(fold) + "_labels"]], axis = 0)

                test_inputs = self.pks_dict[str(i) + "_inputs"]
                test_labels = self.pks_dict[str(i) + "_labels"]

                svm_acc, svm_auroc = train_eval_svm_classifier(self.args,
                                                               self.config,
                                                               train_inputs,
                                                               train_labels["status"],
                                                               test_inputs,
                                                               test_labels["status"],
                                                               slack_var)
                svm_accuracy_holder.append(svm_acc)
                svm_auroc_holder.append(svm_auroc)

            ls_svm_accuracy_holder[str(slack_var.round(3))] = np.mean(svm_accuracy_holder)
            ls_svm_auroc_holder[str(slack_var.round(3))] = np.mean(svm_auroc_holder)

        plot_1d_ls_acc_plot(self.config, ls_svm_accuracy_holder, "Slack Variable", "Accuracy of Support Vector Machine")
        plot_1d_ls_auroc_plot(self.config, ls_svm_auroc_holder, "Slack Variable", "Auroc of Support Vector Machine")
        ls_svm_acc_maxkey = max(ls_svm_accuracy_holder, key=ls_svm_accuracy_holder.get)
        ls_svm_maxacc = ls_svm_accuracy_holder[ls_svm_acc_maxkey]
        ls_svm_auroc_maxkey = max(ls_svm_auroc_holder, key=ls_svm_auroc_holder.get)
        ls_svm_maxauroc = ls_svm_auroc_holder[ls_svm_auroc_maxkey]

        print("CV Accuracy by slack variable(Support Vector Machine): ", ls_svm_accuracy_holder)
        print("Max Accuracy:\t key %s\t Value\t %f" % (ls_svm_acc_maxkey, ls_svm_maxacc))
        print("")
        print("CV Auroc by slack variable(Support Vector Machine): ", ls_svm_auroc_holder)
        print("Max Auroc:\t key %s\t Value\t %f" % (ls_svm_auroc_maxkey, ls_svm_maxauroc))

    def PCA_KSVM(self):
        ### Support Vector Machine using PCA and Kernel trick.
        print("_"*80)
        print("Apply PCA and Kernel Support Vector Machine using RBF kernel")
        ls_pca_svm_accuracy_holder = {}
        ls_pca_svm_auroc_holder = {}
        for reduced_dim in range(1,(self.n_vars + 1)):
            for slack_var in np.linspace(0.1, 10, num = self.n_vars):
                pca_svm_accuracy_holder = []
                pca_svm_auroc_holder = []
                for i in range(self.args.n_cv_folds):
                    train_fold_list = list(range(self.args.n_cv_folds))
                    train_fold_list = [x for x in train_fold_list if x != i]

                    temp = 0
                    for fold in train_fold_list:
                        if temp == 0:
                            train_inputs = self.pks_dict[str(fold) + "_inputs"]
                            train_labels = self.pks_dict[str(fold) + "_labels"]
                            temp = 1
                        else:
                            train_inputs = pd.concat([train_inputs, self.pks_dict[str(fold) + "_inputs"]], axis=0)
                            train_labels = pd.concat([train_labels, self.pks_dict[str(fold) + "_labels"]], axis=0)

                    test_inputs = self.pks_dict[str(i) + "_inputs"]
                    test_labels = self.pks_dict[str(i) + "_labels"]

                    good_of_fit, pca_svm_acc, pca_svm_auroc = train_eval_pca_svm_classifier(self.args,
                                                                                            self.config,
                                                                                            train_inputs,
                                                                                            train_labels["status"],
                                                                                            test_inputs,
                                                                                            test_labels["status"],
                                                                                            reduced_dim,
                                                                                            slack_var)
                    pca_svm_accuracy_holder.append(pca_svm_acc)
                    pca_svm_auroc_holder.append(pca_svm_auroc)

                ls_pca_svm_accuracy_holder[str(reduced_dim) + "/" +  str(slack_var.round(3))] = np.mean(pca_svm_accuracy_holder)
                ls_pca_svm_auroc_holder[str(reduced_dim) + "/" +  str(slack_var.round(3))] = np.mean(pca_svm_auroc_holder)

        plot_2d_ls_acc_plot(self.config, ls_pca_svm_accuracy_holder, [self.n_vars, self.n_vars], "Reduced Dimension", "Slack Variable", "Accuracy of Support Vector Machine using PCA")
        plot_2d_ls_auroc_plot(self.config, ls_pca_svm_auroc_holder, [self.n_vars, self.n_vars], "Reduced Dimension", "Slack Variable", "Auroc of Support Vector Machine using PCA")
        ls_pca_svm_acc_maxkey = max(ls_pca_svm_accuracy_holder, key=ls_pca_svm_accuracy_holder.get)
        ls_pca_svm_maxacc = ls_pca_svm_accuracy_holder[ls_pca_svm_acc_maxkey]
        ls_pca_svm_auroc_maxkey = max(ls_pca_svm_auroc_holder, key=ls_pca_svm_auroc_holder.get)
        ls_pca_svm_maxauroc = ls_pca_svm_auroc_holder[ls_pca_svm_auroc_maxkey]
        print("CV Accuracy by input dimension and slack variable(PCA Support Vector Machine): ", ls_pca_svm_accuracy_holder)
        print("Max Accuracy:\t key %s\t Value\t %f" % (ls_pca_svm_acc_maxkey, ls_pca_svm_maxacc))
        print("")
        print("CV Auroc by input dimension and slack variable(PCA Support Vector Machine): ", ls_pca_svm_auroc_holder)
        print("Max Auroc:\t key %s\t Value\t %f" % (ls_pca_svm_auroc_maxkey, ls_pca_svm_maxauroc))

    def Logistic_regression_using_nn(self):
        ### Logistic Regression Classification using Neural Networks.
        print("_"*80)
        print("Extract low dimensional representation and apply logistic regression via MLP")
        ls_nn_accuracy_holder = {}
        ls_nn_auroc_holder = {}
        for lamda in tqdm(np.linspace(0, 1, 11)):
            for lr in np.linspace(0.001, 0.0001, self.n_vars):
                nn_accuracy_holder = []
                nn_auroc_holder = []
                for i in range(self.args.n_cv_folds):
                    train_fold_list = list(range(self.args.n_cv_folds))
                    train_fold_list = [x for x in train_fold_list if x != i]

                    temp = 0
                    for fold in train_fold_list:
                        if temp == 0:
                            train_inputs = self.pks_dict[str(fold) + "_inputs"]
                            train_labels = self.pks_dict[str(fold) + "_labels"]
                            temp = 1
                        else:
                            train_inputs = pd.concat([train_inputs, self.pks_dict[str(fold) + "_inputs"]], axis=0)
                            train_labels = pd.concat([train_labels, self.pks_dict[str(fold) + "_labels"]], axis=0)

                    test_inputs = self.pks_dict[str(i) + "_inputs"]
                    test_labels = self.pks_dict[str(i) + "_labels"]

                    epoch, loss, nn_acc, nn_auroc, embeddings = train_eval_nn_classifier(self.args,
                                                                                         self.config,
                                                                                         train_inputs,
                                                                                         train_labels["status"],
                                                                                         test_inputs,
                                                                                         test_labels["status"],
                                                                                         lr,
                                                                                         lamda)
                    nn_accuracy_holder.append(nn_acc)
                    nn_auroc_holder.append(nn_auroc)

                ls_nn_accuracy_holder[str(lr.round(5)) + "/" +  str(lamda.round(3))] = np.mean(nn_accuracy_holder)
                ls_nn_auroc_holder[str(lr.round(5)) + "/" +  str(lamda.round(3))] = np.mean(nn_auroc_holder)
                plot_epoch_loss_curve(self.config, epoch, loss, lr, lamda)

        print("")
        plot_2d_ls_acc_plot(self.config, ls_nn_accuracy_holder, [11,self.n_vars], "Learning Rate", "Lamda", "Accuracy of Logistic Regression Classification via Neural Network")
        plot_2d_ls_auroc_plot(self.config, ls_nn_auroc_holder, [11,self.n_vars], "Learning Rate", "Lamda", "Auroc of Logistic Regression Classification via Neural Network")
        ls_nn_acc_maxkey = max(ls_nn_accuracy_holder, key=ls_nn_accuracy_holder.get)
        ls_nn_maxacc = ls_nn_accuracy_holder[ls_nn_acc_maxkey]
        ls_nn_auroc_maxkey = max(ls_nn_auroc_holder, key=ls_nn_auroc_holder.get)
        ls_nn_maxauroc = ls_nn_auroc_holder[ls_nn_auroc_maxkey]
        print("CV Accuracy by learning rate(Logistic Regression using Neural Network): ", ls_nn_accuracy_holder)
        print("Max Accuracy:\t key %s\t Value\t %f" % (ls_nn_acc_maxkey, ls_nn_maxacc))
        print("")
        print("CV Auroc by learning rate(Logistic Regression using Neural Network): ", ls_nn_auroc_holder)
        print("Max Auroc:\t key %s\t Value\t %f" % (ls_nn_auroc_maxkey, ls_nn_maxauroc))

        return ls_nn_auroc_maxkey

    def extract_2d_representation(self, lr, lamda):
        for i in range(1):
            train_fold_list = list(range(self.args.n_cv_folds))
            train_fold_list = [x for x in train_fold_list if x != i]

            temp = 0
            for fold in train_fold_list:
                if temp == 0:
                    train_inputs = self.pks_dict[str(fold) + "_inputs"]
                    train_labels = self.pks_dict[str(fold) + "_labels"]
                    temp = 1
                else:
                    train_inputs = pd.concat([train_inputs, self.pks_dict[str(fold) + "_inputs"]], axis=0)
                    train_labels = pd.concat([train_labels, self.pks_dict[str(fold) + "_labels"]], axis=0)

            test_inputs = self.pks_dict[str(i) + "_inputs"]
            test_labels = self.pks_dict[str(i) + "_labels"]

            epoch, loss, nn_acc, nn_auroc, embeddings = train_eval_nn_classifier(self.args,
                                                                                 self.config,
                                                                                 train_inputs,
                                                                                 train_labels["status"],
                                                                                 test_inputs,
                                                                                 test_labels["status"],
                                                                                 lr,
                                                                                 lamda)

            plot_2d_nn_scatter_plot(self.config, embeddings.numpy(), train_labels["status"])
            plot_2d_pcs_scatter_plot(self.config, train_inputs, train_labels["status"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type = str, default = "configs/pd_config.yml", help = "path to config file")
    parser.add_argument('--n_cv_folds', type = int, default = 5, help = "number of cross validation folds")
    parser.add_argument('--regularizer', type = str, default = 'l2', help = "['none', 'l1', 'l2']")
    parser.add_argument('--n_iter', type = int, default = 500, help = "number of iteration for gradient descent algorithm")
    args = parser.parse_args()

    main = main(args)
    main.load_data4cv()
    main.perform_EDA()
    main.Naive_logistic_regression()
    main.PCA_logistic_regression()
    main.LDA()
    main.PCA_LDA()
    main.KSVM()
    main.PCA_KSVM()
    ls_nn_auroc_maxkey = main.Logistic_regression_using_nn()
    learning_rate, lamda = ls_nn_auroc_maxkey.split('/')
    main.extract_2d_representation(lr = float(learning_rate), lamda = float(lamda))


