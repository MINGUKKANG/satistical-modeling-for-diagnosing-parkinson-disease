from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset

from utils.matplotlib import *

np.random.seed(777)

def train_eval_logistic_regression(args, config, train_xs, train_ys, test_xs, test_ys, penalty = 'l1'):
    """
    :param args: parser.argument
    :param penalty: ["none", "l1", "l2"] standard regularizer for linear SVM
    :return: models' accuracy

    learning rate is determined as following criteria:
    eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
    """
    logistic_reg = SGDClassifier(max_iter = args.n_iter, loss = 'log', penalty = penalty, learning_rate = "optimal")
    logistic_reg.fit(train_xs, train_ys)
    pos_logistic_proba = logistic_reg.predict_proba(test_xs)[:,0]
    logistic_pred = logistic_reg.predict(test_xs)

    fpr, tpr, thresholds = metrics.roc_curve(test_ys, pos_logistic_proba, pos_label = 0)
    auroc = metrics.auc(fpr, tpr)
    accuracy = np.mean(np.equal(logistic_pred, test_ys))*100

    return accuracy, auroc

def train_eval_pca_logistic_regression(args, config, train_xs, train_ys, test_xs, test_ys, reduced_dim, penalty = 'l1'):
    mean = train_xs.mean()
    std = train_xs.std()
    train_normed = (train_xs - mean)/std
    test_normed = (test_xs - mean)/std
    n_inst = np.shape(train_normed)[0]
    train_normed_T = train_normed.T
    R = train_normed_T.dot(train_normed)/(n_inst-1)
    U,S,V = np.linalg.svd(R)
    Eigen_values = np.square(S + 1e-5)

    goodness_of_fit = Eigen_values/ np.sum(Eigen_values).round(3)

    pc_score_train = np.matmul(np.asarray(train_normed), U[:,:reduced_dim])
    pc_score_test = np.matmul(np.asarray(test_normed), U[:, :reduced_dim])

    logistic_reg = SGDClassifier(max_iter = args.n_iter, loss = 'log', penalty = penalty, learning_rate = "optimal")
    logistic_reg.fit(pc_score_train, train_ys)
    pos_logistic_proba = logistic_reg.predict_proba(pc_score_test)[:, 0]
    logistic_pred = logistic_reg.predict(pc_score_test)

    fpr, tpr, thresholds = metrics.roc_curve(test_ys, pos_logistic_proba, pos_label = 0)
    auroc = metrics.auc(fpr, tpr)
    accuracy = np.mean(np.equal(logistic_pred, test_ys))*100

    return goodness_of_fit, accuracy, auroc

def train_eval_LDA(args, config, train_xs, train_ys, test_xs, test_ys):
    lda = LDA()
    lda.fit(train_xs, train_ys)
    lda_pred = lda.predict(test_xs)
    pos_lda_proba = lda.predict_proba(test_xs)[:,0]

    fpr, tpr, thresholds = metrics.roc_curve(test_ys, pos_lda_proba, pos_label = 0)
    auroc = metrics.auc(fpr, tpr)
    accuracy = np.mean(np.equal(lda_pred, test_ys))*100

    return accuracy, auroc

def train_eval_pca_LDA(args, config, train_xs, train_ys, test_xs, test_ys, reduced_dim):
    mean = train_xs.mean()
    std = train_xs.std()
    train_normed = (train_xs - mean) / std
    test_normed = (test_xs - mean) / std
    n_inst = np.shape(train_normed)[0]
    train_normed_T = train_normed.T
    R = train_normed_T.dot(train_normed) / (n_inst - 1)
    U, S, V = np.linalg.svd(R)
    Eigen_values = np.square(S + 1e-5)

    goodness_of_fit = Eigen_values / np.sum(Eigen_values).round(3)

    pc_score_train = np.matmul(np.asarray(train_normed), U[:, :reduced_dim])
    pc_score_test = np.matmul(np.asarray(test_normed), U[:, :reduced_dim])

    lda = LDA()
    lda.fit(pc_score_train, train_ys)
    lda_pred = lda.predict(pc_score_test)
    pos_lda_proba = lda.predict_proba(pc_score_test)[:,0]

    fpr, tpr, thresholds = metrics.roc_curve(test_ys, pos_lda_proba, pos_label=0)
    auroc = metrics.auc(fpr, tpr)
    accuracy = np.mean(np.equal(lda_pred, test_ys))*100

    return goodness_of_fit, accuracy, auroc

def train_eval_svm_classifier(args, config, train_xs, train_ys, test_xs, test_ys, slack_var):
    svm = SVC(kernel='rbf', C = slack_var, gamma = 'auto', probability=True).fit(train_xs, train_ys)
    svm_pred = svm.predict(test_xs)
    pos_svm_prob = svm.predict_proba(test_xs)[:,0]

    fpr, tpr, thresholds = metrics.roc_curve(test_ys, pos_svm_prob, pos_label = 0)
    auroc = metrics.auc(fpr, tpr)
    accuracy = np.mean(np.equal(svm_pred, test_ys))*100

    return accuracy, auroc

def train_eval_pca_svm_classifier(args, config, train_xs, train_ys, test_xs, test_ys, reduced_dim, slack_var):
    mean = train_xs.mean()
    std = train_xs.std()
    train_normed = (train_xs - mean) / std
    test_normed = (test_xs - mean) / std
    n_inst = np.shape(train_normed)[0]
    train_normed_T = train_normed.T
    R = train_normed_T.dot(train_normed) / (n_inst - 1)
    U, S, V = np.linalg.svd(R)
    Eigen_values = np.square(S + 1e-5)

    goodness_of_fit = Eigen_values / np.sum(Eigen_values).round(3)

    pc_score_train = np.matmul(np.asarray(train_normed), U[:, :reduced_dim])
    pc_score_test = np.matmul(np.asarray(test_normed), U[:, :reduced_dim])

    svm = SVC(kernel='rbf', C = slack_var, gamma = 'auto', probability=True).fit(pc_score_train, train_ys)
    svm_pred = svm.predict(pc_score_test)
    pos_svm_prob = svm.predict_proba(pc_score_test)[:, 0]

    fpr, tpr, thresholds = metrics.roc_curve(test_ys, pos_svm_prob, pos_label = 0)
    auroc = metrics.auc(fpr, tpr)
    accuracy = np.mean(np.equal(svm_pred, test_ys))*100

    return goodness_of_fit, accuracy, auroc

def train_eval_nn_classifier(args, config, train_xs, train_ys, test_xs, test_ys, lr, lamda):
    train_X = torch.from_numpy(np.array(train_xs)).float()
    train_Y = torch.from_numpy(np.array(train_ys)).long()
    test_X = torch.from_numpy(np.array(test_xs)).float()
    test_Y = torch.from_numpy(np.array(test_ys)).long()
    n_vars = np.shape(train_X)[1]

    model = MLP(n_vars)
    criterion  = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    loss_holder = []
    epoch_holder = []
    for epoch in range(args.n_iter):
        train_X, train_Y = Variable(train_X), Variable(train_Y)
        optimizer.zero_grad()
        canonical, output = model(train_X)

        if args.regularizer == "l1":
            l1 = None
            for w in model.parameters():
                if l1 is None:
                    l1 = lamda * w.abs().sum()
                else:
                    l1 = l1 + lamda * w.abs().sum()
            loss = criterion(output, train_Y) + l1

        elif args.regularizer == "l2":
            l2 = None
            for w in model.parameters():
                if l2 is None:
                    l2 = lamda * w.norm(2)
                else:
                    l2 = l2 + lamda * w.norm(2)
            loss = criterion(output, train_Y) + l2

        else:
            loss = criterion(output, train_Y)

        loss.backward()
        optimizer.step()
        epoch_holder.append(epoch)
        loss_holder.append(loss.item())

    test_X, test_Y = Variable(test_X), Variable(test_Y)
    test_canonical, test_logits = model(test_X)
    result = torch.max(test_logits.data, 1)[1]
    pos_nn_prob = 1/(1 + np.exp(-test_logits.data[:,0]))

    fpr, tpr, thresholds = metrics.roc_curve(test_ys, pos_nn_prob, pos_label = 0)
    auroc = metrics.auc(fpr, tpr)
    accuracy = sum(test_Y.data.numpy() == result.numpy())/ len(test_Y.data.numpy())
    accuracy = accuracy*100

    embeddings, _ = model(train_X)
    embeddings = embeddings.data

    return epoch_holder, loss_holder, accuracy, auroc, embeddings

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.main = nn.Sequential(
            nn.Linear(self.input_dim, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(True),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(True),
            nn.Linear(24, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(True),
            nn.Linear(12, 2)
        )
        self.log_reg_aprx = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.ReLU(True),
            nn.Linear(2,2))

    def forward(self, x):
        canonical_parms = self.main(x)
        pred_logits = self.log_reg_aprx(canonical_parms)

        return canonical_parms, pred_logits