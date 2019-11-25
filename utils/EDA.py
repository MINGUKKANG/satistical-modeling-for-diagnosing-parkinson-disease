import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


def plot_corr_heatmap(config, df4corr):
    f, ax = plt.subplots(figsize = (12,9))
    sns.heatmap(df4corr.corr(), vmax=.8, square = True, cmap='coolwarm')

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], config.save["corr_heatmap_plot"])

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path, dpi = config.save["dpi"])
    print("%s save directory: %s" % ("Correlation heatmap " , path))
    plt.close()

def plot_corr_heatmap_after_pca(config, df4corr):
    df4corr_normed = (df4corr - df4corr.mean())/df4corr.std()
    n_inst = np.shape(df4corr_normed)[0]
    df4corr_normed_T = df4corr_normed.T
    R = df4corr_normed_T.dot(df4corr_normed)/(n_inst-1)
    U,S,V = np.linalg.svd(R)
    pc_score = np.matmul(np.asarray(df4corr_normed), U)

    f, ax = plt.subplots(figsize = (12,9))
    sns.heatmap(np.corrcoef(pc_score.T), vmax=.8, square = True, cmap='coolwarm')

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], config.save["corr_heatmap_after_pca_plot"])

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path, dpi = config.save["dpi"])
    print("%s save directory: %s" % ("Correlation heatmap after pca" , path))
    plt.close()

def pca_biplot(config, df):
    # Reference
    # https://codeday.me/ko/qa/20190406/255709.html.
    # Thanks for author
    df_normed = (df - df.mean())/df.std()
    n_inst = np.shape(df_normed)[0]
    df_normed_T = df_normed.T
    R = df_normed_T.dot(df_normed)/(n_inst-1)
    U,S,V = np.linalg.svd(R)
    pc_score = np.matmul(np.asarray(df_normed), U)

    xs = pc_score[:, 0]
    ys = pc_score[:, 1]
    n = U.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    colors = ["#1f77b4" if x == 0 else "#2ca02c" for x in df['status']]
    pop_a = mpatches.Patch(color = "#1f77b4", label='Normal')
    pop_b = mpatches.Patch(color = "#2ca02c", label='Abnormal')
    plt.legend(handles=[pop_a, pop_b])
    plt.scatter(xs * scalex, ys * scaley, c = colors)
    for i in range(n):
        plt.arrow(0, 0, U[i, 0], U[i, 1], color='r', alpha=0.5)
        plt.text(U[i, 0] * 1.15, U[i, 1] * 1.15, df.columns[i], color='black', ha='center', va='center')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlabel("PC{} score".format(1))
    plt.ylabel("PC{} score".format(2))
    plt.grid()

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], config.save["pca_biplot"])

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path, dpi = config.save["dpi"])
    print("%s save directory: %s" % ("PCA_biplot" ,path))
    print("")
    plt.close()

def chi_square_normality_test(df_matrix, alpha = 0.001, pca = False):
    ### statistic = s^2 + k^2,
    ### where s is the z-score returned by skewtest and k is the z-score returned by kurtosistest.
    not_normal_dist = []
    normal_dist = []
    p_value_dict = {}
    if pca is False:
        vars = df_matrix.columns
        for var in vars:
            statistic, p_value = stats.normaltest(df_matrix[var])
            if p_value < alpha:
                not_normal_dist.append(var)
            else:
                normal_dist.append(var)
            p_value_dict[var] = p_value

        print("Chi_square normality Test!")
        print("Follow normal distribution: ", normal_dist)
        print("Follow unknown distribution: ", not_normal_dist)
        print("P_value(chi_squre normality test):", p_value_dict)
        print("")
    else:
        df_normed = (df_matrix - df_matrix.mean()) / df_matrix.std()
        n_inst = np.shape(df_normed)[0]
        n_vars = np.shape(df_normed)[1]
        df_normed_T = df_normed.T
        R = df_normed_T.dot(df_normed) / (n_inst - 1)
        U, S, V = np.linalg.svd(R)
        pc_score = np.matmul(np.asarray(df_normed), U)

        for i in range(n_vars):
            statistic, p_value = stats.normaltest(pc_score[:,i])
            if p_value < alpha:
                not_normal_dist.append(str(i))
            else:
                normal_dist.append(str(i))
            p_value_dict[str(i)] = p_value

        print("Chi_square normality Test!")
        print("Follow normal distribution: ", normal_dist)
        print("Follow unknown distribution: ", not_normal_dist)
        print("P_value(chi_squre normality test):", p_value_dict)
        print("")





