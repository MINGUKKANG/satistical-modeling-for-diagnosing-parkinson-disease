import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def plot_scree_plot(config, good_of_fit, num_vars = 6):
    fig = plt.figure(figsize=(8, 5))
    sing_vals = np.arange(num_vars) + 1
    plt.plot(sing_vals, good_of_fit[:num_vars], 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')

    leg = plt.legend(['Eigenvalues from SVD'], loc='best',
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='x-large'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], config.save["scree_plot_name"])

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path, dpi = config.save["dpi"])
    print("Scree plot save directory: %s" % (path))
    plt.close(fig)

def plot_1d_ls_acc_plot(config, acc_dict, xlabel, title):
    fig = plt.figure(figsize=(8, 5))
    x = []
    y = []
    for key in acc_dict:
        x.append(float(key))
        y.append(acc_dict[key])

    x, y = zip(*sorted(zip(x, y)))

    plt.plot(x, y, 'ro-', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], title + ".png").replace(" ", "_")

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path, dpi = config.save["dpi"])
    print("%s save directory: %s" % (title ,path))
    plt.close(fig)

def plot_1d_ls_auroc_plot(config, auroc_dict, xlabel, title):
    fig = plt.figure(figsize=(8, 5))
    x = []
    y = []
    for key in auroc_dict:
        x.append(float(key))
        y.append(auroc_dict[key])

    x, y = zip(*sorted(zip(x, y)))

    plt.plot(x, y, 'ro-', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Auroc')

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], title + ".png").replace(" ", "_")

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path, dpi = config.save["dpi"])
    print("%s save directory: %s" % (title ,path))
    plt.close(fig)

def plot_2d_ls_acc_plot(config, acc_dict, xy_shape , xlabel, ylabel, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = []
    y = []
    z = []
    for key in acc_dict:
        x_element, y_element = key.split('/')
        x.append(float(x_element))
        y.append(float(y_element))
        z.append(acc_dict[key])

    X = np.reshape(np.asarray(x), xy_shape)
    Y = np.reshape(np.asarray(y), xy_shape)
    Z = np.reshape(np.asarray(z), xy_shape)

    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.10, top=0.90)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Accuracy')

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], title + ".png").replace(" ", "_")

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path, dpi = config.save["dpi"])
    print("%s save directory: %s" % (title ,path))
    plt.close(fig)

def plot_2d_ls_auroc_plot(config, auroc_dict, xy_shape, xlabel, ylabel, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = []
    y = []
    z = []
    for key in auroc_dict:
        x_element, y_element = key.split('/')
        x.append(float(x_element))
        y.append(float(y_element))
        z.append(auroc_dict[key])

    X = np.reshape(np.asarray(x), xy_shape)
    Y = np.reshape(np.asarray(y), xy_shape)
    Z = np.reshape(np.asarray(z), xy_shape)


    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.90)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Auroc')

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], title + ".png").replace(" ", "_")

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path, dpi = config.save["dpi"])
    print("%s save directory: %s" % (title ,path))
    plt.close(fig)

def plot_epoch_loss_curve(config, epoch, loss, lr, lamda):
    title = "Loss Curve of training set"
    xlabel = "Epoch"
    ylabel = "Loss"
    fig = plt.figure(figsize=(8, 5))

    plt.plot(epoch, loss, 'r-', linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    leg = plt.legend(["lr/lamda: " + str(lr)[:7] + "/" + str(lamda)], loc='best',
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='x-large'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], title + "_" + str(lr)[:7]  + "_" + str(lamda) + ".png").replace(" ", "_")

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path, dpi = config.save["dpi"])
    # print("%s save directory: %s" % (title ,path))
    plt.close(fig)

def plot_2d_pcs_scatter_plot(config, train_xs, status_idx):
    df_normed = (train_xs - train_xs.mean())/train_xs.std()
    n_inst = np.shape(df_normed)[0]
    df_normed_T = df_normed.T
    R = df_normed_T.dot(df_normed)/(n_inst-1)
    U,S,V = np.linalg.svd(R)
    pc_score = np.matmul(np.asarray(df_normed), U)

    xs = pc_score[:, 0]
    ys = pc_score[:, 1]

    colors = ["#1f77b4" if x == 0 else "#2ca02c" for x in status_idx]
    pop_a = mpatches.Patch(color = "#1f77b4", label = 'Normal')
    pop_b = mpatches.Patch(color = "#2ca02c", label = 'Abnormal')
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.scatter(xs, ys, c = colors)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlabel("PC{} score".format(1))
    plt.ylabel("PC{} score".format(2))
    plt.grid()

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], config.save["2d_pc_scatter_plot"])

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path , dpi = config.save["dpi"])
    print("%s save directory: %s" % ("2D pc scatter plot" ,path))
    print("")
    plt.close()

def plot_2d_nn_scatter_plot(config, embeddings, status_idx):
    colors = ["#1f77b4" if x == 0 else "#2ca02c" for x in status_idx]
    pop_a = mpatches.Patch(color = "#1f77b4", label = 'Normal')
    pop_b = mpatches.Patch(color = "#2ca02c", label = 'Abnormal')
    plt.legend(handles=[pop_a, pop_b] ,loc='upper right')
    plt.scatter(embeddings[:,0], embeddings[:,1], c = colors)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlabel("NN{} score".format(1))
    plt.ylabel("NN{} score".format(2))
    plt.grid()

    if not os.path.isdir(config.save["plt_save_folder"]):
        os.makedirs(config.save["plt_save_folder"])

    path = os.path.join(config.save["plt_save_folder"], config.save["2d_nn_scatter_plot"])

    if os.path.isfile(path):
        os.remove(path)

    plt.savefig(path, dpi = config.save["dpi"])
    print("%s save directory: %s" % ("2D nn scatter plot" ,path))
    print("")
    plt.close()