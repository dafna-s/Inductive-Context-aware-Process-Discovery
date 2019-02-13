from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nltk
from nltk.cluster.kmeans import KMeansClusterer
import k_means
import pandas as pd
import os
import time
from IPython.utils.io import Tee
from contextlib import closing

import fig_tree
import config
import util
import util_IM


def split_by_data(df, labels):
    """
    This func get list of label for every line in the df and split the lines by the list to sublogs
    :param labels: list with label for every line
    :return: list of sublogs.
    """
    set_l = list(set(labels))
    sub_df = []
    for j in set_l:
        # if labels[j] not in set_l:
        #     continue
        index_j = [i for i, e in enumerate(labels) if e == j]
        sub_df.append(df.iloc[index_j, :].copy())
        # set_l.remove(labels[j])

    return sub_df


def split_by_IM(df, activities):
    num_splits = len(activities)
    sub_df = []
    for j in range(num_splits):
        activities_j = activities[j].split(',')
        sub_df.append(df[df['activity'].isin(activities_j)])
    return sub_df


def find_labels_IM(df, activities):
    dict_l = {}
    for i, act in enumerate(activities):
        for a in act.split(','):
            dict_l[a] = i
    labels_IM = list(df['activity'].map(dict_l))
    return labels_IM


def find_inside_operator(tree):
    """
    given tree and return what inside the () and the index of the ')'
    :param tree: process tree.
    :return: string inside the () and the index of the ')' .
    """
    orig_tree = tree
    counter = 0
    end = 0
    while tree.find('(') > 0:
        if tree.find('(') < tree.find(')'):
            counter += 1
            end += tree.find('(') + 1
            tree = tree[tree.find('(') + 1:]
            end += tree.find(')') + 1 if tree.find('(') == -1 else 0
        else:
            counter -= 1
            end += tree.find(')') + 1
            tree = tree[tree.find(')') + 1:]
        if counter == 0 or tree.find('(') == -1:
            break
    return orig_tree[orig_tree.find('(') + 1:end - 1], end


def find_decision_point(df):
    # func
    if len(df['activity'].unique()) <= 1:
        return df.activity.unique()[0]

    util_IM.send_to_IM(df)
    cut = util_IM.find_cut()
    print('cut:', cut)
    IM_operator = cut[:1].upper() + cut[1:3].lower()
    if IM_operator == 'Con':
        IM_operator = 'And'
    elif IM_operator == 'Loo':
        IM_operator = 'Loop'
    activities = []
    while cut.find('{') > 0:
        activities.append(cut[(cut.find('{') + 1): cut.find('}')].replace(" ", ""))
        cut = cut[cut.find('}') + 1:]
    print('activities: ', activities, len(activities))

    # calculate silhouette for IM
    X = df.drop(['activity', 'case'], axis=1).values
    print(X, type(X))

    #   clustering - k-means
    dist_cat = lambda u, v: sum([1 if ui != vi else 0 for ui, vi in zip(u, v)])
    dist_num = nltk.cluster.util.euclidean_distance

    num_features = len(X[0])
    num_categorical_type = len(config.categorical)
    num_numeric_type = num_features - num_categorical_type

    dist = lambda u, v: (num_categorical_type / (num_numeric_type + num_categorical_type)) * \
                        dist_num(u[0:num_numeric_type], v[0:num_numeric_type]) \
                        + (num_numeric_type / (num_numeric_type + num_categorical_type)) * \
                        dist_cat(u[num_numeric_type:], v[num_numeric_type:])

    kclusterer = k_means.myKMeansClusterer(config.num_splits, distance=dist, repeats=25, avoid_empty_clusters=True)
    labels = kclusterer.cluster(X, assign_clusters=True)
    centers = kclusterer.means()

    print('k-means')
    print('labels: ', labels)
    df_only_features = df.drop(['case', 'activity'], axis=1).copy()
    columns_name = df_only_features.columns.values
    decision_point, decision_operators = k_means.decision(X, labels, centers, columns_name)
    print('centers: ', centers)
    print('decision_point: ', decision_point)
    print('decision_operators: ', decision_operators)

    #  Find the most influence features
    goodness = 0
    labels_for_feature = []
    data_split_by = ''

    if config.kmeans_feature == '1s':
        goodness, labels_for_feature, data_split_by = k_means.one_silhouette_influence_feature(df)
    elif 1 < len(set(labels)) <= len(X) - 1:  # number of labels is 2 <= n_labels <= n_samples - 1
        silhouette_avg = silhouette_score(X, labels)
        print('silhouette: ', silhouette_avg)
        if silhouette_avg > config.silhouette_threshold:
            if config.kmeans_feature == '1':
                goodness, labels_for_feature, data_split_by = k_means.one_influence_feature(labels, decision_point, decision_operators,
                                                                                    df_only_features)
            elif config.kmeans_feature == 'n':
                goodness, labels_for_feature, data_split_by = k_means.n_influence_feature(labels, decision_point,
                                                                                          decision_operators, df_only_features)
            elif config.kmeans_feature == 'k':
                goodness, labels_for_feature, data_split_by = k_means.k_influence_feature(df_only_features, decision_point, decision_operators, labels)

    # Choose split (data or IM)
    if goodness >= config.threshold_for_data_split:
        sub_df = split_by_data(df, labels_for_feature)
        config.silhouette.append([silhouette_score(X, labels_for_feature), len(X)])
        operator = 'Xor'
        split_by = data_split_by
        num_splits = config.num_splits
    else:
        sub_df = split_by_IM(df, activities)
        labels_IM = find_labels_IM(df, activities)
        if 1 < len(set(labels_IM)) <= len(X) - 1:
            config.silhouette.append([silhouette_score(X, labels_IM), len(X)])
        else:
            config.silhouette.append([1.0, len(X)])
        operator = IM_operator
        split_by = 'IM'
        num_splits = len(activities)
        IM_sub_tree = util_IM.find_tree()
    sub_tree = ''
    for i in range(num_splits):

        print(sub_df[i])
        print(sub_tree)

        sub_tree = sub_tree + str(find_decision_point(sub_df[i])) + ','

    config.explain_operator.append({operator + '(' + sub_tree[:-1] + ')': split_by})
    return operator + '(' + sub_tree[:-1] + ')'


if __name__ == '__main__':
    df = util.get_data()
    util_IM.send_to_IM(df)
    IM_tree = util_IM.find_tree()
    print('IM tree: ', IM_tree)

    df_for_cluster = util.pre_process(df)

    tree = find_decision_point(df_for_cluster)
    IM_tree = IM_tree.replace(" ", "")

    fig_tree.create_tree(tree, config.silhouette)
    sum_sil = 0.0
    max_row = 0
    for node in config.process_tree:
        if node.sil == None:
            continue
        parent_log_rows = node.parent.log_rows if node.parent != -1 else node.log_rows
        sum_sil += (node.sil * (node.log_rows/parent_log_rows))
        max_row = node.row if node.row > max_row else max_row
    total_sil = sum_sil / (max_row+1)

    with closing(Tee('log_file.txt', "a", channel="stdout")) as outputstream:
        print("=============================================")
        print(time.strftime("%d/%m/%Y  %H:%M:%S"))
        print("{} \nsilhouette threshold: {}".format(config.data_file, config.silhouette_threshold ))
        print('IM tree: ', IM_tree)
        print('tree: ', tree)
        print(config.explain_operator)
        print('sil: ', config.silhouette)
        print('total_sil: ', total_sil)

    fig_tree.create_tree_fig()



    # activity_set = set(df['activity'].unique())
