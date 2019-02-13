import nltk
import sys
import copy
from nltk.cluster.kmeans import KMeansClusterer
from collections import Counter
import numpy as np
from sklearn.metrics import silhouette_score
import operator

import util
import config


class myKMeansClusterer(KMeansClusterer):

    def _centroid(self, cluster, mean):
        if self._avoid_empty_clusters:
            if len(cluster) == 0:
                return mean
            centroid = copy.copy(mean)
            index = list(range(len(cluster[0])))
            i_c = []
            for i in index:
                if type(cluster[0][i]) == type(1):
                    i_c += [i, ]
            for vector in cluster:
                centroid += vector
            centroid = centroid / (1 + len(cluster))
            for i in i_c:
                c = Counter(np.array(cluster)[:, i])
                centroid[i], _ = c.most_common()[0]
                # centroid[i] = np.unique(np.array(cluster)[:, i])
            return centroid
        else:
            if not len(cluster):
                sys.stderr.write('Error: no centroid defined for empty cluster.\n')
                sys.stderr.write(
                    'Try setting argument \'avoid_empty_clusters\' to True\n'
                )
                assert False
            centroid = copy.copy(cluster[0])
            for vector in cluster[1:]:
                centroid += vector
            return centroid / len(cluster)



    def _cluster_vectorspace(self, vectors, trace=False):
        counter = 0
        if self._num_means < len(vectors):
            # perform k-means clustering
            converged = False
            while not converged:
                # assign the tokens to clusters based on minimum distance to
                # the cluster means
                if counter >= 50:
                    break
                clusters = [[] for m in range(self._num_means)]
                for vector in vectors:
                    index = self.classify_vectorspace(vector)
                    clusters[index].append(vector)

                if trace: print('iteration')
                #for i in range(self._num_means):
                    #print '  mean', i, 'allocated', len(clusters[i]), 'vectors'

                # recalculate cluster means by computing the centroid of each cluster
                new_means = list(map(self._centroid, clusters, self._means))

                # measure the degree of change from the previous step for convergence
                difference = self._sum_distances(self._means, new_means)
                if difference < self._max_difference:
                    converged = True

                # remember the new means
                self._means = new_means
                counter +=1


def check_match(pred_labels, true_labels):
    # check match between two lists
    counter_match = 0
    for i in range(len(pred_labels)):
        counter_match += 1 if true_labels[i] == pred_labels[i] else 0
    accuracy_match = counter_match / len(pred_labels)
    return accuracy_match


def decision(X, labels, centers, columns_name):
    decision_point = []
    decision_operators = []
    for i, feature in enumerate(centers[0]):
        if columns_name[i] not in config.categorical:
            decision_point.append((centers[0][i] + centers[1][i]) / 2)
            if decision_point[i] < centers[0][i]:
                decision_operators.append('>')
            else:
                decision_operators.append('<')
        else:
            index_0 = [j for j, e in enumerate(labels) if e == 0]
            decision_point.append(np.unique(X[index_0,i]))
            decision_operators.append('in')

    return decision_point, decision_operators


def decision_labels_by_feature(operator, df_col, decision_val):
    if operator == '>':
        decision_labels = list(1-((df_col > decision_val) *1 ))
    elif operator == '<':
        decision_labels = list(1-((df_col < decision_val)*1))
    elif operator == 'in':
        decision_labels = list(1-(df_col.isin(decision_val)*1))
    return decision_labels


def k_influence_feature(df_only_features, decision_point, decision_operators, labels):
    goodness = 0
    labels_for_feature = []
    split_by = 'data'
    best_set = set()
    columns = list(df_only_features.columns.values)
    feature_options = util.get_power_set(columns)
    labels_per_feture =[]
    for feature_num, feature in enumerate(columns):
        decision_labels = decision_labels_by_feature(decision_operators[feature_num], df_only_features[feature],
                                                 decision_point[feature_num])

        print(feature, 'decision_labels', decision_labels)
        labels_per_feture.append(decision_labels)
    for index_set, set_featutes in enumerate(feature_options):
        print(set_featutes)
        labels_set = [0] * len(labels)
        for e in set_featutes:
            index = columns.index(e)
            labels_set = [1-(1-a)*(1-b) for a,b in zip(labels_per_feture[index], labels_set)]

        accuracy = check_match(labels_set, labels)
        print(set_featutes, ' acc: ', accuracy)

        if accuracy > goodness:
            goodness = accuracy
            best_set = set_featutes
            labels_for_feature = labels_set.copy()
        if goodness == 1.0:
            break

    for col in best_set:
        i = columns.index(col)
        split_by = data_explain(split_by, col, decision_point[i], decision_operators[i])

    return goodness, labels_for_feature, split_by


def one_influence_feature(labels, decision_point, decision_operators, df_only_features):
    """ Check how is the most influence feature in k-means
    input:
    labels: k-means labels.
    centers: The centroids of the groups by k-means.
    decision_point: The split point for every feature.
    goodness: How much close the split of the feature to the full k-means split.
    df_only_features: The df only with features.

    output:
    labels_for_feature:  The labels of the influence feature.
    goodness: How much close the split of the feature to the full k-means split.
    split_by: The explanation for the data split
    """
    # The number of the most influence feature.
    influence_feature = 0
    # The decision point of the feature.
    feature_value = 0
    goodness = 0

    accuracy_per_feature = [0] * len(df_only_features.columns)
    for feature_num, feature in enumerate(df_only_features.columns):
        decision_labels = decision_labels_by_feature(decision_operators[feature_num], df_only_features[feature],
                                                     decision_point[feature_num])

        print('decision_labels', decision_labels)

        accuracy_per_feature[feature_num] = check_match(decision_labels, labels)

        if accuracy_per_feature[feature_num] > goodness:
            goodness = accuracy_per_feature[feature_num]
            feature_value = decision_point[feature_num]
            influence_feature = feature_num
            labels_for_feature = decision_labels.copy()
        if goodness == 1.0:
            break

    print('feature num: ', influence_feature, 'value: ', feature_value)
    # value = util.hour_to_time(feature_value) if df_only_features.columns[influence_feature] == 'time' \
    #     else str(feature_value)
    # split_by = 'data, ' + df_only_features.columns[influence_feature] + ' ' + decision_operators[influence_feature] + ' ' + value
    split_by = 'data'
    split_by = data_explain(split_by, df_only_features.columns[influence_feature], feature_value, decision_operators[influence_feature])
    return goodness, labels_for_feature, split_by


def one_silhouette_influence_feature(df):
    goodness = 0
    max_silhuette = 0
    labels_for_feature = []
    split_by = 'data'
    columns = list(df.columns.values)
    columns.remove('case')
    columns.remove('activity')
    X_all = df.drop(['activity', 'case'], axis=1).values
    for i, col in enumerate(columns):
        c_type = 0
        n_type = 0
        X = X_all[:, i :i+1]
        print(col)
        print(X, type(X))
        if type(X[0][0]) == type(1.0):
            n_type = 1
        else:
            c_type = 1
        dist_cat = lambda u, v: sum([1 if ui != vi else 0 for ui, vi in zip(u, v)])
        dist_num = nltk.cluster.util.euclidean_distance

        # dist = lambda u, v: c_type * dist_num + n_type * dist_cat
        dist = dist_cat if c_type == 1 else dist_num

        kclusterer = myKMeansClusterer(config.num_splits, distance=dist, repeats=25, avoid_empty_clusters=True)
        labels = kclusterer.cluster(X, assign_clusters=True)
        centers = kclusterer.means()
        decision_point, decision_operators = decision(X, labels, centers, [col])
        print('column: ', col, 'labels: ', labels)
        print('centers: ', centers, 'decision_point: ', decision_point, 'decision_operators: ', decision_operators)
        if 1 < len(set(labels)) <= len(X) - 1:  # number of labels is 2 <= n_labels <= n_samples - 1
            silhouette_avg = silhouette_score(X_all, labels)
            print('silhouette: ', silhouette_avg)
            if silhouette_avg > config.silhouette_threshold and silhouette_avg > max_silhuette:
                max_silhuette = silhouette_avg
                goodness = 1
                labels_for_feature = labels
                split_by = data_explain(split_by, col, decision_point[0], decision_operators[0])
    return goodness, labels_for_feature, split_by


def n_influence_feature(labels, decision_point,  decision_operators, df_only_features):
    goodness = 1.0
    labels_for_feature = labels
    split_by = 'data'
    for i, col in enumerate(df_only_features.columns):
        split_by = data_explain(split_by, col, decision_point[i], decision_operators[i])
    return goodness, labels_for_feature, split_by


def data_explain(split_by, col, decision_value, operator):
    if col == 'time':
        value = util.hour_to_time(decision_value)
    elif type(decision_value) == np.ndarray:
        v = []
        for j in decision_value:
            c_index = config.categorical.index(col)
            v.append(config.categorical_map[c_index].get(j))
        value = str(v)
    else:
        value = f'{decision_value:.2f}'
    split_by = split_by + ', ' + col + ' ' + operator + ' ' + value
    return split_by
