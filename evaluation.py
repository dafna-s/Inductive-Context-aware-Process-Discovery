import config
import os
import util

from pm4py.objects.process_tree import util as pt_util
from pm4py.objects.conversion.process_tree import factory as tree_to_petri
from pm4py.evaluation import factory as evaluation_factory
from pm4py.objects.log.importer.csv.versions import pandas_df_imp
from pm4py.objects.log import transform
from pm4py.objects.petri.exporter import pnml as petri_exporter


def tree_for_eval(tree, activity_set):
    """
    Convert format of string tree for parsing it to process tree.
    :param tree:
    :param activity_set:
    :return: string tree
    """
    tree = tree.lower()
    tree_for_eval = tree.replace('seq', '->')
    tree_for_eval = tree_for_eval.replace('xor', 'X')
    tree_for_eval = tree_for_eval.replace('loop', '*')
    tree_for_eval = tree_for_eval.replace('and', '+')

    for activity in activity_set:
        low_act = activity.lower()
        tree_for_eval = tree_for_eval.replace(low_act, "X('" + activity + "')")
    return tree_for_eval


def export_file(orig_tree, activity_set):
    """
    Export petri net to pnml file
    :param orig_tree: string of the tree from the main algorithm
    :param activity_set: set of all the activities
    :return: pnml file
    """
    string_tree = tree_for_eval(orig_tree, activity_set)
    tree = pt_util.parse(string_tree)
    net, initial_marking, final_marking = tree_to_petri.apply(tree)
    file_name = config.data_file[:config.data_file.find('.')] + '_' + str(config.silhouette_threshold) + '.pnml'
    output_file = os.path.join(config.base_directory, config.data_dir, file_name)
    petri_exporter.export_net(net, initial_marking, output_file, final_marking=final_marking)


activity_set = set(['a0','c0','c1','b2','c2', 'b1','a2', 'd3', 'c3', 'd2', 'b3', 't'])
tree = 'Seq(t, Xor(Xor(Seq(a0,c0),Seq(c1,b1)),Seq(a2,Xor(Xor(d3,c3),d2),b3)))'
df = util.get_data()
export_file(tree, activity_set)
