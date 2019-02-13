# Folders
data_dir = 'dana'
# data_dir = 'Synthetic_log'
output_dir = 'Output'
input_folder = 'EventLogs'  # folder
edu_prom = 'EDU-ProM-master'  # folder
base_directory = ''

# Files
input_file = 'test.csv'
data_file = 'dana1-no-f.csv'
minerout_file = 'minerout.txt'

# Constance
find_cut = 'cut:'
find_discovered_node = 'discovered node'
find_tree = 'tree'
tree = ''
operator = 'Xor'

# Hyper parameter
num_splits = 2
threshold_for_data_split = 1.0
silhouette_threshold = 0.45
# The number of explainable features
kmeans_feature = 'n'  # can be '1' or 'k' or 'n' or '1s'
"""
example for k:
data_file = 'LogEx1-k.csv'
threshold_for_data_split = 0.85
silhouette_threshold = 0.1
"""

# categorical = ['resource']
categorical = ['resource1', 'resource2']
# categorical = ['resource', 'case id']

operators = ['xor', 'and', 'loop', 'seq']
# Help parameters
categorical_map = []
explain_operator = []
process_tree = []
tree_traces = []
silhouette = []
num_traces_log = 0
