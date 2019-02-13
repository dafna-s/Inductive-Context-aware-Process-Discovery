import config

import os
import subprocess


def save_file_for_miner(df):
    # Save the file for input to the process miner
    process_input_file = os.path.join(config.base_directory, config.edu_prom, config.input_folder, config.input_file)
    os.remove(process_input_file) if os.path.exists(process_input_file) else None
    df[['case', 'activity']].to_csv(os.path.join(process_input_file), encoding='utf-8')


def send_to_IM(df):
    # send to jar
    save_file_for_miner(df)
    process_miner = os.path.join(config.base_directory, 'MyScript.bat')
    subprocess.call([process_miner])


def find_tree():
    # Find the output of the directory for the tree
    f = open(os.path.join(config.base_directory, config.minerout_file), 'r')
    # Read from the file the cut
    lines = f.read()
    answer = lines.find(config.find_tree)
    start_point = answer + len(config.find_tree)
    end_point = lines.find('\n', answer)
    tree = lines[start_point + 1: end_point]
    return tree


def find_cut():
    # Find the output of the directory for the cut
    f = open(os.path.join(config.base_directory, config.minerout_file), 'r')
    # Read from the file the cut
    lines = f.read()
    answer_cut = lines.find(config.find_cut)
    answer = lines.find(config.find_discovered_node)
    discovered_node = ''
    if answer < answer_cut:
        start_point = answer + len(config.find_discovered_node)
        end_point = lines.find('\n', answer)
        discovered_node = lines[start_point + 1: end_point]

    start_cut = answer_cut + len(config.find_cut)
    end_cut = lines.find('\n', answer_cut)
    cut = lines[start_cut + 1: end_cut]
    if discovered_node != '':
        cut = cut[:-1] + ', {' + discovered_node + '}]'
    return cut
