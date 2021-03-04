import os
from os.path import join as pjoin
import pickle as p
import random


def create_seed_tree(pkl_path, pkl_tree_path):
    with open(pkl_path, 'rb') as pkl_file:
        filenames = p.load(pkl_file)

    seeds_tree = {}
    for filename in filenames:
        print('Processing {}'.format(filename))
        fname = filename.split('/')[-1]
        file_id, file_piece = fname.split('.')[0], fname.split('.')[1]
        if file_id not in seeds_tree.keys():
            seeds_tree[file_id] = []
        seeds_tree[file_id].append(file_piece)

    with open(pkl_tree_path, 'wb') as tree_file:
        p.dump(seeds_tree, tree_file)


def check_seeds_tree(pkl_tree_path):
    with open(pkl_tree_path, 'rb') as tree_file:
        trees = p.load(tree_file)
    print(trees)


def select_seeds(root_dir, pkl_tree_path, num_segments):
    with open(pkl_tree_path, 'rb') as tree_file:
        trees = p.load(tree_file)
    ids = trees.keys()
    filepieces = []
    for file_id in ids:
        pieces = random.sample(trees[file_id], num_segments)
        for piece in pieces:
            filepiece = file_id + '.' + piece + '.npy'
            filepieces.append(filepiece)
    filepieces = [pjoin(root_dir, fp) for fp in filepieces]
    return filepieces


if __name__ == '__main__':
    pkls_save2 = '/nas/staff/data_work/Sure/Edinburg_Speech/magphase_train.pkl'
    pkl_tree = '/nas/staff/data_work/Sure/Edinburg_Speech/magphase_seeds_tree.pkl'
    # create_seed_tree(pkls_save2, pkl_tree)
    # check_seeds_tree(pkl_tree)
    root_dir = '/nas/staff/data_work/Sure/Edinburg_Speech/processed_5frames_MagPhase'
    filepieces = select_seeds(root_dir, pkl_tree, 2)
    print(filepieces)
