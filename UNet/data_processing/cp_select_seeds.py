import os
from data_processing.select_pkls import select_seeds
pkl_tree = '/nas/staff/data_work/Sure/Edinburg_Speech/magphase_seeds_tree.pkl'
root_dir = '/nas/staff/data_work/Sure/Edinburg_Speech/processed_5frames_MagPhase'
filepieces = select_seeds(root_dir, pkl_tree, 1)
print(filepieces)

for filep in filepieces:
    print('Processing {}'.format(filep))
    os.system('cp {} {}'.format(filep,
                                filep.replace('processed_5frames_MagPhase', 'processed_5frames_MagPhase_sectionX/section1')))

