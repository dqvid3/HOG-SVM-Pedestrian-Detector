from utils.utility_functions import extract_features, make_dir

make_dir('features')

extract_features('train_assignment', 'train_neg')
extract_features('val_assignment', 'val_neg')
extract_features('test_assignment', 'neg')

#%% Sidenote: volevo usare le matrici sparse ma le hog features non sono effettivamente sparse
'''import numpy as np
A = np.load('assignment_2/features/X_train.npy')
sparsity = np.count_nonzero(A == 0) / A.size
print("Sparsity:", sparsity)'''
