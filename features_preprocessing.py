import math
import cv2
import joblib
import numpy as np
import os
from joblib import Parallel, delayed
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.utility_functions import save

k = .1

def parallel_transform(X, processor, i, batch_size):
    start = i
    end = min(i + batch_size, X.shape[0])
    return processor.transform(X[start:end]).astype(np.float16), start


def transform_data(features_name, processor, batch_size, save_features=True, X=None):
    processor_name = processor.__class__.__name__.lower()
    if X is None:
        if processor_name in ['incrementalpca', 'pca']:
            if processor_name == 'pca':
                X = np.load(f'features/sta/X_{features_name}_sta.npy', mmap_mode='r')
            else:
                X = np.load(f'features/sta/X_{features_name}_sta.npy', mmap_mode='r')
        else:
            X = np.load(f'features/X_{features_name}.npy', mmap_mode='r')
        if features_name == 'train':
            size = int(X.shape[0] * k)
            X = X[:size]
    print(f'Sto applicando {processor_name} transform ai dati di {features_name}...')
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    if processor_name == 'incrementalpca':
        batch_size = min(batch_size, X.shape[0])
        transformed_batches = Parallel(n_jobs=-1)(
            delayed(parallel_transform)(X, processor, i, batch_size)
            for i in tqdm(range(0, X.shape[0], batch_size),
                          desc=f'Transform {processor_name} dei dati di {features_name}')
        )
        transformed_batches.sort(key=lambda batch: batch[1])
        X = np.concatenate([batch[0] for batch in transformed_batches])
        X = X[:, :processor.n_components]
    else:
        X = processor.transform(X)
    end_time = cv2.getTickCount() / cv2.getTickFrequency()
    print(f"{processor_name} transform su {features_name} finito: {end_time - start_time}")
    if save_features:
        if X.dtype != np.float16:
            X = X.astype(np.float16)
        save(f'{features_name}_{processor_name[:3]}', X, new_dir=f'/{processor_name[:3]}')
    else:
        return X


def parallel_fit(X, processor, i, batch_size):
    start = i
    end = min(i + batch_size, X.shape[0])
    processor.partial_fit(X[start:end])
    if end + batch_size >= X.shape[0]:
        return processor


def fit_data(processor, batch_size):
    processor_name = processor.__class__.__name__.lower()
    if processor_name in ['incrementalpca', 'pca']:
        if processor_name == 'pca':
            X = np.load('features/sta/X_train_sta.npy', mmap_mode='r')
        else:
            X = np.load('features/sta/X_train_sta.npy', mmap_mode='r')
        # impericamente le n_components si riduranno di almeno / 3, è per risparmiare calcoli
        n_components = X.shape[1] // 3
        processor.n_components = n_components
    else:
        X = np.load('features/X_train.npy', mmap_mode='r')
    size = int(X.shape[0] * k)
    X = X[:size].astype(np.float32)
    print(f'Sto applicando {processor_name} fit ai dati di train...')
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    if processor_name == 'incrementalpca':
        batch_size = min(batch_size, X.shape[0])
        batch_size = max(batch_size, n_components)
        n_batches = math.floor(X.shape[0] / batch_size)
        ipca_list = Parallel(n_jobs=-1)(
            delayed(parallel_fit)(X, processor, i, batch_size)
            for i in tqdm(range(0, n_batches * batch_size, batch_size),
                          desc='Fitting IPCA dei dati di train')
        )
        processor = ipca_list[-1]
    else:
        processor.fit(X)
    if processor_name in ['incrementalpca', 'pca']:
        cumulative_variance_ratio = np.cumsum(processor.explained_variance_ratio_)
        max_variance = cumulative_variance_ratio[-1]
        if max_variance >= 0.95:
            processor.n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    end_time = cv2.getTickCount() / cv2.getTickFrequency()
    print(f"{processor_name} fit su train finito: {end_time - start_time}")
    return processor


def main():
    _batch_size = 4_000
    preprocessing_choice = int(input("Inserisci la scelta di preprocessing (1: Pca, 2: Standardizza, "
                                     "3: InrementalPca (Pca in parallelo)): "))
    if preprocessing_choice == 1:
        processor = PCA(random_state=42, copy=False)
    elif preprocessing_choice == 2:
        processor = StandardScaler(copy=False)
    elif preprocessing_choice == 3:
        processor = IncrementalPCA(copy=False)
    else:
        print("Scelta non esistente.")
        exit()
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    processor = fit_data(processor, _batch_size)
    preprocessing_name = processor.__class__.__name__.lower()[:3]
    os.makedirs('preprocessing', exist_ok=True)
    joblib.dump(processor, f'preprocessing/{preprocessing_name}.pkl')
    if preprocessing_choice in [1, 3]:
        cumulative_variance = np.cumsum(processor.explained_variance_ratio_)
        variance = cumulative_variance[processor.n_components-1]
        print(f"Componente {processor.n_components} spiega il {variance:.2%} della varianza.")
    features_list = ['train', 'val', 'test']
    Parallel(n_jobs=-1)(
        delayed(transform_data)(features, processor, _batch_size)
        for features in features_list
    )
    for features in features_list:
        transform_data(features, processor, _batch_size)
    end_time = cv2.getTickCount() / cv2.getTickFrequency()
    print(f"Tempo totale preprocessing: {end_time - start_time}")


if __name__ == "__main__":
    main()
