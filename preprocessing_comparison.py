import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from utils.utility_functions import make_dir

n = 'inc'  #nome cartella features
dir = name = ''
if n:
    dir = f'/{n}'
    name = f'_{n}'
k = 0.1
scelta = int(input('1 = Addestra, 2 = Mostra report: '))
if scelta == 1:
    X_train = np.load(f'features{dir}/X_train{name}.npy', mmap_mode='r')
    y_train = np.load('features/y_train.npy', mmap_mode='r')
    train_size = int(X_train.shape[0]*k)
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    svm = LinearSVC(C=.1, random_state=42, dual=False)
    svm.fit(X_train, y_train)
    del X_train, y_train
    make_dir('models/comparison')
    joblib.dump(svm, f'models/comparison/model{name}.joblib')
    end_time = cv2.getTickCount() / cv2.getTickFrequency()
    print(f"Tempo totale: {end_time - start_time}")
else:
    svm = joblib.load(f'models/comparison/model{name}.joblib')
    X_test = np.load(f'features{dir}/X_test{name}.npy')
    y_test = np.load('features/y_test.npy')
    y_pred = svm.predict(X_test)
    del X_test
    target_names = ['Non pedone', 'Pedone']
    print("Test classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
