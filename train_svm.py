import os
import cv2
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import LinearSVC
import joblib
import matplotlib.pyplot as plt
from utils.utility_functions import make_dir, get_npy_shape

k = 1
# Definisci la griglia di iperparametri da esplorare
C_values = np.logspace(-2, 2, 5)
print(C_values)
n = 'pca'
dir = name = ''
if n:
    dir = f'/{n}'
    name = f'_{n}'
# Carica le features e le etichette di training
train_path = f'features{dir}/X_train{name}.npy'
#X_train = np.memmap(train_path, mode='r', dtype='float16', shape=get_npy_shape(train_path))
X_train = np.load( f'features{dir}/X_train{name}.npy', mmap_mode='r')
y_train = np.load('features/y_train.npy', mmap_mode='r')
train_size = int(X_train.shape[0]*k)
X_train = X_train[:train_size]
y_train = y_train[:train_size]

# Esegui la ricerca grid per trovare il miglior valore di C
make_dir('models')
start_time = cv2.getTickCount() / cv2.getTickFrequency()
results = {}

for C in C_values:
    svm_model = LinearSVC(C=C, dual=False)
    print(f'Sto addestrando con C = {C}...')
    svm_model.fit(X_train, y_train)
    X_val = np.load(f'features{dir}/X_val{name}.npy')
    y_val = np.load('features/y_val.npy')
    y_pred = svm_model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    results[C] = f1
    joblib.dump(svm_model, f'models/svm_model_C_{C}.joblib')
    
end_time = cv2.getTickCount() / cv2.getTickFrequency()
print(f"Fine ricerca miglior C: {end_time - start_time}")
# Stampa le prestazioni nel grafico
plt.plot(list(results.keys()), list(results.values()))
plt.xlabel('Valore di C')
plt.ylabel('F1-score')
plt.xscale('log')
plt.title('F1-score ottenuto al variare di C')
make_dir('plots')
plt.savefig('plots/prestazioni.png')
plt.show()

best_C = max(results, key=results.get)
print(f'best_C: {best_C}')
os.rename(f'models/svm_model_C_{best_C}.joblib', f'models/best_svm_model_{n}.joblib')
best_model = joblib.load(f'models/best_svm_model_{n}.joblib')

X_test = np.load(f'features{dir}/X_test{name}.npy')
y_test = np.load('features/y_test.npy')
y_pred = best_model.predict(X_test)

print("Test classification report:")
print(classification_report(y_test, y_pred, target_names=['Non pedone', 'Pedone']))
