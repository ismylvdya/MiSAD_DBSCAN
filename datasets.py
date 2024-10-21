'''в модуле инициализируются 5 датасетов (все двумерные) заданного размера средствами библиотеки sklearn (scikit)

импортируемый в main scikit_datasets -- тьюпл из этих 5-ти датасетов'''

from sklearn import datasets
import numpy as np

points_count = 800  # количество точек в датасете
seed = 30  # начальное значение для рандомизации (чтобы было постояноство в НУ)

# все 5 нижеинициализируемых датасетов имеют структуру ( [ [коорд,инаты],[каж,дой],..,[точ,ки] ], [массив_эталонного_распределения] )
# (!) в ней каждый массив это нумпаевский массив

# 0 noisy_circles
noisy_circles = datasets.make_circles(n_samples=points_count, factor=0.5, noise=0.05, random_state=seed)

# 1 noisy_moons
noisy_moons = datasets.make_moons(n_samples=points_count, noise=0.05, random_state=seed)

# 2 blobs
blobs = datasets.make_blobs(n_samples=points_count, random_state=seed)

# 3 aniso
random_state = 170
X, y = datasets.make_blobs(n_samples=points_count, random_state=random_state)
transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# 4 varied
varied = datasets.make_blobs(n_samples=points_count, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

scikit_datasets = (noisy_circles, noisy_moons, blobs, aniso, varied)