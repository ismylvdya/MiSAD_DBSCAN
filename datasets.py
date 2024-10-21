'''в модуле инициализируются 6 датасетов заданного размера средствами библиотеки sklearn (scikit)

scikit_datasets -- тьюпл из этих 6-ти датасетов'''

from sklearn import datasets
import numpy as np

n_samples = 800  # количество точек в датасете
seed = 30  # начальное значение для рандомизации (чтобы было постояноство в НУ)

# все они 👇 (кроме no_structure)️ имеют структуру ( [ [коорд,инаты],[каж,дой],..,[точ,ки] ], [массив_эталонного_распределения] )
# no_structure       имеет       структуру        ( [ [коорд,инаты],[каж,дой],..,[точ,ки] ], None )
# (!) в ней каждый массив это нумпаевский массив

# 1 noisy_circles
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)

# 2 noisy_moons
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)

# 3 blobs
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)

# 4 no_structure
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# 5 aniso
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# 6 varied
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

scikit_datasets = (noisy_circles, noisy_moons, blobs, no_structure, aniso, varied)

