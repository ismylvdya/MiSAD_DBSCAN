from datasets import scikit_datasets
from itertools import cycle  # Используется для циклического выбора цветов при визуализации кластеров
from math import sqrt  # для евклидова расстояния
import matplotlib.pyplot as plt


def Euc_distance(x, y):
    '''евклидово расстояние'''
    return sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def dbscan(P, eps, m, distance):
    '''
    функция всего алгоритма DBSCAN

    возвращает словарь структуры {номер кластера : [(точ,ки),(дан,ного),..,(клас,тера)]}

    P: список точек данных;
    eps: радиус (эпсилон) для поиска соседей;
    m: минимальное количество соседей для расширения кластера;
    distance: функция, которая вычисляет расстояние между двумя точками
    '''

    NOISE = 0  # Индикатор для шума (точек, не принадлежащих ни одному кластеру)
    C = 0  # Номер текущего кластера

    visited_points = set()  # Множество посещённых точек
    clustered_points = set()  # Множество точек, уже отнесённых к кластерам
    clusters = {NOISE: []}  # Словарь кластеров, начинаем с кластера "шум"

    def region_query(p):
        """Возвращает всех соседей точки p, которые находятся в радиусе eps"""
        return [q for q in P if distance(p, q) < eps]  # Фильтрация точек на основе расстояния

    def expand_cluster(p, neighbours):
        """Расширение кластера для точки p, если она может стать ядром"""
        if C not in clusters:
            clusters[C] = []  # Инициализация нового кластера
        clusters[C].append(p)  # Добавление точки p в кластер C
        clustered_points.add(p)  # Добавляем точку p в множество точек, относящихся к кластерам
        while neighbours:  # Пока есть соседи
            q = tuple(neighbours.pop())  # Извлекаем соседа q из списка
            if q not in visited_points:  # Если точка ещё не посещена
                visited_points.add(q)  # Помечаем её как посещённую
                neighbourz = region_query(q)  # Ищем соседей для этой точки
                if len(neighbourz) > m:  # Если достаточно соседей, добавляем их в список для дальнейшего поиска
                    neighbours.extend(neighbourz)
            if q not in clustered_points:  # Если точка ещё не принадлежит никакому кластеру
                clustered_points.add(q)  # Добавляем её в кластер
                clusters[C].append(q)  # Включаем её в текущий кластер C
                if q in clusters[NOISE]:  # Если точка была помечена как шум, убираем её из шума
                    clusters[NOISE].remove(q)

    for p in P:
        p_tuple = tuple(p)
        if p_tuple in visited_points:  # Пропускаем точки, которые уже были обработаны
            continue
        visited_points.add(p_tuple)  # Помечаем точку как посещённую
        neighbours = region_query(p_tuple)  # Ищем соседей
        if len(neighbours) < m:  # Если соседей меньше, чем m, помечаем точку как шум
            clusters[NOISE].append(p_tuple)
        else:
            C += 1  # Инициализируем новый кластер
            expand_cluster(p_tuple, neighbours)  # Расширяем кластер начиная с этой точки

    return clusters  # Возвращаем все кластеры (включая точки, помеченные как шум)


def plot_our_clusterisation(clusters):
    '''визуализация полученной кластеризации'''
    for c, points in zip(cycle('bgrcmykgrcmykgrcmykgrcmykgrcmykgrcmyk'), clusters.values()):
        X = [p[0] for p in points]  # Координаты X всех точек кластера
        Y = [p[1] for p in points]  # Координаты Y всех точек кластера
        plt.scatter(X, Y, c=c)  # Отображение точек кластера на графике с цветом c
    plt.show()  # Показать график


def plot_etalon_clusterisation(dataset):
    # Создание пользовательской цветовой карты триколора
    from matplotlib.colors import LinearSegmentedColormap  # для своего триколора
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap',['orangered', 'royalblue', 'gold'])

    points = dataset[0]
    if dataset[1] is not None:
        etalon_clusters = dataset[1]

        plt.scatter(points[:, 0], points[:, 1], c=etalon_clusters, cmap=custom_cmap)
    else:
        plt.scatter(points[:, 0], points[:, 1])

    plt.show()  # Показать график


if __name__ == "__main__":

    dataset = scikit_datasets[4]

    points = dataset[0]

    # Запуск DBSCAN
    clusters = dbscan(points, 0.2, 6, Euc_distance)  # Вызываем алгоритм DBSCAN

    print('полученный словарь clusters:')
    for i in range(len(clusters)):
        print(f"{list(clusters.keys())[i]:2d} ({len(clusters[i]):3d} шт) :", clusters[i])

    # Визуализация нашей кластеризации
    plot_our_clusterisation(clusters)

    # Визуалаизация эталонной кластеризации
    plot_etalon_clusterisation(dataset)
