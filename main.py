from datasets import scikit_datasets, points_count  # 5 датасетов данных в условии лабораторной и количество точек в датасете
from math import sqrt  # для евклидова расстояния
import numpy as np  # для пробегания разных значений eps
import matplotlib.pyplot as plt


def Euc_distance(x, y):
    '''евклидово расстояние'''
    return sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def dbscan(P, eps, m, distance):
    '''
    функция всего алгоритма DBSCAN

    возвращает словарь структуры {номер кластера : [(точ,ки),(дан,ного),..,(клас,тера)]}

    :param P: список точек данных;
    :param eps: радиус (эпсилон) для поиска соседей;
    :param m: минимальное количество соседей для расширения кластера;
    :param distance: функция, которая вычисляет расстояние между двумя точками
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
            q = tuple(neighbours.pop())  # Извлекаем соседа q из списка. tuple() -- т.к. нумпаевский массив не хешируем
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

        p_tuple = tuple(p)  # т.к. нумпаевский массив не хешируем
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


def dbscan_running(eps1, eps2, eps_count, m1, m2, dataset):
    '''
    прогоняет внутри себя DBSCAN для значений [eps1,eps2] и [m1, m2] и выводит таблицу с accuracy для каждого сочетания eps m. Потом эту таблицу удобно раскрашивать в excel

    :param eps1: первое значение eps
    :param eps2: последнее значение eps
    :param eps_count: сколько значений eps должно быть в этом отрезке
    :param m1: первое значение m
    :param m2: последнее значение m
    '''
    cells_count = 0
    str_table = '|      |'
    for eps in np.linspace(eps1, eps2, eps_count):
        str_table += f'{eps:6.2f}|'
        cells_count += 1
    cells_count *= m2 + 1 - m1
    calculated_cells_count = 0

    # # если нужен markdown формат:
    # str_table += '\n|------|'
    # for eps in np.linspace(eps1, eps2, eps_count):
    #     str_table += '------|'

    for m in range(m1, m2 + 1):
        str_table += f'\n|{m:6d}|'
        for eps in np.linspace(eps1, eps2, eps_count):
            # DBSCAN для данных eps и m
            clusters = dbscan(dataset[0], eps, m, Euc_distance)  # Вызываем алгоритм DBSCAN

            # Сравнение нашей кластеризации с эталонной
            clusters_list = our_clusterisation_but_etalon_like(dataset, clusters)
            etalon_clusters_list = dataset[1].tolist()
            (matches_count, diff_indexes) = matches_counts_in(clusters_list, etalon_clusters_list)

            accuracy = 100 * matches_count / points_count

            calculated_cells_count += 1
            print(f'\r{calculated_cells_count:2d}/{cells_count:2d}', end='')

            str_table += f'{accuracy:6.2f}|'
    print(f'\rвыполнено {cells_count} вариаций DBSCAN (по строкам m, по столбцам eps):')
    print(str_table)


def print_our_clusters(clusters):
    '''печатает словарь полученного распределения по кластерам'''
    print('полученный словарь clusters:')
    for i in range(len(clusters)):
        print(f"{list(clusters.keys())[i]:2d} ({len(clusters[i]):3d} шт) :", clusters[i])


def our_clusterisation_but_etalon_like(dataset, clusters):
    ''' возвращает словарь нашей класстеризации но в формате привычного списка (расмером с количество точек, его элементами являются номера кластеров)

    это нужно чтобы сравнить полученную кластеризацию с эталонной, в которой как раз-таки массив кластеров имеет такой вид'''

    new_clusters_list = []
    for point in dataset[0].tolist():  # point тут это ЛИСТ-координаты данной точки
        for our_cluster, our_points in clusters.items():
            if tuple(point) in our_points:
                new_clusters_list.append(our_cluster)
                break  # переход к след point

    return new_clusters_list


def matches_counts_in(clusters_list, targets_list):
    '''возвращает тьюпл формата (количество совпадающих точек, индексы несовпадений) между нашей кластеризацией (на k кластеров) и эталонной С УЧЕТОМ РАЗНОЙ НУМЕРАЦИИ КЛАСТЕРОВ'''

    k = len(set(targets_list))  # количество кластеров которое должно быть исходя из targets_list

    matches_count = 0
    diff_indexes = []

    posible_pairs = {}  # словарь типа {(элемент_из_clusters, элемент_из_targets) : ск_раз_встретилась_эта_пара}

    for i in range(points_count):
        cur_pair = (clusters_list[i], targets_list[i])
        if cur_pair not in posible_pairs:
            posible_pairs[(clusters_list[i], targets_list[i])] = 1
        else:
            posible_pairs[(clusters_list[i], targets_list[i])] += 1

    top_k_pairs = sorted(posible_pairs, key=posible_pairs.get, reverse=True)[:k]  # -- массив из тех K пар кластеров, которые встретились чаще других

    # проверка на уникальность кластеров в нашей кластеризации
    top_k_pairs_modifided = []
    our_set_clusters = set()
    for (our, targ) in top_k_pairs:
        if our not in our_set_clusters:  # если кластер еще не был встречен
            our_set_clusters.add(our)  # добавляем кластер в множество
            top_k_pairs_modifided.append((our, targ))
    # теперь top_k_pairs_modifided -- массив в котором НЕТ тьюплов с повторяющимися первыми (т.е. нашими) кластерами

    for i in range(points_count):
        cur_pair = (clusters_list[i], targets_list[i])
        if cur_pair in top_k_pairs_modifided:
            matches_count += 1
        else:
            diff_indexes.append(i)

    return (matches_count, diff_indexes)


def plot_our_clusterisation(clusters, matches_count, eps, m):
    '''визуализация нашей кластеризации (палитра из 10-ти цветов)
    аргументы matches_count, eps, m -- для их вывода на изображение под и над графиком
    '''

    # Создание пользовательской цветовой карты (10 цветов)
    from matplotlib.colors import LinearSegmentedColormap  # для своего триколора
    custom_cmap_ten = LinearSegmentedColormap.from_list('custom_cmap',
                                                        ['black', 'orangered', 'royalblue', 'gold', '#9467bd',
                                                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    pointsX, pointsY = [], []  # списки абсцис и ординат каждой точки
    clusters_list_for_plot = []  # списоко кластеров для каждой точки
    for our_cluster, points_list in clusters.items():
        for point in points_list:
            pointsX.append(point[0])
            pointsY.append(point[1])
            clusters_list_for_plot.append(our_cluster)

    plt.scatter(pointsX[:], pointsY[:], c=clusters_list_for_plot, cmap=custom_cmap_ten)

    # Подписываем график и оси
    plt.title(f'DBSCAN  eps = {eps}  m = {m}')
    plt.xlabel(f'правильно кластеризованных: {matches_count} из {points_count}')

    plt.show()


def plot_etalon_clusterisation(dataset):
    ''' визуализация эталонной кластеризации (палитра из 3-ех цветов)'''

    # Создание пользовательской цветовой карты (3 цвета)
    from matplotlib.colors import LinearSegmentedColormap  # для своего триколора
    custom_cmap_three = LinearSegmentedColormap.from_list('custom_cmap', ['orangered', 'royalblue', 'gold'])

    points = dataset[0]
    etalon_clusters = dataset[1]

    plt.scatter(points[:, 0], points[:, 1], c=etalon_clusters, cmap=custom_cmap_three)
    plt.title('эталонная кластеризация')

    plt.show()  # Показать график


# здесь можно выбрать датасет из 5-ти предложенных
dataset = scikit_datasets[3]  # (0-4)


############################################################
##### прогонка DBSCAN для нескольких значений eps и m: #####
############################################################

dbscan_running(0.1, 0.75, 8, 1, 25, dataset)


############################################################
##### прогонка DBSCAN для конкретных значений eps и m: #####
############################################################

# # Запуск DBSCAN
# eps = 0.38
# m = 4
# clusters = dbscan(dataset[0], eps, m, Euc_distance)  # Вызываем алгоритм DBSCAN
#
# # Сравнение нашей кластеризации с эталонной
# clusters_list = our_clusterisation_but_etalon_like(dataset, clusters)
# etalon_clusters_list = dataset[1].tolist()
# (matches_count, diff_indexes) = matches_counts_in(clusters_list, etalon_clusters_list)
#
# # печать словаря получившихся кластеров
# print_our_clusters(clusters)
#
# # вычистывание и печать точности (процент правильно кластеризированных точек)
# accuracy = 100 * matches_count / points_count
# print('accuracy:', accuracy)
#
# # Визуализация нашей кластеризации
# plot_our_clusterisation(clusters, matches_count, eps, m)
#
# # Визуализация эталонной кластеризации
# plot_etalon_clusterisation(dataset)