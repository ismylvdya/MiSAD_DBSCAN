from itertools import \
    cycle  # Импортируем функцию для циклического перебора значений (используется для цветов при графической визуализации)
from math import hypot  # Импортируем функцию для вычисления евклидова расстояния
from numpy import random  # Импортируем генератор случайных чисел
import matplotlib.pyplot as plt  # Импортируем библиотеку для построения графиков
import numpy as np  # Импортируем библиотеку для работы с массивами

set1 = set()  # Создаем множество для хранения индексов кластеров, которые уже учтены
s1 = []  # Список для подсчета точек, принадлежащих каждому кластеру


# Функция для подсчета правильности кластеризации
def func(l, clusters):
    print(set1, s1)
    kk = [0 for i in range(len(clusters))]  # Создаем список для подсчета количества точек в каждом кластере
    for i in l:  # Для каждой точки в наборе данных
        for j in range(len(clusters)):  # Для каждого кластера
            if i in clusters[j]:  # Если точка принадлежит кластеру
                kk[j] += 1  # Увеличиваем счетчик для этого кластера
    print('kk=', kk)

    # Проверяем, был ли уже учтен кластер с наибольшим количеством точек
    if np.argmax(np.array(kk)) in set1:
        if kk[np.argmax(np.array(kk))] > s1[np.argmax(np.array(kk))]:
            # Если текущий кластер содержит больше точек, чем ранее, вычисляем разницу
            rez = kk[np.argmax(np.array(kk))] - s1[np.argmax(np.array(kk))]
            s1[np.argmax(np.array(kk))] = kk[np.argmax(np.array(kk))]
            return rez  # Возвращаем разницу как результат
        else:
            return 0  # Если точек не больше, возвращаем 0
    else:
        # Если кластер еще не учтен, добавляем его в set1 и обновляем счетчик точек
        s1[np.argmax(np.array(kk))] = kk[np.argmax(np.array(kk))]
        set1.add(np.argmax(np.array(kk)))
    return kk[np.argmax(np.array(kk))]  # Возвращаем количество точек в кластере


# Алгоритм DBSCAN
def dbscan(P, eps, m, distance):
    NOISE = 0  # Специальная метка для шума (точки, которые не принадлежат кластерам)
    C = 0  # Индекс текущего кластера
    visited_points = set()  # Множество посещенных точек
    clustered_points = set()  # Множество точек, которые уже принадлежат кластерам
    clusters = {NOISE: []}  # Словарь кластеров (кластер с ключом NOISE содержит шум)

    # Функция для поиска соседей точки p в пределах радиуса eps
    def region_query(p):
        return [q for q in P if distance(p, q) < eps]

    # Функция для расширения кластера
    def expand_cluster(p, neighbours):
        if C not in clusters:  # Если текущий кластер не существует, создаем его
            clusters[C] = []
        clusters[C].append(p)  # Добавляем точку p в кластер C
        clustered_points.add(p)  # Добавляем точку p в множество кластеризованных точек
        while neighbours:  # Пока есть соседи
            q = neighbours.pop()  # Извлекаем соседа
            if q not in visited_points:  # Если сосед еще не был посещен
                visited_points.add(q)  # Помечаем его как посещенного
                neighbourz = region_query(q)  # Ищем соседей этой точки
                if len(neighbourz) > m:  # Если соседей достаточно для формирования кластера
                    neighbours.extend(neighbourz)  # Добавляем этих соседей в список для проверки
            if q not in clustered_points:  # Если сосед еще не принадлежит кластерам
                clustered_points.add(q)  # Добавляем его в кластер
                clusters[C].append(q)  # И добавляем его в текущий кластер
                if q in clusters[NOISE]:  # Если эта точка была помечена как шум
                    clusters[NOISE].remove(q)  # Удаляем ее из шума

    for p in P:  # Для каждой точки в наборе данных
        if p in visited_points:  # Если точка уже посещена, пропускаем ее
            continue
        visited_points.add(p)  # Иначе помечаем ее как посещенную
        neighbours = region_query(p)  # Ищем соседей этой точки
        if len(neighbours) < m:  # Если соседей меньше, чем порог m
            clusters[NOISE].append(p)  # Помечаем точку как шум
        else:
            C += 1  # Иначе создаем новый кластер
            expand_cluster(p, neighbours)  # И расширяем кластер
    return clusters  # Возвращаем все кластеры


if __name__ == "__main__":
    # Генерация искусственных данных
    l1, l2, l3, l4, l5 = [], [], [], [], []
    l1 = [(i / 50 - 1.4, (random.randn() / 5) ** 2) for i in range(20)]  # Данные для первого кластера
    l2 = [(random.randn() / 4 + 2.5, random.randn() / 7) for i in range(20)]  # Данные для второго кластера
    l3 = [(random.randn() / 5 + 1, random.randn() / 2 + 1.5) for i in range(20)]  # Данные для третьего кластера
    l4 = [(i / 25 + 1.7, -(i / 50) ** (2) - random.randn() / 20 + 2) for i in range(20)]  # Данные для четвертого кластера
    l5 = [(i / 25 - 2.5, 3 - (i / 50 - 2) ** 2 + random.randn() / 20) for i in range(20)]  # Данные для пятого кластера

    graph = {'0': l1, '1': l2, '2': l3, '3': l4, '4': l5}  # Словарь, хранящий реальные кластеры
    P = []  # Объединенный набор данных
    P.extend(l1)
    P.extend(l2)
    P.extend(l3)
    P.extend(l4)
    P.extend(l5)

    # Запуск DBSCAN для различных значений m и eps
    for m in range(3, 8):
        for e in range(1, 4):
            clusters = dbscan(P, e / 10, m, lambda x, y: hypot(x[0] - y[0], x[1] - y[1]))  # Вызов DBSCAN
            set1 = set()  # Сбрасываем набор учтенных кластеров
            s1 = [0 for i in range(len(clusters))]  # Сбрасываем счетчики кластеров

            # Построение графиков
            fig, axs = plt.subplots(2)
            fig.set_figheight(7 * 2)
            fig.set_figwidth(7)
            axs[0].set_title('DBSCAN' + 'm=' + str(m) + ' eps=' + str(e / 10))
            axs[1].set_title('Реальное разбиение')
            qq = ['#AF5', '#A05', '#F7F', 'b', 'g', 'r', 'c', 'm', 'y', 'k']  # Цвета для кластеров

            # Рисуем результаты DBSCAN
            for c, points in zip(cycle(qq), clusters.values()):
                X = [p[0] for p in points]
                Y = [p[1] for p in points]
                axs[0].scatter(X, Y, c=c)

            # Рисуем реальное распределение
            for c, points in zip(cycle('bgrcmyk'), graph.values()):
                X = [p[0] for p in points]
                Y = [p[1] for p in points]
                axs[1].scatter(X, Y, c=c)

            # Подсчет правильных точек
            x1 = func(l1, clusters)
            x2 = func(l2, clusters)
            x3 = func(l3, clusters)
            x4 = func(l4, clusters)
            x5 = func(l5, clusters)
            print('m=', m, 'eps =', e / 10, 'кол-во верно кластеризованных=', x1 + x2 + x3 + x4 + x5)
            print('Точность =', (x1 + x2 + x3 + x4 + x5) / 100)
