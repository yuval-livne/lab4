from sys import argv
import os
from cross_validation import CrossValidation
from knn import KNN
from metrics import accuracy_score
from normalization import *


def load_data():
    """
    Loads data from path in first argument
    :return: returns data as list of Point
    """
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print(os.getcwd())
        print('Input file does not exist')
        exit(1)

    points = []
    with open(input_path, 'r') as f:
        for index, row in enumerate(f.readlines()):
            row = row.strip()
            values = row.split(',')
            points.append(Point(str(index), values[:-1], values[-1]))
    return points


def run_knn(points):
    m = KNN()
    m.train(points)
    # print(f'predicted class: {m.predict(points[0])}')
    # print(f'true class: {points[0].label}')
    cv = CrossValidation()
    cv.run_cv(points, 10, m, accuracy_score, print_final_score=False)


def implementation(points):
    # Q1
    # m = KNN()
    # m.train(points)
    # predicted = m.predict(points)
    # real = [point.get_label() for point in points]
    # print(sum([real[i] == predicted[i] for i in range(len(real))]) / len(real))

    # Q2
    max = 0
    best_k = 0
    #  this part
    for i in range(30):
        m = KNN(i + 1)
        m.train(points)
        cv = CrossValidation()
        temp_average_score = cv.run_cv(points, len(points), m, accuracy_score, print_final_score=False)
        if max < temp_average_score:
            max = temp_average_score
            best_k = i + 1

    print("Question 3:")
    print(f'K={best_k}')
    list_n_folds = [2, 10, 20]
    k_q3 = KNN(best_k)
    k_q3.train(points)
    for n in list_n_folds:
        print(f'{n}-fold-cross-validation:')
        # print(f'K={best_k}')
        cv.run_cv(points, n, K_Q3, accuracy_score, print_final_score=False, print_fold_score=True)

    print("Question 4:")
    list_k = [5, 7]
    dummy = DummyNormalizer()
    z_norm = ZNormalizer()
    sum_norm = SumNormalizer()
    min_max_norm = MinMaxNormalizer()
    list_norm = [dummy, sum_norm, min_max_norm, z_norm]
    n_folds_q4 = 2
    for k in list_k:
        k_q4 = KNN(k)
        print(f'K={k}')
        for norm in list_norm:
            norm.fit(points)
            t_points = norm.transform(points)
            k_q4.train(t_points)
            avg_acc = cv.run_cv(t_points, n_folds_q4, k_q4, accuracy_score,
                                print_final_score=False, print_fold_score=True)
            print('Accuracy of {} is {:.2f}\n'.format(norm.print_name(), avg_acc))


if __name__ == '__main__':
    loaded_points = load_data()
    # for point in loaded_points:
    #     print(point)
    run_knn(loaded_points)
    implementation(loaded_points)
