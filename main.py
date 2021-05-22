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
        cv.run_cv(points, 10, m, accuracy_score,print_final_score=False)




#  this part is the code for q2

def implementation(points):
    max = 0
    best_k = 0
    #  this part
    for i in range(30):
        m = KNN(i + 1)
        m.train(points)
        # print(f'predicted class: {m.predict(points[0])}')
        # print(f'true class: {points[0].label}')
        cv = CrossValidation()
        temp_average_score = cv.run_cv(points, 100, m, accuracy_score,print_final_score=False)
        if max < temp_average_score:
            max = temp_average_score
            best_k = i + 1
    print("Question 3:")
    print(f'K={best_k}')
    list=[2,10,20]
    K_Q3=KNN(7)
    K_Q3.train(points)
    for i in list:
        print(f'{i}-fold-cross-validation:')
        temp_average_score = cv.run_cv(points, i, K_Q3, accuracy_score,print_final_score=False,print_fold_score=True)

    print("Question 4:")
    K_Q3 = KNN(7)





if __name__ == '__main__':
    loaded_points = load_data()
    run_knn(loaded_points)
    implementation(loaded_points)

