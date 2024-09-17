import multiprocessing
import time
import random

# Function maps row and vector to a vector and stores the result in the result list
def map(row, vector):
    return sum(row[i] * vector[i] for i in range(len(vector)))

# Trivial reduce function, since our result is already a list
def reduce(result):
    return result

# Matrix by vector multiplication using multiprocessing
def mul_using_multiprocessing(matrix, vector, timer=False):
    start, end = 0, 0
    if timer:
        start = time.time()

    results = []

    with multiprocessing.Pool() as pool:
        res = pool.starmap(map, [(matrix[i], vector) for i in range(len(matrix))])
        results.append(res)

    result = reduce(results)

    if timer:
        end = time.time()
        print("     Time taken using multiprocessing: ", end - start)

    return (result, end - start)

# Matrix by vector multiplication without using multiprocessing
def mul_without_multiprocessing(matrix, vector, timer=False):
    start, end = 0, 0
    if timer:
        start = time.time()

    result = [0] * len(matrix)

    for i in range(len(matrix)):
        result[i] = sum(matrix[i][j] * vector[j] for j in range(len(vector)))

    result = reduce(result)

    if timer:
        end = time.time()
        print("     Time taken without multiprocessing: ", end - start)

    return (result, end - start)

# Function to generate a random matrix and vector
def generate_matrix_and_vector(size):
    matrix = [[random.random() for _ in range(size)] for _ in range(size)]
    vector = [random.random() for _ in range(size)]
    return matrix, vector

# Main function
def main():
    sizes = [100, 1000, 10000, 20000]
    results = {}

    for size in sizes:
        matrix, vector = generate_matrix_and_vector(size)

        print("Size: ", size)
        multi_res = mul_using_multiprocessing(matrix, vector, timer=True)
        no_multi_res = mul_without_multiprocessing(matrix, vector, timer=True)

        results[size] = (multi_res, no_multi_res)

if __name__ == "__main__":
    main()

"""
Result Output:

Size:  100
     Time taken using multiprocessing:  0.13927507400512695
     Time taken without multiprocessing:  0.0010752677917480469
Size:  1000
     Time taken using multiprocessing:  0.17867755889892578
     Time taken without multiprocessing:  0.06899833679199219
Size:  10000
     Time taken using multiprocessing:  4.57902193069458
     Time taken without multiprocessing:  6.540006637573242
Size:  20000
     Time taken using multiprocessing:  127.17238879203796
     Time taken without multiprocessing:  138.55338287353516
"""