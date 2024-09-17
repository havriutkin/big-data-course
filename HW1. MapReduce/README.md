# Homework 1. Map Reduce

## Task
Implement matrix by vector multiplication. Use MapReduce approach and regular approach. Compare them.

## Implementation
Tha Map Reduce approach works as follows:
    - map - function that accepts row and vector. Returns a scalar - their product as a result
    - reduce - function that aggregates all of the results. In this case is trivial, since we just need to put all scalars into one list, that represents a vector.

The regular approach:
    - Just standard iterative solution 

In each case we measure time that it took function to execute. 
In both approaches time complexity is O(n^2), where n - size of a vector (and a matrix).

### Result table

| Size   | Time without Multiprocessing (s) | Time with Multiprocessing (s) | Improvement (s) | Improvement (%) |
|--------|----------------------------------|-------------------------------|-----------------|-----------------|
| 100    | 0.001075                         | 0.139275                      | -0.138          | -12852.594      |
| 1000   | 0.068998                         | 0.178678                      | -0.110          | -158.959        |
| 10000  | 6.540007                         | 4.579022                      | 1.961           | 29.984          |
| 20000  | 138.553383                       | 127.172389                    | 11.381          | 8.214           |


### Map Reduce Results
We can see, how for smaller sizes of matrix (< 10000) map reduce approach takes longer. It is related to the fact, that it takes additional time for process switching. But, as size of the data grows, advantage over the regular approach becomes more explicit.

### Regular Approach Results
For smaller sizes of the data, iterative approach works better, since it doesn't spend time on context-switching. But as size of the data grows, single-process computation can not compete with multiprocessing one.

## Conclusion
One should prefer Map Reduce for big data. At the same time, if there is relatively small amount of data it would make sense to choose single-process approach.