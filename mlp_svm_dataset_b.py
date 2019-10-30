import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    dataset_bx = pd.read_csv('unknown_dataset/BX.csv', header=None)
    dataset_by = pd.read_csv('unknown_dataset/BY.csv', header=None)
    
    print(dataset_bx.head())
    print(dataset_by.head())

    print("\nLength of Dataset BX =", len(dataset_bx.values))
    print("Length of Dataset BX =", len(dataset_by.values))

    print("\nType of Dataset BX =", type(dataset_bx))
    print("Type of Dataset BY =", type(dataset_by))

    print("\nShape of Dataset BX =", dataset_bx.shape)
    print("Shape of Dataset BY =", dataset_by.shape)

    all_as = dataset_bx.iloc[:,0]
    all_bs = dataset_bx.iloc[:,1]
    all_cs = dataset_bx.iloc[:,2]
    
    all_as = all_as.values.tolist()
    all_bs = all_bs.values.tolist()
    all_cs = all_cs.values.tolist()

    x = np.arange(-max(dataset_by.max()), max(dataset_by.max()), 0.1)
    
    for a, b, c, roots in zip(all_as, all_bs, all_cs, dataset_by.values):
        if ((b**2) - 4*a*c) >= 0:
            y = a*(x**2) + b*x + c
            y_ = a*(roots**2) + b*roots + c
            plt.plot(x, y)
            plt.plot(roots, y_, 'ro')

    plt.title('Second Degree Polynomio for each set of coeficients')
    plt.show()