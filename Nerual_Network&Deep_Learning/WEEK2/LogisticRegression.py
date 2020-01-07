## Logistic as Neural Network ##

# Binary Classification #

#Input: x (R)
#Output: y ({0, 1})

# Logistic Regression #

#Parameter: w, b
# y = ∂(w*x + b)
#w is a dimensional vector, b is a real number 

# Sigmoid function 

#Logistic Regression Cost Function 
#Loss function 
#L(yl, y) = 1/2* (yl- y)^2 

#L(yl,y) = -(y*log(yl)+ (1-y)log(1-yl))

#Note: cost function is for the averge
#Loss function is for the entire set

# Gradient Descent #


# Computation Graph #
#One step of backward propagation on a computation graph yields derivative of final output variable.

# Logistic Regression Gradient Descent #
#The value of dw in the function is cumulative 


## Python and Vectorization ##
#Vectorization 

#Vectorization demo 
import numpy as np 
a = np.array([1,2,3,4])
print(a)

import time 
a = np.random.rand(1000000)
b = np.random.rand(1000000)
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print(c)

print("Vectorized Version "+ str(1000*(toc - tic)) + "ms")

c = 0 
tic = time.time()
for i in range(1000000):
    c += a[i]* b[i]
toc = time.time()
print(c)

print("For loop:" + str(1000* (toc - tic))+ "ms")

# The result is no-vectorized version takes much longer than vectorized version 

# Vectorizing Logistic Regression 
# Vectorizing peration without a for loop 
# z = np.dot(w.T, x)
#The dimension of the vectorized function should be (nx, m)

#Quiz 
#How do you compute the derivative of b in one line of code in Python numpy?
#Answer: 1 / m*(np.sum(dz))

#  12/17/2019  #
# Lucid Dream #
#import numpy as np 
#Sum of a column in a matrix 
#In the sample denoted by the cal (calories)
#cal = A.sum(axis = X) (X is the index of the column)
#percentage = 100 * A/(cal.reshape(1,4))

## A note on python and Jupyter Notebook ##
import numpy as np
a = np.random.randn(5,1)
#Transpose pf a 
print(a.T)


