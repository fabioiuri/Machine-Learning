import numpy as np

# logistic function
def logistic(x, deriv=False):
    if(deriv == True): return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# input dataset
X = np.array([  [1,0,0],
                [1,0,1],
                [1,1,0],
                [1,1,1] ])
    
# desired output dataset            
d = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
syn1 = 2*np.random.random((4,1)) - 1
initial_syn0 = syn0
initial_syn1 = syn1

eta = 0.2 # learning rate

for epoch in range(20000):
    for x, i in zip(X, range(X.size)):
        
        ### Forward Step
        l0 = x
        v1 = np.dot(l0, syn0)
        y1 = logistic(v1)
        
        l1 = np.concatenate((x, y1), axis=0)
        v2 = np.dot(l1, syn1)
        y2 = logistic(v2)
        
        
        ### Backward Step
        y2_error = y2 - d[i] # Prevision - desired
        d2 = y2_error * logistic(y2, True)
        new_syn1 = syn1 - eta * d2 * l1.reshape(l1.size, 1)
        
        d1 = (d2 * syn1[-1]) * logistic(l1[-1], True)
        new_syn0 = syn0 - eta * d1 * l0.reshape(l0.size, 1)
        
        # Update weights
        syn1 = new_syn1
        syn0 = new_syn0


for x, i in zip(X, range(X.size)):
    print("x = ", x)
    print("desired = ", d[i])
    
    l0 = x
    v1 = np.dot(l0, syn0)
    l1 = logistic(v1)
    
    l1 = np.concatenate((x, l1), axis=0)
    v2 = np.dot(l1, syn1)
    l2 = logistic(v2)
    print("y =", l2)
    
    
print("\n\n")
 
print("Initial Weights:")
print(initial_syn0)
print(initial_syn1)

print("Weights After Training:")
print(syn0)
print(syn1)

