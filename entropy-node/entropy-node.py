import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y)
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    print(counts) 
    
    prob = counts / len(y)

    entropy = - np.sum(prob * np.log2(prob))

    return entropy