import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    max_prob = np.max(y_pred, axis=1)

    loss = - (1/len(y_true)) * np.sum(np.log(max_prob))

    return loss