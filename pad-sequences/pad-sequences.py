import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        L = max(len(seq) for seq in seqs) if seqs else 0
    else:
        L = max_len
    N = len(seqs)
    
    padded_matrix = np.full((N,L), pad_value)

    for i, seq in enumerate(seqs):
        length_to_copy = min(len(seq), L)
        padded_matrix[i, :length_to_copy] = seq[:length_to_copy]
    return padded_matrix
        
        
    