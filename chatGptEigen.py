import numpy as np
import scipy.linalg as slin

# Example matrices
M = np.array([[2, 0], [0, 1]])  # Mass matrix
K = np.array([[6, -2], [-2, 4]])  # Stiffness matrix
C = np.array([[0.5, 0], [0, 0.5]])  # Damping matrix

# Size of the system
n = M.shape[0]

# Construct the state-space matrix A
A = np.block([
    [np.zeros((n, n)), np.eye(n)],
    [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]
])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = slin.eig(A)

print("Eigenvalues:")
print(eigenvalues)

print("Eigenvectors:")
print(eigenvectors)
