import numpy as np

ux = 1
uy = 2

A = np.array([[ux**2, ux * uy],
              [ux * uy, uy**2]])

def calc_vector_manually(mu, A): 
    A = (A - mu * np.identity(2))

    U, S, Vt = np.linalg.svd(A)

    v = Vt[-1]

    v = v / np.linalg.norm(v)

    print(v)
    return v
    
def calc_mu_manually():
    sum_squares = ux**2 + uy**2
    product_squares = ux**2 * uy**2
    cross_term = (ux * uy)**2

    discriminant = sum_squares**2 - 4 * (product_squares - cross_term)

    mu1 = (sum_squares + np.sqrt(discriminant)) / 2
    mu2 = (sum_squares - np.sqrt(discriminant)) / 2

    return mu1, mu2


eigenvalues, eigenvectors = np.linalg.eig(A)

print("Ref eigenvalues: ", eigenvalues)
print("Ref eigenvectors: ", eigenvectors)

mu1, mu2 = calc_mu_manually()
v1 = calc_vector_manually(mu1, A)
v2 = calc_vector_manually(mu2, A)

print("Calculated eigenvalues: ", [mu1, mu2])
print("Calculated eigenvectors: ", [v1, v2])
