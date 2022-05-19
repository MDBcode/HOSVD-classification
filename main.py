import numpy as np
from keras.datasets import mnist
import tensorly
import tensorly.tenalg.core_tenalg as tl
import random

tensors_list = [None]*10
test_images = [None]*100
test_labels = [0]*100

def get_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    for i in range(10):
        digits_i = train_X[train_y == i]
        digits_i_reduced = digits_i[:200].flatten() / 255
        np.save(str(i) + ".csv", digits_i_reduced)

        for j in range(10):
            rnd = random.randint(0,len(test_X))
            test_images[i*10+j] = test_X[rnd].flatten() / 255
            test_labels[i*10+j] = test_y[rnd]

        tensors_list[i] = np.load(str(i) + ".csv.npy")
        tensors_list[i] = tensors_list[i].reshape((200,28,28))

    np.save("test_images.csv", test_images)
    np.save("test_labels.csv", test_labels)

def HOSVD(tensor):
    U1, _, _ = np.linalg.svd(tensorly.base.unfold(tensor,0))
    U2, _, _ = np.linalg.svd(tensorly.base.unfold(tensor,1))
    U3, _, _ = np.linalg.svd(tensorly.base.unfold(tensor,2))
    S = tl.mode_dot(tl.mode_dot(tl.mode_dot(tensor, np.transpose(U1), 0), np.transpose(U2), 1), np.transpose(U3), 2)
    return S,U1,U2,U3

#izračunaj bazne matrice za svaki tenzor 0,...,9
def find_basis_matrices(tensors_list):
    basis_matrices = []
    for i in range(10):
        basis_matrices.append(np.zeros((200,28,28)))

    for index,tensor in enumerate(tensors_list):
        S, U1, U2, U3 = HOSVD(tensor)
        tensor_check = np.zeros(
            (200, 28, 28))  # provjera - na kraju treba biti isti kao tensor - dobiva se pomocu baznih matrica
        for i in range(200):  # 200 baznih matrica za svaki tenzor
            b_matrix = np.matmul(np.matmul(U2, S[i, :, :]), np.transpose(U3))
            tensor_check += tl.mode_dot(b_matrix.reshape(1, 28, 28), U1[:, i].reshape(200, 1), 0)
            b_matrix /= np.linalg.norm(b_matrix)
            basis_matrices[index][i, :, :] = b_matrix
        
    np.save("basis_matrices.csv", basis_matrices)


if __name__ == "__main__":
    get_data()
    find_basis_matrices(tensors_list)