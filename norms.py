import numpy as np
import matplotlib.pyplot as plt
import copy


def find_norms(image, basis_matrices):
    all_norms = [0]*10
    for i in range(10):
        Z = copy.deepcopy(image)
        for j in range(200):
            z_j = np.trace(np.matmul(np.transpose(basis_matrices[i][j]), Z))
            z_j /= np.linalg.norm(basis_matrices[i][j])**2
            Z -= z_j*basis_matrices[i][j]
        all_norms[i] = np.linalg.norm(Z)
        print("all_norms(" + str(i) + "): ", all_norms[i])
    return all_norms


if __name__ == "__main__":
    predicted_values = []
    bm = np.load("basis_matrices.csv.npy")
    basis_matrices = []
    for i in range(10):
        basis_matrices.append(bm[i])

    test_labels = np.load("test_labels.csv.npy")
    ti = np.load("test_images.csv.npy")
    test_images = []
    for i in range(100):
        test_images.append(ti[i].reshape((28,28)))
    cnt = 0
    for i in range(len(test_images)):
        img = test_images[i]
        imgplot = plt.imshow(img)
        plt.show()
        norms = find_norms(img, basis_matrices)
        label = np.argmin(norms)
        print("Predicted value: ", label)
        if label == test_labels[i]:
            cnt += 1
        predicted_values.append(label)
        score = cnt/len(test_images)
        score *= 100
    #print(predicted_values)
    print("Score: " + str(score) + "%")