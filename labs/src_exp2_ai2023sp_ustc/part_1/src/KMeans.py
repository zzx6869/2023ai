import numpy as np
import matplotlib.pyplot as plt
import cv2


def read_image(filepath='./data/ustc-cow.png'):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class KMeans:
    def __init__(self, k=4, max_iter=10):
        self.k = k
        self.max_iter = max_iter


    def initialize_centers(self, points):

        n, d = points.shape

        centers = np.zeros((self.k, d))
        for k in range(self.k):
            # use more random points to initialize centers, make kmeans more stable
            random_index = np.random.choice(n, size=10, replace=False)
            centers[k] = points[random_index].mean(axis=0)

        return centers


    def assign_points(self, centers, points):

        n_samples, _ = points.shape
        labels = np.zeros(n_samples)

        for i in range(n_samples):
            distances = np.linalg.norm(points[i] - centers, axis=1)
            labels[i] = np.argmin(distances)

        return labels


    def update_centers(self, centers, labels, points):
        for k in range(self.k):
            cluster_points = points[labels == k]
            if len(cluster_points) > 0:
                centers[k] = np.mean(cluster_points, axis=0)

        return centers

    def fit(self, points):
        centers = self.initialize_centers(points)
        for _ in range(self.max_iter):
            labels = self.assign_points(centers, points)
            new_centers = self.update_centers(centers, labels, points)
            if np.allclose(centers, new_centers):
                break
            ceters = new_centers

        return centers

    def compress(self, img):
        points = img.reshape((-1, img.shape[-1]))
        centers = self.fit(points)
        labels = self.assign_points(centers, points)

        compressed_points = centers[labels.astype(int)]
        compressed_img = compressed_points.reshape(img.shape)

        return compressed_img


if __name__ == '__main__':
    img = read_image(filepath='../data/ustc-cow.png')
    kmeans = KMeans(k=32, max_iter=10)
    compressed_img = kmeans.compress(img).round().astype(np.uint8)

    plt.figure(figsize=(10, 10))
    plt.imshow(compressed_img)
    plt.title('Compressed Image')
    plt.axis('off')
    plt.savefig('./compressed_image.png')