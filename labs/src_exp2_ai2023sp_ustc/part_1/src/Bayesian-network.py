import numpy as np


class BayesianNetwork:
    def __init__(self, n_labels=10, n_pixels=784, n_values=2) -> None:
        self.n_labels = n_labels
        self.n_pixels = n_pixels
        self.n_values = n_values
        self.labels_prior = np.zeros(n_labels)
        self.pixels_prior = np.zeros((n_pixels, n_values))
        self.pixels_cond_label = np.zeros((n_pixels, n_values, n_labels))

    def fit(self, pixels, labels):
        n_samples = len(labels)
        labels_counts = np.bincount(labels, minlength=self.n_labels)
        self.labels_prior = labels_counts / n_samples

        for pixel in range(self.n_pixels):
            for value in range(self.n_values):
                for label in range(self.n_labels):
                    pixel_value_mask = (pixels[:, pixel] == value)
                    label_pixel_mask = (labels == label)
                    count = np.sum(pixel_value_mask & label_pixel_mask)
                    self.pixels_cond_label[pixel, value, label] = count / labels_counts[label]

    def predict(self, pixels):
        n_samples = len(pixels)
        labels_pred = np.zeros(n_samples, dtype=np.int8)

        for i in range(n_samples):
            probabilities = np.zeros(self.n_labels)
            for label in range(self.n_labels):
                label_prob = self.labels_prior[label]
                for pixel in range(self.n_pixels):
                    pixel_value = pixels[i, pixel]
                    pixel_prob = self.pixels_cond_label[pixel, pixel_value, label]
                    label_prob *= pixel_prob
                probabilities[label] = label_prob

            labels_pred[i] = np.argmax(probabilities)

        return labels_pred

    def score(self, pixels, labels):
        labels_pred = self.predict(pixels)
        accuracy = np.mean(labels_pred == labels)
        return accuracy


if __name__ == '__main__':
    train_data = np.loadtxt('../data/train.csv', delimiter=',', dtype=np.uint8)
    test_data = np.loadtxt('../data/test.csv', delimiter=',', dtype=np.uint8)
    pixels_train, labels_train = train_data[:, :-1], train_data[:, -1]
    pixels_test, labels_test = test_data[:, :-1], test_data[:, -1]

    bn = BayesianNetwork()
    bn.fit(pixels_train, labels_train)
    test_score = bn.score(pixels_test, labels_test)
    print('test score: %.4f' % test_score)