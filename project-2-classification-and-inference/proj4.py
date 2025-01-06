import os
import csv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler

img_size = 300
classes = ['Alzheimer', 'COVID', 'Brazilian_seeds', 'Brazilian_leaves', 'skin_cancer']
data_dir = './image_dataset/'
case1_csv_filename = 'case1_statistics.csv'
case2_csv_filename = 'case2_statistics.csv'
reg_param = 1e-5  # Adjusted regularization parameter

def load_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    images = []

    for image_file in image_files:
        img = cv.imread(os.path.join(folder_path, image_file), cv.IMREAD_GRAYSCALE)
        ratio = img_size / img.shape[1]
        img_resized = cv.resize(img, (img_size, int(img.shape[0] * ratio)), cv.INTER_AREA)
        images.append(img_resized)

    return images

def makeHist(image, bits):
    hist, bins = np.histogram(image.flatten(), bins=np.arange(0, 257, 2 ** (8 - bits)))
    return hist, bins

def normalizeHist(hist):
    total_pixels = np.sum(hist)
    normalized_hist = hist / total_pixels
    return normalized_hist

def bin_centers(bin_borders):
    bin_center_list = (bin_borders[:-1] + bin_borders[1:]) / 2
    bin_center_list[-1] = 255
    return bin_center_list

def expectancy(hist, bin_centers):
    return np.sum(hist * bin_centers)

def median(hist, bin_centers):
    cumulative_freq = 0
    for i, freq in enumerate(hist):
        cumulative_freq += freq
        if cumulative_freq >= 0.5:
            return bin_centers[i]

def mode(hist, bin_centers):
    return bin_centers[np.argmax(hist)]

def moment(hist, bin_centers, order, expectancy):
    return np.sum(((bin_centers - expectancy) ** order) * hist)

def entropy(hist):
    epsilon = 1e-27
    return -np.sum(hist * np.log2(hist + epsilon))

def calculate_statistics(image, bins):
    hist, bin_edges = makeHist(image, bins)
    normalized_hist = normalizeHist(hist)
    centered_bins = bin_centers(bin_edges)

    var_exp = expectancy(normalized_hist, centered_bins)
    var_median = median(normalized_hist, centered_bins)
    var_mode = mode(normalized_hist, centered_bins)
    var_variance = moment(normalized_hist, centered_bins, 2, var_exp)
    var_skewness = moment(normalized_hist, centered_bins, 3, var_exp)
    var_kurtosis = moment(normalized_hist, centered_bins, 4, var_exp)
    var_entropy = entropy(normalized_hist)

    return hist, var_exp, var_mode, var_median, var_variance, var_skewness, var_kurtosis, var_entropy

def save_statistics_to_csv(images, class_label, case1_csv_filename, case2_csv_filename):
    with open(case1_csv_filename, mode='a', newline='') as case1_file, open(case2_csv_filename, mode='a', newline='') as case2_file:
        case1_writer = csv.writer(case1_file)
        case2_writer = csv.writer(case2_file)

        for image in images:
            hist, var_exp, var_mode, var_median, var_variance, var_skewness, var_kurtosis, var_entropy = calculate_statistics(image, bins=8)

            case_1_data = list(hist) + [class_label]
            case_2_data = [var_exp, var_mode, var_median, var_variance, var_skewness, var_kurtosis, var_entropy, class_label]

            case1_writer.writerow(case_1_data)
            case2_writer.writerow(case_2_data)

def load_data_from_csv(filename):
    data = []
    labels = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append([float(x) for x in row[:-1]])
            labels.append(int(row[-1]))
    return np.array(data), np.array(labels)

def train_test_split(data, labels, test_size=0.1):
    np.random.seed(np.random.randint(0,99))
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(len(data) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return data[train_indices], data[test_indices], labels[train_indices], labels[test_indices]

with open(case1_csv_filename, mode='w', newline='') as case1_file, open(case2_csv_filename, mode='w', newline='') as case2_file:
    case1_writer = csv.writer(case1_file)
    case2_writer = csv.writer(case2_file)

    case1_writer.writerow([f'h[{i}]' for i in range(8)] + ['class'])
    case2_writer.writerow(['expectancy', 'mode', 'median', 'variance', 'skewness', 'kurtosis', 'entropy', 'class'])

for class_id, class_name in enumerate(classes):
    folder_path = os.path.join(data_dir, class_name)
    images = load_images_from_folder(folder_path)
    save_statistics_to_csv(images, class_id, case1_csv_filename, case2_csv_filename)

data_case1, labels_case1 = load_data_from_csv(case1_csv_filename)
data_case2, labels_case2 = load_data_from_csv(case2_csv_filename)

X_train_case1, X_test_case1, y_train_case1, y_test_case1 = train_test_split(data_case1, labels_case1, test_size=0.1)
X_train_case2, X_test_case2, y_train_case2, y_test_case2 = train_test_split(data_case2, labels_case2, test_size=0.1)

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for cls in self.classes:
            X_c = X[y == cls]
            self.mean[cls] = np.mean(X_c, axis=0)
            self.var[cls] = np.var(X_c, axis=0) + reg_param
            self.priors[cls] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        posteriors = []

        for cls in self.classes:
            prior = np.log(self.priors[cls])
            posterior = np.sum(np.log(self._pdf(cls, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, cls, x):
        mean = self.mean[cls]
        var = self.var[cls]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        results = numerator / denominator

        for i in range(len(results)):
            if results[i] == 0:
                results[i] = reg_param

        return results

class QuadraticDiscriminantAnalysis:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.covariance = {}
        self.priors = {}

        for cls in self.classes:
            X_c = X[y == cls]
            self.mean[cls] = np.mean(X_c, axis=0)
            self.covariance[cls] = np.cov(X_c, rowvar=False) + np.eye(X.shape[1]) * reg_param
            self.priors[cls] = X_c.shape[0] / X.shape[0]

        #print("Means:", self.mean)
        #print("Covariances:", self.covariance)
        #print("Priors:", self.priors)

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        discriminants = []

        for cls in self.classes:
            G = self._quadratic_discriminant(cls, x)
            discriminants.append(G)

        return self.classes[np.argmax(discriminants)]

    def _quadratic_discriminant(self, cls, x):
        mean = self.mean[cls]
        covariance = self.covariance[cls]
        inv_covariance = np.linalg.inv(covariance)
        det_cov = np.linalg.det(covariance)

        if det_cov == 0:
            det_cov = reg_param

        W_k = -0.5 * inv_covariance
        w_k = np.dot(inv_covariance, mean)
        w_k0 = -0.5 * np.dot(mean.T, np.dot(inv_covariance, mean)) - 0.5 * np.log(det_cov) + np.log(self.priors[cls])

        discriminant = np.dot(x.T, np.dot(W_k, x)) + np.dot(w_k.T, x) + w_k0

        return discriminant

class LinearDiscriminantAnalysis:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.covariance = {}
        self.priors = {}

        for cls in self.classes:
            X_c = X[y == cls]
            self.mean[cls] = np.mean(X_c, axis=0)
            self.covariance[cls] = np.cov(X_c, rowvar=False) + np.eye(X.shape[1]) * reg_param
            self.priors[cls] = X_c.shape[0] / X.shape[0]

        #print("Means:", self.mean)
        #print("Covariances:", self.covariance)
        #print("Priors:", self.priors)

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        discriminants = []

        for cls in self.classes:
            G = self._linear_discriminant(cls, x)
            discriminants.append(G)

        return self.classes[np.argmax(discriminants)]

    def _linear_discriminant(self, cls, x):
        mean = self.mean[cls]
        covariance = self.covariance[cls]
        inv_covariance = np.linalg.inv(covariance)

        w_k = np.dot(inv_covariance, mean)
        w_k0 = -0.5 * np.dot(mean.T, np.dot(inv_covariance, mean)) + np.log(self.priors[cls])

        discriminant = np.dot(w_k.T, x) + w_k0

        return discriminant

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    return cm

def accuracy_per_class(cm):
    return np.diag(cm) / np.sum(cm, axis=1)

def overall_accuracy(cm):
    return np.sum(np.diag(cm)) / np.sum(cm)

data, labels = load_data_from_csv(case1_csv_filename)

X_train, X_test, y_train, y_test = train_test_split(data, labels)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb, len(classes))
acc_nb = accuracy_per_class(cm_nb)
overall_acc_nb = overall_accuracy(cm_nb)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)
cm_qda = confusion_matrix(y_test, y_pred_qda, len(classes))
acc_qda = accuracy_per_class(cm_qda)
overall_acc_qda = overall_accuracy(cm_qda)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
cm_lda = confusion_matrix(y_test, y_pred_lda, len(classes))
acc_lda = accuracy_per_class(cm_lda)
overall_acc_lda = overall_accuracy(cm_lda)

print("\n--------------------CASE1--------------------\n")

print("Bayes Confusion Matrix:\n", cm_nb)
print("Bayes Accuracy per Class:\n", acc_nb)
print("Bayes Accuracy:", overall_acc_nb)

print("\n----------------------------------------\n")

print("QDA Confusion Matrix:\n", cm_qda)
print("QDA Accuracy per Class:\n", acc_qda)
print("QDA Accuracy:", overall_acc_qda)

print("\n----------------------------------------\n")

print("LDA Confusion Matrix:\n", cm_lda)
print("LDA Accuracy per Class:\n", acc_lda)
print("LDA Accuracy:", overall_acc_lda)

data, labels = load_data_from_csv(case2_csv_filename)

X_train, X_test, y_train, y_test = train_test_split(data, labels)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb, len(classes))
acc_nb = accuracy_per_class(cm_nb)
overall_acc_nb = overall_accuracy(cm_nb)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)
cm_qda = confusion_matrix(y_test, y_pred_qda, len(classes))
acc_qda = accuracy_per_class(cm_qda)
overall_acc_qda = overall_accuracy(cm_qda)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
cm_lda = confusion_matrix(y_test, y_pred_lda, len(classes))
acc_lda = accuracy_per_class(cm_lda)
overall_acc_lda = overall_accuracy(cm_lda)

print("\n--------------------CASE2--------------------\n")

print("Bayes Confusion Matrix:\n", cm_nb)
print("Bayes Accuracy per Class:\n", acc_nb)
print("Bayes Accuracy:", overall_acc_nb)

print("\n----------------------------------------\n")

print("QDA Confusion Matrix:\n", cm_qda)
print("QDA Accuracy per Class:\n", acc_qda)
print("QDA Accuracy:", overall_acc_qda)

print("\n----------------------------------------\n")

print("LDA Confusion Matrix:\n", cm_lda)
print("LDA Accuracy per Class:\n", acc_lda)
print("LDA Accuracy:", overall_acc_lda)