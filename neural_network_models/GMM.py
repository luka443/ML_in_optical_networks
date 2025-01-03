# coding = UTF-8
import sys
import datetime
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from get_das_data import get_das_data
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

sys.stdout = Logger('gmm_result.log', sys.stdout)

# Paths
rootpath = 'das_data'
train_rootpath = rootpath+'/train'
train_labelpath = rootpath+'/train/label.txt'
test_rootpath = rootpath+'/test'
test_labelpath = rootpath+'/test/label.txt'

# Load and preprocess data
start_train = datetime.datetime.now()
X_train, y_train = get_das_data(train_rootpath, train_labelpath)
X_test, y_test = get_das_data(test_rootpath, test_labelpath)

# Separate normal data
normal_idx = y_train == 0
X_train_normal = X_train[normal_idx]

# Scale data
minMaxScaler = preprocessing.MinMaxScaler()
X_train_normal_scaled = minMaxScaler.fit_transform(X_train_normal)
testData = minMaxScaler.transform(X_test)

# Save feature data
pre_y_test = y_test[:, np.newaxis]
feature_data = np.concatenate((testData, pre_y_test), axis=1)
np.savetxt('6class_gmm_feature_data.csv', feature_data, delimiter=',')

# Train GMM
clf = GaussianMixture(
    n_components=5,  # Increased from 3
    covariance_type='full',
    random_state=42,
    n_init=10,  # Increased from 5
    max_iter=200,
    warm_start=True
)

clf.fit(X_train_normal_scaled)
end_train = datetime.datetime.now()

# Predict anomalies
start_test = datetime.datetime.now()
scores = -clf.score_samples(testData)  # Negative log-likelihood
threshold = np.percentile(scores, 20)  # Adjust percentile as needed
test_result = (scores > threshold).astype(int)
end_test = datetime.datetime.now()

# Convert labels to binary
y_test_binary = (y_test != 0).astype(int)

# Calculate confusion matrix
test_matrix = confusion_matrix(y_test_binary, test_result)

# Print results
print('Test Confusion Matrix:\n', test_matrix)
print('Train time:', end_train - start_train)
print('Test time:', end_test - start_test)

# Visualize confusion matrix
C = test_matrix
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
df = pd.DataFrame(C)
sns.heatmap(df, fmt='g', annot=True, robust=True,
            annot_kws={'size': 10},
            xticklabels=['Background', 'Activity'],
            yticklabels=['Background', 'Activity'],
            cmap='Blues')
ax.set_xlabel('Predicted label', fontsize=15)
ax.set_ylabel('True label', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.savefig('./GMM_confusion_matrix.jpg')
plt.show()

# Calculate metrics
TP = C[1][1]
TN = C[0][0]
FP = C[0][1]
FN = C[1][0]

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
FAR = FP / (TN + FP) if (TN + FP) > 0 else 0
MDR = FN / (TP + FN) if (TP + FN) > 0 else 0

print('\nPerformance Metrics:')
print('Accuracy: %.4f' % Accuracy)
print('Precision: %.4f' % Precision)
print('Recall: %.4f' % Recall)
print('F1-Score: %.4f' % F1)
print('False Alarm Rate: %.4f' % FAR)
print('Miss Detection Rate: %.4f' % MDR)

# Test Confusion Matrix:
#  [[ 559   28]
#  [  58 2437]]
# Train time: 1:59:44.251231
# Test time: 0:00:00.066977
# /home/nemezjusz/inzynierka/gaussianMix.py:99: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
#   plt.show()

# Performance Metrics:
# Accuracy: 0.9721
# Precision: 0.9886
# Recall: 0.9768
# F1-Score: 0.9827
# False Alarm Rate: 0.0477
# Miss Detection Rate: 0.0232