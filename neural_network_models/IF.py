# coding = UTF-8
import sys
import datetime
from sklearn.ensemble import IsolationForest
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

sys.stdout = Logger('iforest_result.log', sys.stdout)

# Paths
rootpath = 'das_data'
train_rootpath = rootpath+'/train'
train_labelpath = rootpath+'/train/label.txt'
test_rootpath = rootpath+'/test'
test_labelpath = rootpath+'/test/label.txt'
# das_data/test/filtered_label_test.txt
# Load and preprocess data
start_train = datetime.datetime.now()
X_train, y_train = get_das_data(train_rootpath, train_labelpath)
X_test, y_test = get_das_data(test_rootpath, test_labelpath)

# Separate normal (class 0) data for training
normal_idx = y_train == 0
X_train_normal = X_train[normal_idx]

# Scale the data
minMaxScaler = preprocessing.MinMaxScaler()
X_train_normal_scaled = minMaxScaler.fit_transform(X_train_normal)
testData = minMaxScaler.transform(X_test)

# Save feature data
pre_y_test = y_test[:, np.newaxis]
feature_data = np.concatenate((testData, pre_y_test), axis=1)
np.savetxt('5km_10km_iforest_feature_data.csv', feature_data, delimiter=',')

# Train Isolation Forest
clf = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination='auto',
    random_state=42
)

clf.fit(X_train_normal_scaled)
end_train = datetime.datetime.now()

# Predict
# IsolationForest returns: 1 for normal data, -1 for anomalies
# Convert to: 0 for normal (background), 1 for anomalies
start_test = datetime.datetime.now()
test_result = clf.predict(testData)
test_result = (test_result == -1).astype(int)
end_test = datetime.datetime.now()

# Convert original labels to binary
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
plt.savefig('./IForest_confusion_matrix.jpg')
plt.show()

# Calculate metrics
TP = C[1][1]  # True Positives
TN = C[0][0]  # True Negatives
FP = C[0][1]  # False Positives
FN = C[1][0]  # False Negatives

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
FAR = FP / (TN + FP) if (TN + FP) > 0 else 0  # False Alarm Rate
MDR = FN / (TP + FN) if (TP + FN) > 0 else 0  # Miss Detection Rate

print('\nPerformance Metrics:')
print('Accuracy: %.4f' % Accuracy)
print('Precision: %.4f' % Precision)
print('Recall: %.4f' % Recall)
print('F1-Score: %.4f' % F1)
print('False Alarm Rate: %.4f' % FAR)
print('Miss Detection Rate: %.4f' % MDR)


# /home/nemezjusz/inzynierka/get_das_data.py:8: RuntimeWarning: invalid value encountered in cast
#   data_diff = np.empty([m-1, n]).astype(int)   # 9999,12
# Test Confusion Matrix:
#  [[563  25]
#  [ 17  58]]
# Train time: 1:52:16.893746
# Test time: 0:00:00.011740
# /home/nemezjusz/inzynierka/IF.py:98: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
#   plt.show()

# Performance Metrics:
# Accuracy: 0.9367
# Precision: 0.6988
# Recall: 0.7733
# F1-Score: 0.7342
# False Alarm Rate: 0.0425
# Miss Detection Rate: 0.2267
# nemezjusz@SecretService:~/inzynierka$ python3 IF.py 
# /home/nemezjusz/inzynierka/get_das_data.py:8: RuntimeWarning: invalid value encountered in cast
#   data_diff = np.empty([m-1, n]).astype(int)   # 9999,12
# Test Confusion Matrix:
#  [[ 562   25]
#  [ 569 1926]]
# Train time: 1:51:43.201209
# Test time: 0:00:00.022874
# /home/nemezjusz/inzynierka/IF.py:98: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
#   plt.show()

# Performance Metrics:
# Accuracy: 0.8073
# Precision: 0.9872
# Recall: 0.7719
# F1-Score: 0.8664
# False Alarm Rate: 0.0426
# Miss Detection Rate: 0.2281