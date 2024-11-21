import numpy as np
import os
import scipy.io as scio
import joblib
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader

def normalize(data):                    
    rawdata_max = max(map(max, data))
    rawdata_min = min(map(min, data))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = (data[i][j] - rawdata_min) / (rawdata_max - rawdata_min)
    return data

class MyDataset(Dataset):
    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + ' does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f.strip())
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_path = self.root_dir + self.names_list[idx].split(' ')[0]
        if not os.path.isfile(data_path):
            print(data_path + ' does not exist!')
            return None
        rawdata = scio.loadmat(data_path)['data']
        rawdata = rawdata.astype(int)
        data = normalize(rawdata)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

def prepare_data(dataset):
    X = []
    y = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for sample in dataloader:
        data = sample['data'].squeeze().numpy()
        label = sample['label'].item()

        if len(data.shape) > 1:
            data = data.reshape(1, -1)

        X.append(data)
        y.append(label)

    X = np.vstack(X)
    y = np.array(y)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y

def train_isolation_forest(X_train, y_train, X_test, y_test, contamination=0.11, n_estimators=100, random_state=42):
    
    # Initialize and train Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    iso_forest.fit(X_train)

    # Predict anomalies
    y_pred_train = iso_forest.predict(X_train)
    y_pred_test = iso_forest.predict(X_test)

    # Convert predictions to binary format (0 for outliers, 1 for inliers)
    y_pred_train = (y_pred_train == -1).astype(int)
    y_pred_test = (y_pred_test == -1).astype(int)
    print(y_pred_test)

    # Calculate anomaly scores
    anomaly_scores = iso_forest.score_samples(X_test)

    return iso_forest, y_pred_test, y_test, anomaly_scores


def save_model(model, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"isolation_forest_{timestamp}.joblib"
    filepath = os.path.join(save_dir, filename)
    joblib.dump(model, filepath)
    return filepath

def evaluate_results(y_true, y_pred, anomaly_scores):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)

    # Print additional metrics
    print("\nAnomaly Score Statistics:")
    print(f"Mean anomaly score: {np.mean(anomaly_scores):.4f}")
    print(f"Min anomaly score: {np.min(anomaly_scores):.4f}")
    print(f"Max anomaly score: {np.max(anomaly_scores):.4f}")


# Set your paths
root_dir = "drive/MyDrive/inzynierka/das_data2/train"
names_file = "/content/drive/MyDrive/inzynierka/das_data2/train/filtered_label2.txt"

root_di_test = "drive/MyDrive/inzynierka/das_data2/test"
names_file_test = "/content/drive/MyDrive/inzynierka/das_data2/test/filtered_label_test3.txt"

# Create dataset
dataset = MyDataset(root_dir=root_dir, names_file=names_file)
dataset_test = MyDataset(root_dir=root_dir, names_file=names_file)
# Prepare data
X_train, y_train = prepare_data(dataset)
X_test, y_test = prepare_data(dataset_test)

# Train Isolation Forest
iso_forest, y_pred, y_test, anomaly_scores = train_isolation_forest(
  X_train,
  y_train,
  X_test,
  y_test,
  contamination=0.11,  # Adjust based on expected anomaly ratio
  n_estimators=200)

# Evaluate results
evaluate_results(y_test, y_pred, anomaly_scores)

# Save the model
saved_model_path = save_model(iso_forest)
print(f"\nModel saved to: {saved_model_path}")

