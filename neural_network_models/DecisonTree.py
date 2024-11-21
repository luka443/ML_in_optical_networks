import numpy as np
import os
import scipy.io as scio
import joblib
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader

def normalize(data):
    rawdata_max = max(map(max, data))
    rawdata_min = min(map(min, data))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = round(((255 - 0) * (data[i][j] - rawdata_min) / (rawdata_max - rawdata_min)) + 0)
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

            data = data.reshape(1, -1)  # make it 1 x 120000

        X.append(data)
        y.append(label)


    X = np.vstack(X)
    y = np.array(y)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y

def train_decision_tree(X_train, X_test, y_train, y_test, max_depth=None):

    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt_classifier.fit(X_train, y_train)

    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return dt_classifier, accuracy, X_test, y_test, y_pred

def save_model(model, accuracy, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"decision_tree_{timestamp}_acc_{accuracy:.4f}.joblib"
    filepath = os.path.join(save_dir, filename)
    joblib.dump(model, filepath)
    return filepath

def main():

    root_dir = "drive/MyDrive/inzynierka/das_data2/train"
    names_file = "/content/drive/MyDrive/inzynierka/das_data2/train/label.txt"

    root_dir_test = "drive/MyDrive/inzynierka/das_data2/test"
    names_file_test = "/content/drive/MyDrive/inzynierka/das_data2/test/label.txt"

    # Create dataset
    dataset = MyDataset(root_dir=root_dir, names_file=names_file)
    dataset_test = MyDataset(root_dir=root_dir_test, names_file=names_file_test)
    # Prepare data
    X_train, y_train = prepare_data(dataset)
    X_test, y_test = prepare_data(dataset_test)


    dt_classifier, accuracy, X_test, y_test, y_pred = train_decision_tree(X_train, X_test,y_train,y_test, max_depth=16)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    saved_model_path = save_model(dt_classifier, accuracy)
    print(f"\nModel saved to: {saved_model_path}")

if __name__ == "__main__":
    main()