import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import scipy.io as scio
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Enhanced Weighted Reconstruction Loss
class WeightedReconstructionLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha  # Controls the aggressiveness of deviation punishment

    def forward(self, reconstructed, original):
        # Ensure both tensors have the same dimensions
        # If reconstructed is smaller, resize original
        if reconstructed.shape != original.shape:
            # Resize original to match reconstructed using interpolation
            original = F.interpolate(original, size=reconstructed.shape[2:], mode='nearest')
        
        # Compute element-wise squared error
        base_loss = (reconstructed - original) ** 2
        
        # Add a power term to exponentially increase loss for larger deviations
        weighted_loss = base_loss * (1 + torch.abs(base_loss)) ** self.alpha
        
        return torch.mean(weighted_loss)

# Enhanced Normalization Function
def normalize(data):                    
    rawdata_max = max(map(max, data))
    rawdata_min = min(map(min, data))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = (data[i][j] - rawdata_min) / (rawdata_max - rawdata_min)
    return data

# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, root_dir, names_file, transform=None, background_only=False, sample_numbers=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            raise FileNotFoundError(f"{self.names_file} does not exist!")

        # Load all the file paths and labels from the names_file
        with open(self.names_file) as file:
            for i, line in enumerate(file):
                label = int(line.strip().split(' ')[1])
                if sample_numbers is not None and label not in sample_numbers:
                    continue
                # If background_only is True, only include samples with label 0
                if not background_only or label == 0:
                    self.names_list.append(line.strip())
                    self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_path = self.root_dir + self.names_list[idx].split(' ')[0]
        label = int(self.names_list[idx].split(' ')[1])

        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"{data_path} does not exist!")

        rawdata = scio.loadmat(data_path)['data']
        data = normalize(rawdata).astype(np.float32)  # Normalize data to [0, 1]
        data = torch.tensor(data).unsqueeze(0)

        sample = {'data': data, 'label': label}

        if self.transform:
            sample['data'] = self.transform(sample['data'])

        return sample

# Shallower Autoencoder Model
class ShallowerAutoencoder(nn.Module):
    def __init__(self):
        super(ShallowerAutoencoder, self).__init__()

        # Encoder with CNN-like structure
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=(200, 3), stride=(50, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, padding=1),  # Matches first CNN conv1

            nn.Conv2d(5, 10, kernel_size=(20, 2), stride=(4, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # Matches second CNN conv2
        )

        # Decoder mirrors the encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 5, kernel_size=(20, 2), stride=(4, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose2d(5, 1, kernel_size=(200, 3), stride=(50, 1), padding=1),
            nn.Sigmoid()  # Ensures output range matches the input range (e.g., normalized data)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Anomaly Detector Class
class AnomalyDetector:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ShallowerAutoencoder().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.threshold = None

    def train(self, train_loader, num_epochs=50, learning_rate=0.001):
        # Loss and optimizer
        criterion = WeightedReconstructionLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        self.model.train()
        training_losses = []

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                inputs = batch['data'].to(self.device)

                # Forward pass
                reconstructed = self.model(inputs)
                loss = criterion(reconstructed, inputs)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Calculate and log average loss for the epoch
            avg_loss = total_loss / len(train_loader)
            training_losses.append(avg_loss)

            # Step scheduler
            scheduler.step(avg_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        return training_losses

    def compute_threshold(self, validation_loader, percentile=97, std_multiplier=1.5):
        """Advanced threshold computation with reconstruction error"""
        reconstruction_errors = []

        self.model.eval()
        with torch.no_grad():
            for batch in validation_loader:
                inputs = batch['data'].to(self.device)
                reconstructed = self.model(inputs)

                # Ensure both tensors have the same dimensions
                if reconstructed.shape != inputs.shape:
                    # Resize inputs to match reconstructed using interpolation
                    inputs = F.interpolate(inputs, size=reconstructed.shape[2:], mode='nearest')
                
                # Compute reconstruction error with more nuanced calculation
                rec_error = torch.sqrt(torch.mean((reconstructed - inputs) ** 2, dim=(1, 2, 3)))
                reconstruction_errors.extend(rec_error.cpu().numpy())

        # Use both percentile and standard deviation for threshold
        base_threshold = np.percentile(reconstruction_errors, percentile)
        std_threshold = np.mean(reconstruction_errors) + std_multiplier * np.std(reconstruction_errors)
        
        self.threshold = min(base_threshold, std_threshold)
        return self.threshold

    def detect_anomalies(self, test_loader):
        self.model.eval()
        anomaly_results = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['data'].to(self.device)
                labels = batch['label']

                reconstructed = self.model(inputs)

                # Ensure both tensors have the same dimensions
                if reconstructed.shape != inputs.shape:
                    inputs = F.interpolate(inputs, size=reconstructed.shape[2:], mode='nearest')

                # Compute reconstruction error with additional metrics
                rec_error = torch.sqrt(torch.mean((reconstructed - inputs) ** 2, dim=(1, 2, 3)))
                
                # Additional anomaly scoring
                spatial_variance = torch.std(torch.abs(reconstructed - inputs), dim=(1, 2, 3))
                
                # Combined anomaly score
                anomaly_score = rec_error * (1 + spatial_variance)
                
                is_anomaly = anomaly_score > self.threshold

                anomaly_results.append({
                    'inputs': inputs.cpu(),
                    'labels': labels,
                    'is_anomaly': is_anomaly.cpu(),
                    'scores': anomaly_score.cpu()
                })

        return anomaly_results

    def save_model(self, path):
        """Saves the model's state dictionary"""
        torch.save(self.model.state_dict(), path)

    def evaluate(self, test_loader):
        """
        Evaluate the model's performance with multiple metrics
        Returns a dictionary containing various performance metrics
        """
        results = self.detect_anomalies(test_loader)
        
        # Collect predictions and labels
        predictions = torch.cat([r['is_anomaly'] for r in results]).numpy().astype(int)
        true_labels = torch.cat([r['labels'] for r in results]).numpy().astype(int)

        # For anomaly detection, convert multi-class to binary
        true_labels_binary = (true_labels > 0).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(true_labels_binary, predictions),
            'precision': precision_score(true_labels_binary, predictions, zero_division=0),
            'recall': recall_score(true_labels_binary, predictions, zero_division=0),
            'f1': f1_score(true_labels_binary, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(true_labels_binary, predictions)
        }

        # Calculate error statistics
        all_scores = torch.cat([r['scores'] for r in results])
        metrics['mean_error'] = torch.mean(all_scores).item()
        metrics['std_error'] = torch.std(all_scores).item()
        metrics['median_error'] = torch.median(all_scores).item()

        return metrics

    def plot_error_distribution(self, validation_loader, test_loader, threshold=None):
        """
        Plot distribution of reconstruction errors
        """
        # Collect reconstruction errors for normal (label 0) data
        normal_errors = self._collect_reconstruction_errors(validation_loader)

        # Collect reconstruction errors for anomaly (labels 1-5) data
        anomaly_errors = self._collect_reconstruction_errors(test_loader, include_anomalies=True)

        # Plot distributions
        plt.figure(figsize=(10, 6))
        plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal (Label 0)')
        plt.hist(anomaly_errors, bins=50, alpha=0.5, label='Anomalies (Labels 1-5)')

        if threshold is not None:
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, label=f"Threshold ({threshold:.2f})")

        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title("Distribution of Reconstruction Errors")
        plt.legend()
        plt.savefig("error_distribution.png")
        plt.close()

    def _collect_reconstruction_errors(self, loader, include_anomalies=False):
        self.model.eval()
        errors = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch['data'].to(self.device)
                labels = batch['label']
                
                # Filter data based on the include_anomalies flag
                if include_anomalies:
                    mask = (labels > 0) & (labels <= 5)  # Only consider anomalies (1-5)
                else:
                    mask = labels == 0  # Only consider normal data
                
                inputs = inputs[mask]
                if inputs.size(0) == 0:  # Skip if no relevant samples in the batch
                    continue
                
                outputs = self.model(inputs)
                
                # Ensure both tensors have the same dimensions
                if outputs.shape != inputs.shape:
                    inputs = F.interpolate(inputs, size=outputs.shape[2:], mode='nearest')
                
                batch_errors = torch.sqrt(torch.mean((outputs - inputs) ** 2, dim=(1, 2, 3)))
                errors.extend(batch_errors.cpu().numpy())
        
        return errors


    def print_evaluation_results(self, metrics):
        """
        Print evaluation metrics
        """
        print("\n=== Anomaly Detection Performance ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

        print("\n=== Reconstruction Error Statistics ===")
        print(f"Mean Error: {metrics['mean_error']:.4f}")
        print(f"Std Error: {metrics['std_error']:.4f}")
        print(f"Median Error: {metrics['median_error']:.4f}")

        print("\n=== Confusion Matrix ===")
        print(metrics['confusion_matrix'])

def main():
    # Hyperparameters
    batch_size = 16
    num_epochs = 15
    learning_rate = 0.001

    # Create datasets
    # Create datasets
    train_dataset = MyDataset(
        root_dir='das_data/train',
        names_file='das_data/train/label.txt',
        background_only=True
    )

    test_dataset = MyDataset(
        root_dir='das_data/test',
        names_file='das_data/test/filtered_label_test3auto.txt',
        background_only=False
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize and train anomaly detector
    detector = AnomalyDetector(model_path="anomaly_detector_shallow_normal3.pth")

    # Training or loading pre-trained model
    if os.path.exists("anomaly_detector_shallow_normal3.pth"):
        print("Loading pre-trained model...")
    else:
        print("Training on background data...")
        training_losses = detector.train(train_loader, num_epochs=num_epochs, learning_rate=learning_rate)
        detector.save_model("anomaly_detector_shallow.pth")

    # Compute adaptive threshold
    print("\nComputing threshold...")
    threshold = detector.compute_threshold(train_loader, percentile=95)
    print(f"Anomaly threshold: {threshold:.4f}")

    # Detect anomalies in test data
    print("\nDetecting anomalies...")
    anomaly_results = detector.detect_anomalies(test_loader)
    detector.plot_error_distribution(validation_loader=train_loader, test_loader=test_loader, threshold=threshold)

    print("\nEvaluating model performance...")
    metrics = detector.evaluate(test_loader)
    detector.print_evaluation_results(metrics)

    # Analyze anomalies
    def analyze_anomalies(anomaly_results):
        total_samples = 0
        anomaly_count = 0

        for result in anomaly_results:
            total_samples += len(result['labels'])
            anomaly_count += result['is_anomaly'].sum().item()

        print(f"\nTotal Samples: {total_samples}")
        print(f"Detected Anomalies: {anomaly_count}")
        print(f"Anomaly Percentage: {(anomaly_count/total_samples)*100:.2f}%")

    analyze_anomalies(anomaly_results)

if __name__ == "__main__":
    main()