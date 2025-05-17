import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from premier_league import MatchStatistics

MatchStatistics().create_dataset("data_set.csv")
data_set = pd.read_csv("dataset.csv")

def get_match_result(home_goals, away_goals):
    if home_goals > away_goals:
        return 2
    elif home_goals == away_goals:
        return 1
    else:
        return 0

data_set['result'] = data_set.apply(lambda row: get_match_result(row['home_goals'], row['away_goals']), axis=1)

# Calculate Dataset Distribution
labels = ['Home Wins', 'Draws', 'Away Wins']
ordered_indices = [2, 1, 0]
class_counts = data_set['result'].value_counts().reindex(ordered_indices, fill_value=0)

# Plot on Bar Graph
plt.figure(figsize=(12, 6))
plt.bar(class_counts.index, class_counts.values, alpha=0.7, color=['red', 'blue', 'green'])
plt.xticks(class_counts.index, labels)
plt.title("Distribution of Match Outcomes")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.savefig("match_outcome_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Set random seeds for reproducibility
np.random.seed(88)
torch.manual_seed(1293)

# Drop unnecessary Columns
targets = data_set['result'].values.astype(np.int64)
features = data_set.drop(['home_goals', 'away_goals', 'game_id', 'date', 'season', 'match_week',
                          'home_team', 'away_team', 'home_points', 'away_points', 'home_team_id', 'away_team_id', 'result'], axis=1)

n_features = features.shape[1]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features.values)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, targets, test_size=0.2, random_state=42)

# Apply SMOTE to Training Data to equalize the occurrence of classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

# Create DataLoader for batch training
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the classification model
class SoccerOutcomeClassifier(nn.Module):
    def __init__(self):
        super(SoccerOutcomeClassifier, self).__init__()
        self.fc0 = nn.Linear(n_features, 256)
        self.bn0 = nn.BatchNorm1d(256)
        self.dropout0 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.15)

        self.output = nn.Linear(32, 3)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu(self.bn0(self.fc0(x)))
        x = self.dropout0(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.output(x)
        return x


# Initialize model, loss, and optimizer
model = SoccerOutcomeClassifier()

# Calculate class weights
class_counts = Counter(y_train)
total_samples = len(y_train)
class_weights = {class_id: total_samples / (len(class_counts) * count) for class_id, count in class_counts.items()}
class_weights_tensor = torch.tensor([class_weights[0], class_weights[1], class_weights[2]], dtype=torch.float32)

# Apply weighted Cross-Entropy Loss
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.0005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training loop
n_epochs = 100
lambda_l1 = 0.0009
train_losses = []
validation_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0

    for inputs, targets in train_loader:
        # Forward pass
        predictions = model(inputs)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = criterion(predictions, targets) + lambda_l1 * l1_norm

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Calculate validation loss
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test_tensor)
        val_loss = criterion(val_predictions, y_test_tensor)
        validation_losses.append(val_loss.item())

    # Store train loss
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss.item():.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    # Get raw predictions (logits)
    test_predictions = model(X_test_tensor)

    # Apply softmax to get probabilities
    probabilities = torch.softmax(test_predictions, dim=1)

    # Get the predicted classes (highest probability index)
    predicted_classes = torch.argmax(probabilities, dim=1)

    # Calculate accuracy
    correct_predictions = (predicted_classes == y_test_tensor).float()
    accuracy = correct_predictions.mean().item()

    # Calculate Class Distribution
    for cls in [0, 1, 2]:
        cls_indices = (y_test_tensor == cls)
        correct = (predicted_classes[cls_indices] == y_test_tensor[cls_indices]).float()
        class_accuracy = correct.mean().item()
        print(f"Accuracy for class {cls}: {class_accuracy:.4f}")

    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Sample predictions vs actual
    print("\nSample predictions (first 5):")
    for i in range(5):
        actual = y_test_tensor[i].item()
        predicted = predicted_classes[i].item()
        probs = probabilities[i].numpy()
        print(f"Actual: {actual}, Predicted: {predicted}, Probabilities: {probs}")

# Save model
torch.save(model.state_dict(), "soccer_score_model.pth")
print("\nModel weights saved as 'soccer_score_model.pth'")
