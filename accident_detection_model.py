import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import time

# Define constants
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")    

# Custom dataset for loading images
class AccidentImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        if self.is_test:
            return img
        else:
            return img, self.labels[idx]

# Define the model architecture
class AccidentDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(AccidentDetectionModel, self).__init__()
        
        # Load a pre-trained CNN as the base model (EfficientNet for high accuracy)
        self.model = models.efficientnet_b3(weights='DEFAULT')
        
        # Freeze early layers to prevent overfitting
        for param in list(self.model.parameters())[:-30]:
            param.requires_grad = False
        
        # Replace the classifier
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Function to prepare dataloaders from image directories
def prepare_dataloaders(train_dir, val_dir, test_dir):
    # Define image transformations with data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation and test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare data paths and labels
    def prepare_data(data_dir):
        accident_dir = os.path.join(data_dir, 'accident')
        non_accident_dir = os.path.join(data_dir, 'not_accident')
        
        # Get image paths
        accident_paths = [os.path.join(accident_dir, f) for f in os.listdir(accident_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        non_accident_paths = [os.path.join(non_accident_dir, f) for f in os.listdir(non_accident_dir) 
                              if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        paths = accident_paths + non_accident_paths
        labels = [1] * len(accident_paths) + [0] * len(non_accident_paths)
        
        return paths, labels
    
    train_paths, train_labels = prepare_data(train_dir)
    val_paths, val_labels = prepare_data(val_dir)
    test_paths, test_labels = prepare_data(test_dir)
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    # Create datasets
    train_dataset = AccidentImageDataset(train_paths, train_labels, train_transform)
    val_dataset = AccidentImageDataset(val_paths, val_labels, val_transform)
    test_dataset = AccidentImageDataset(test_paths, test_labels, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, val_transform

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        # Update scheduler with validation loss
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_accident_detection_model.pth')
            print("Saved best model!")
    
    print(f'Best val Acc: {best_val_acc:.4f}')
    return model, history

# Function to evaluate model on test set
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    report = classification_report(all_labels, all_preds, target_names=['Non-Accident', 'Accident'])
    print(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1], ['Non-Accident', 'Accident'])
    plt.yticks([0, 1], ['Non-Accident', 'Accident'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return report

# Function to process video frame by frame
def analyze_video(model, video_path, transform, window_size=5, overlap=2, threshold=0.6):
    model.eval()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output video
    out_path = video_path.rsplit('.', 1)[0] + '_analyzed.' + video_path.rsplit('.', 1)[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    accident_frames = []
    recent_predictions = []  # For temporal smoothing
    
    print(f"Processing {total_frames} frames...")
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Preprocess and get prediction
        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            accident_prob = probs[0][1].item()  # Probability of accident class
            recent_predictions.append(accident_prob)
        
        # Temporal smoothing (moving average)
        if len(recent_predictions) > window_size:
            recent_predictions.pop(0)
        smoothed_prob = sum(recent_predictions) / len(recent_predictions)
        
        # Detect accident
        is_accident = smoothed_prob > threshold
        
        if is_accident:
            accident_frames.append(frame_count)
            # Add visual indicator (red border)
            cv2.putText(frame, f"ACCIDENT DETECTED! {smoothed_prob:.2f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 5)
        else:
            # Add confidence score
            cv2.putText(frame, f"No accident: {smoothed_prob:.2f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add frame number and timestamp
        timestamp = frame_count / fps
        cv2.putText(frame, f"Frame: {frame_count} | Time: {timestamp:.2f}s", 
                    (50, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
    
    end_time = time.time()
    print(f"Video processing complete! Total time: {end_time - start_time:.2f} seconds")
    print(f"Processed at {frame_count / (end_time - start_time):.2f} FPS")
    
    cap.release()
    out.release()
    
    # Return statistics
    total_time = total_frames / fps
    accident_seconds = len(set([int(frame / fps) for frame in accident_frames]))
    
    print(f"Video duration: {total_time:.2f} seconds")
    print(f"Accident detected in {accident_seconds} seconds")
    print(f"Output saved to: {out_path}")
    
    return {
        'total_frames': total_frames,
        'total_time': total_time,
        'accident_frames': accident_frames,
        'accident_seconds': accident_seconds,
        'output_path': out_path
    }

# Real-time video analysis (webcam or video file)
def analyze_video_realtime(model, video_path=None, transform=None, threshold=0.6):
    model.eval()
    
    # Open webcam if no video path provided
    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # For temporal smoothing
    recent_predictions = []
    window_size = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video or webcam disconnected
            break
        
        # Process frame
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            accident_prob = probs[0][1].item()
            recent_predictions.append(accident_prob)
        
        # Temporal smoothing
        if len(recent_predictions) > window_size:
            recent_predictions.pop(0)
        smoothed_prob = sum(recent_predictions) / len(recent_predictions)
        
        # Visualize result
        is_accident = smoothed_prob > threshold
        height, width = frame.shape[:2]
        
        if is_accident:
            cv2.putText(frame, f"ACCIDENT DETECTED! {smoothed_prob:.2f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 5)
        else:
            cv2.putText(frame, f"No accident: {smoothed_prob:.2f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display processed frame
        cv2.imshow('Accident Detection', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function to train and evaluate the model
def main():
    # Paths to your dataset directories
    train_dir = 'Data/train'
    val_dir = 'Data/valid'
    test_dir = 'Data/test'
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader, transform = prepare_dataloaders(train_dir, val_dir, test_dir)
    
    # Initialize model
    model = AccidentDetectionModel().to(DEVICE)
    
    # Define loss function, optimizer, and learning rate scheduler
    # Use weighted loss if dataset is imbalanced
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Fixed scheduler to use StepLR instead of ReduceLROnPlateau to avoid the error
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    print("Starting training...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS)
    
    # Load the best model
    model.load_state_dict(torch.load('best_accident_detection_model.pth'))
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    evaluate_model(model, test_loader)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Training and evaluation completed!")
    
    return model, transform

# Run video analysis on a test video
def test_on_video(model_path, video_path):
    # Load the model
    model = AccidentDetectionModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Run video analysis
    print(f"Analyzing video: {video_path}")
    stats = analyze_video(model, video_path, transform, threshold=0.6)
    
    # Or for real-time analysis:
    # analyze_video_realtime(model, video_path, transform)
    
    return stats

if __name__ == "__main__":
    # Train and evaluate the model
    model, transform = main()
    
    # Test on a video (uncomment and specify path to your video)
    video_path = "videoplayback (online-video-cutter.com).mp4"
    test_on_video('best_accident_detection_model.pth', video_path)
    
    # For real-time analysis (webcam)
    # analyze_video_realtime(model, transform=transform)