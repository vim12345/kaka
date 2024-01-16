# Import necessary libraries
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from model import create_model  # Import a function to create your custom model

# Define a function to train the model
def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # Check if GPU is available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets
    image_datasets = {x: datasets.ImageFolder(data_dir + '/' + x, transform=data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid']}

    # Create the model
    model = create_model(arch, hidden_units)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        validation_loss = 0.0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                validation_loss += criterion(outputs, labels)
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Training loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

    # Save the model as a checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_dir)

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to the dataset directory")
    parser.add_argument("--save_dir", default="checkpoint.pth", help="Directory to save the model checkpoint")
    parser.add_argument("--arch", default="vgg16", help="Model architecture (e.g., 'vgg16', 'resnet50')")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    # Call the train_model function
    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)