# Import necessary libraries
import argparse
import torch
import json
from PIL import Image
from torchvision import transforms
from model import load_checkpoint  # Import a function to load the model checkpoint

# Define a custom classifier for your model
# Define the available architectures and their input sizes
architectures = {
    'vgg16': 25088,
    'densenet121': 1024,
}

#Create an argument parser
parser = argparse.ArgumentParser(description="Train a neural network")
parser.add_argument("data_dir", help="Path to the data directory")
parser.add_argument("--arch", type=str, default="vgg16", choices=architectures.keys(), help="Choose an architecture (vgg16 or densenet121)")

args = parser.parse_args()

#Use the chosen architecture to set the input size
input_size = architectures[args.arch]

# Load the selected model
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    # Customize classifier for vgg16
    # VGG's classifier consists of three fully connected layers

    my_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()), # ReLU activation
        ('drop1', nn.Dropout(0.5)), #Drpout Layer
        ('fc2', nn.Linear(4096, 1024)), 
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(0.5)),
        ('fc3', nn.Linear(1024, 102)),  
        ('output', nn.LogSoftmax(dim=1)) # LogSoftmax activation for output
    ]))

    model.classifier = my_classifier

elif args.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
     # Customize classifier for densenet121
    # Densenet's classifier is a single fully connected layer
    my_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, 102)),  
        ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = my_classifier

# Define a function to process an image
def process_image(image):
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transformations(image)
    return image

# Define a function to predict the class for an image
def predict_class(image, model, topk, idx_mapping, device):
    pre_processed_image = preprocess_image(image).to(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        log_ps = model(pre_processed_image)
        ps = torch.exp(log_ps)
        top_ps, top_idx = ps.topk(topk, dim=1)
        list_ps = top_ps.tolist()[0]
        list_idx = top_idx.tolist()[0]
        classes = [idx_mapping[x] for x in list_idx]
    
    model.train()
    return list_ps, classes

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("checkpoint", help="Path to the model checkpoint")
    parser.add_argument("--topk", type=int, default=5, help="Top K most likely classes")
    parser.add_argument("--category_names", default="cat_to_name.json", help="Mapping of category names to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()

    # Load category names mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load the model checkpoint
    model, class_to_idx = load_checkpoint(args.checkpoint)  # Modify load_checkpoint to return class_to_idx
    model.to("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Make predictions
    probabilities, class_labels = predict(args.image_path, model, args.topk, class_to_idx, "cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Print the top K predicted class labels and their probabilities
    for label, prob in zip(class_labels, probabilities):
        print(f"Class: {label}, Probability: {prob:.4f}")
