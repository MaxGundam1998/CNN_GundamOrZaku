import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.optim import Adam

#Input the image yourself, and tell the computer if it's right or wrong. Then perform readjustments 

#Check if cuda is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The device is", device)

class ConvNet(nn.Module):
    def __init__(self, num_classes = 2):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        #Shape = (256, 12, 150, 150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        #Shape = (256, 12, 150, 150)
        self.relu1 = nn.ReLU()
        #Shape = (256, 12, 150, 150)

        self.pool = nn.MaxPool2d(kernel_size=2)
        #Reduce Image size by factor 2
        #Shape = (256, 12, 75, 75)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        #Shape = (256, 20, 75, 75)
        self.relu2 = nn.ReLU()
        #Shape = (256, 20, 75, 75)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        #Shape = (256, 32, 75, 75)
        self.bn3 = nn.BatchNorm2d(num_features = 32)
        #Shape = (256, 32, 75, 75)
        self.relu3=nn.ReLU()
        #Shape = (256, 32, 75, 75)

        self.fc = nn.Linear(in_features=32*75*75, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)
            
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        #Above output will be in matrix form

        output = output.view(-1, 32*75*75)

        output = self.fc(output)

        return output

#Load the saved model

model_path = 'best_mobilepoint.model'
model = ConvNet()
model.load_state_dict(torch.load(model_path))
model.eval()

optimizer = Adam(model.parameters(), lr = 0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

#Transform
image_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

#Input the images path

while True:
    user_input = input("Enter an Image Path or enter quit to exit: ")
    if user_input == "quit":
        break

    try:
        image = Image.open(user_input)
        image = image_transforms(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image)
        
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()

        class_names = ['Gundam', 'Zaku']
        predicted_class_name = class_names[predicted_class]

        print("Predicted class:", predicted_class_name)

        #After we've predicted the value, tell the CNN if it's correct or not. 

        valid_input = False

        while valid_input == False:
            correct_class = input("Enter the correct class label (Gundam/Zaku): ")

            if correct_class.lower() == predicted_class_name.lower():
                print("The prediction is correct!")
                valid_input = True

            elif correct_class.lower() in [class_name.lower() for class_name in class_names]:
            # Perform backpropagation and update the model
                print("Reworking Model")
                target = torch.tensor([class_names.index(correct_class)])
                output.requires_grad = True 
                loss = loss_function(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("Model retrained")
                valid_input = True

                torch.save(model.state_dict(), 'best_mobilepoint.model')

            else:
                print("Invalid label input")




    
    except FileNotFoundError:
        print("Invalid image path. Please Try again.")

    except Exception as e:
        print("Error:", str(e))