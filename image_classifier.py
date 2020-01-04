# from: https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
# from: https://towardsdatascience.com/a-beginners-tutorial-on-building-an-ai-image-classifier-using-pytorch-6f85cb69cba7
import math
import os
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from matplotlib import rcParams
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
from datetime import datetime, timedelta
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = './data'
START_TIME = datetime.now()

# Normalizing all the images
#   transforms.Compose lets us compose multiple transforms together so we can use more than one transformation
#   transforms.Resize((255)) resizes the images so the shortest side has a length of 255 pixels. The other side is scaled to maintain the aspect ratio of the image
#   transforms.CenterCrop(224) crops the center of the image so it is a 224 by 224 pixels square image
#   transforms.ToTensor() converts our image into numbers
#     It separates the three colors that every pixel of our picture is comprised of: red, green & blue.
#     This essentially turns one image into three images (one tinted red, one green, one blue).
#     Then, it converts the pixels of each tinted image into the brightness of their color, from 0 to 255.
#     These values are divided by 255, so they can be in a range of 0 to 1. Our image is now a Torch Tensor (a data structure that stores lots of numbers).
#   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) subtracts the mean from each value and then divides by the standard deviation.
TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def print_status(explanation):
    current_time = datetime.now()
    time_elapsed = current_time - START_TIME
    total_seconds = time_elapsed.total_seconds()
    hours = int(total_seconds / 3600.0)
    minutes = int((total_seconds - hours * 3600.0) / 60.0)
    seconds = int(total_seconds - hours * 3600 - minutes * 60)
    hours = '0' + str(hours) if len(str(hours)) == 1 else str(hours)
    minutes = '0' + str(minutes) if len(str(minutes)) == 1 else str(minutes)
    seconds = '0' + str(seconds) if len(str(seconds)) == 1 else str(seconds)
    print(current_time.now(), " - Elapsed time: ", time_elapsed.days, "d", hours, "h", minutes, "m", seconds, "s - ", explanation)


# Defining the train/validation dataset loader, using the SubsetRandomSampler for the split
def load_split_train_test(root_dir, valid_size=.2):
    # Load in each dataset and apply transformations using the torchvision.datasets as datasets library
    # Note: While usually separate ./data/train/label/image.png and ./data/validation/label/image.png folders are used for train_data and test_data respectively,
    # here we're using the same directory since the train images are taken randomly from the total set
    # Note: we are using here the same transforms for both train and test data, though theoretically we could use different transforms for each
    train_data = datasets.ImageFolder(root_dir, transform=TEST_TRANSFORMS)
    test_data = datasets.ImageFolder(root_dir, transform=TEST_TRANSFORMS)

    # Getting a random samples from the images
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Then we want to put our imported images into a Dataloader
    #   Dataloader is able to spit out random samples of our data, so our model won’t have to deal with the entire dataset every time.
    #   This makes training more efficient.
    #   We specify how many images we want at once as our batch_size (so 32 means we want to get 32 images at one time).
    #   We also want to shuffle our images so it gets inputted randomly into our AI model.
    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=32)  # shuffle=True
    val_loader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=32)

    return train_loader, val_loader


def train_model():
    print_status("\tLoading the dataset")
    # Loading the dataset
    train_loader, val_loader = load_split_train_test(DATA_DIR, .2)
    # print(train_loader.dataset.classes)

    print_status("\tCreating the model")
    # Creating the model
    #   AI models need to be trained on a lot of data to be effective.
    #   Since we don’t have that much data, we want to take a pre-trained model
    #   (a model that has been previously trained on many images) but tailor it to recognize our specific images.
    #   This process is called transfer learning.
    # Image recognition models have two parts: the convolutional part and the classifier part.
    # We want to keep the pre-trained convolutional part but put in our own classifier. Here’s why:
    #   The convolution/pooling section of our model is used to recognize the features inside an image.
    #   It first identifies edges, then using the edges it identifies shapes, and using the shapes it can identify objects.
    #   But it takes A LOT of data to train this section — probably more than we have — so instead, we can use the default, pre-trained convolutional layers.
    #   These pre-trained convolutional layers were trained to identify these feature very well, regardless of what kind of image you have.
    #   There are also pooling layers in between convolutional layers that distill an image to a smaller size so it can be easily inputted to our classifier.
    # The last part of the model is the classifier.
    #   The classifier takes all the information extracted from the photo in the convolution part, and uses it to identify the image.
    #   This is the part of the pre-trained model we want to replace and to train on our own images.
    #   This makes the model tailored to identify the images we give it.
    # We use the torchvision.models library to download a pre-trained model.
    #   There are many different models we can download, and more info can be found here: https://pytorch.org/docs/stable/torchvision/models.html
    #   I chose a model called densenet161 and specified that we want it to be pre-trained by setting pretrained=True.
    model = models.densenet161(pretrained=True)
    print(model)

    print_status("\tFreezing pre-trained layers to prevent back-propagation")
    # Freezing pre-trained layers to prevent back-propagation through them during training
    # We want to make sure we don’t train this model since it is already trained, and we only want to train the classifier we will put in next.
    # We tell the model not to calculate the gradients of any parameter since that is only done for training.
    for param in model.parameters(): param.requires_grad = False

    print_status("\tUpdating the classifier")
    # Now we want to replace the default classifier of the model with our own classifier.
    # Classifiers are fully connected neural networks, so to do this, we must first build our own neural network.

    # A neural network is just a method to find complex patterns and connections between numbers of the input and the output.
    # In this case, it takes the features of the image that were highlighted by the convolution section to determine how likely the image is a certain label.
    # The first thing we want to do is to determine the amount of numbers inputted to our neural network.
    # This has to match the amount of numbers that is outputted from the section before (the convolutional section).
    # Since we didn’t change the convolutional section at all, the amount of numbers inputted in our classifier should be the same as the model’s default classifier.

    # Next, we want to determine the number of outputs. This number should match how many types of images you have.
    # The model will give you a list of percentages, each corresponding to how certain the picture is to that label.
    # So if you have images of bees, ants, and flies, there are 3 labels.
    # And there should be 3 numbers in the output layer each corresponding to the probability of the input being a bee, ant, or fly.

    # Once we have those details, we use the torch.nn library to create the classifier.
    #   nn.Sequential can help us group multiple modules together.
    #   nn.Linear specifies the interaction between two layers. We give it 2 numbers, specifying the number of nodes in the two layer.
    #     For example, in the first nn.Linear, the first layer is the input layer, and we can choose how many numbers we want in the second layer (I went with 1024).
    #   nn.ReLU is an activation function for hidden layers.
    #     Activation functions helps the model learn complex relationships between the input and the output.
    #     We use ReLU on all layers except for the output.
    #   We repeat this for as many hidden layers as you want, with as many nodes as you want in each layer.
    #   nn.LogSoftmax is the activation function for the output.
    #     The softmax function turns the outputted numbers into percentages for each label, and the log function is applied to make it computationally faster.
    #     We must specify that the output layer is a column, so we set dimension equal to 1.

    # Re-defining the final fully-connected the layer, the one that we’ll train with our images
    # classifier = nn.Sequential(nn.Linear(2048, 512),
    #                          nn.ReLU(),
    #                          nn.Dropout(0.2),
    #                          nn.Linear(512, 10),
    #                          nn.LogSoftmax(dim=1))

    classifier_input = model.classifier.in_features
    num_labels = 12  # PUT IN THE NUMBER OF LABELS IN YOUR DATA
    classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                               nn.ReLU(),
                               nn.Linear(1024, 512),
                               nn.ReLU(),
                               nn.Linear(512, num_labels),
                               nn.LogSoftmax(dim=1))

    # Replace default classifier with our new classifier
    model.classifier = classifier

    print_status("\tAdapting Torch to computation engine type: " + ("cuda" if torch.cuda.is_available() else "cpu"))
    # Adapting Torch to computation engine type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_status("\tMoving the model to the device")
    # Moving the model to the device specified above
    model.to(device)

    print_status("\tCreating the criterion (loss function)")
    # Creating the criterion (loss function)
    #   While training, we need to determine how “off” our model is.
    #   To evaluate the amount of error our model has, we use nn.NLLLoss.
    #   This function takes in the output of our model, for which we used the nn.LogSoftmax function.
    criterion = nn.NLLLoss()

    print_status("\tPicking the optimizer (Adam) and learning rate")
    # Picking the optimizer (Adam) and learning rate
    #   To train our model, we take our error and see how we can adjust the weights we multiplied our numbers by to get the smallest error.
    #   The method of calculating how we adjust our weights and applying it to our weights is called Adam.
    #   We use the torch.optim library to use this method and give it our parameters.
    # optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    optimizer = optim.Adam(model.classifier.parameters())

    print_status("\tTraining the model")
    # Training the model
    # Now we train. We want our model to go through the entire dataset multiple times, so we use a for loop.
    # Every time it has gone over the entire set of images, it is called an epoch.
    # In one epoch we want the model to go through both the training set and the validation set.
    epochs = 10
    train_loss = 0
    print_every = 1
    train_losses, test_losses, accuracies = [], [], []
    for epoch in range(epochs):

        print_status("\t\tSetting to training mode and looping through every image")
        # We first set the model to training mode and we use a for-loop (feed-forward loop) to go through every image.
        model.train()
        counter = 0
        for inputs, labels in train_loader:

            # Moving the images and the labels to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # Clearing the adjustments of the weights by declaring optimizer.zero_grad().
            optimizer.zero_grad()

            # Computing the output of our model given our images and how “off” our model is given its output and the correct answers
            output = model.forward(inputs)

            # Calculating the loss function
            # Then we can find the adjustments we need to make to decreases this error by calling loss.backward() and use our optimizer
            # to adjust the weights by calling optimizer.step() (aka. applying gradient descent in back-propagation).
            loss = criterion(output, labels)
            loss.backward()

            # Adjusting parameters based on the gradients
            optimizer.step()

            # Adding the loss to the training set's running loss
            # As we train, we want to know how things are going, so we keep track of the total errors we calculated and print out the progress of the training.
            train_loss += loss.item()

            # Displaying the losses and calculating accuracy every 10 batches
            counter += 1
            if counter % print_every == 0:
                print_status("\t\t\tClass " + str(counter) + " out of " + str(len(train_loader)))

        print_status("\t\tEvaluating the model")
        # Evaluating the model
        model.eval()
        test_loss = 0
        accuracy = 0
        counter = 0

        with torch.no_grad():  # Tell torch not to calculate gradients
            for inputs, labels in val_loader:

                # Moving the images and the labels to the appropriate device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                output = model.forward(inputs)

                # Calculate Loss
                val_loss = criterion(output, labels)

                # Add loss to the validation set's running loss
                test_loss += val_loss.item()

                # Since our model outputs a LogSoftmax, find the real percentages by reversing the log function
                ps = torch.exp(output)

                # Get the top class of the output
                top_p, top_class = ps.topk(1, dim=1)

                # See how many of the classes were correct?
                equals = top_class == labels.view(*top_class.shape)

                # Calculate the mean (get the accuracy for this batch) and add it to the running accuracy for this epoch
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Print the progress of our evaluation
                counter += 1
                if counter % print_every == 0:
                    print_status("\t\t\tClass " + str(counter) + " out of " + str(len(train_loader)))

        # Get the average loss for the entire epoch
        train_loss = train_loss / len(train_loader.dataset)
        validation_loss = val_loss / len(val_loader.dataset)

        # Print out the information
        print_status("\t\tResults:")
        print_status('\t\t\tTest Accuracy: ' + str(accuracy / len(val_loader)))
        print_status('\t\t\tTest Loss: ' + str(test_loss / len(val_loader)))
        print_status(f'\t\t\tEpoch: {epoch + 1}/{epochs}')
        print_status(f'\t\t\tTraining Loss: {train_loss / print_every:.3f}')
        print_status(f'\t\t\tValidation Loss: {validation_loss / print_every:.3f}')

        accuracies.append(accuracy / len(val_loader))
        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(val_loader))

        train_loss = 0

    print_status("\tPlotting")
    # Plotting the training and validation losses
    plt.plot(accuracies, label='Accuracy')
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

    return model


def prepare_image(image_path):
    # Load Image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((255, int(255 * (height / width))) if width < height else (int(255 * (width / height)), 255))

    # Get the dimensions of the new image size
    width, height = img.size

    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))

    # Turn image into numpy array
    img = np.array(img)

    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))

    # Make all values between 0 and 1
    img = img / 255

    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485) / 0.229
    img[1] = (img[1] - 0.456) / 0.224
    img[2] = (img[2] - 0.406) / 0.225

    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis, :]

    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image


def predict_image(image, model):
    # Pass the image through our model
    output = model.forward(image)

    # Reverse the log function in our output
    output = torch.exp(output)

    # Get the top predicted class, and the output percentage for that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()


def show_image(image):
    # Convert image to numpy
    image = image.numpy()

    # Un-normalize the image
    image[0] = image[0] * 0.226 + 0.445

    # Print the image
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))


def get_random_images(num):
    data = datasets.ImageFolder(DATA_DIR, transform=TEST_TRANSFORMS)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.__next__()
    images, labels, sample_fname = enumerate(loader, 0)
    return images, labels, classes


def classify_image(model, image_path):
    # Then we use a function that can process the image so it can be inputted into our model.
    #   We open the image, resize it by keeping the aspect ratio but making the shortest side only 255 px, and crop the center 224px by 224px.
    #   We then turn the picture into an array and make sure that the number of color channels is the first dimension instead of the last
    #   dimension by transposing the array. Next, we convert each value between 0 and 1 by dividing by 255.
    #   We then normalize the values by subtracting the mean and dividing by the standard deviation.
    #   Lastly, we convert the array into a Torch tensor and convert the values to float.
    image = prepare_image(image_path)

    # After processing the image, we can build a function to use our model to predict the label.
    # We input the image into our model and obtain the output.
    # We then reverse the log in the LogSoftmax function that we applied in the output layer and return the top class the model predicted and how certain it is of its guess.
    # Using our model to predict the label
    top_prob, top_class_index = predict_image(image, model)

    # Lastly, we want to display the image. We turn the image back into an array, and un-normalize it by multiplying by the
    # standard deviation and adding back the mean. We then use the matplotlib.pyplot library to plot the picture.
    # show_image(image)

    return image, top_prob, top_class_index


def get_random_image_paths(num_images):
    data = datasets.ImageFolder(DATA_DIR, transform=TEST_TRANSFORMS)
    classes = data.classes

    image_paths = []
    for path, subdirs, files in os.walk(DATA_DIR):
        for name in files:
            image_paths.append(os.path.join(path, name).replace("\\", '/'))
    random_image_paths = random.sample(image_paths, num_images)
    random_image_labels = [path.split('/')[2] for path in random_image_paths]

    return random_image_paths, random_image_labels, classes


def main():
    print_status("Starting model training")
    model = train_model()
    torch.save(model, 'stormfrei_model.pth')

    print_status("Starting image classifier")
    model = torch.load('stormfrei_model.pth')
    model.eval()  # Setting the model for evaluation mode.

    num_images = 24
    image_paths, labels, classes = get_random_image_paths(num_images)
    fig = plt.figure(figsize=(10, 10))
    fontdict = {
        'fontsize': 'medium',
        'fontweight': rcParams['axes.titleweight'],
        'verticalalignment': 'baseline',
        'horizontalalignment': 'center'
    }
    total_num_rows = 4
    total_num_cols = math.ceil(num_images / total_num_rows)
    for img_index in range(len(image_paths)):
        try:
            image, top_prob, top_class_index = classify_image(model, image_paths[img_index])
            print_status(f"The model is {top_prob * 100:1f} % certain that {image_paths[img_index]} has a predicted class of {classes[top_class_index]}")
            sub = fig.add_subplot(total_num_rows, total_num_cols, img_index + 1)
            label_matches_prediction = labels[img_index] == classes[top_class_index]
            sub.set_title(fontdict=fontdict, label="P:" + classes[top_class_index]
                                                   + "\nO:" + labels[img_index]
                                                   + "\nMatches: " + str(label_matches_prediction))
            plt.axis('off')
            output_image = mpimg.imread(image_paths[img_index])
            plt.imshow(output_image)
        except Exception:
            print_status(f"Failed to classify " + image_paths[img_index])
    plt.show()


main()
