# Semantic-Segmentation-of-Real-World-Road-Images
This project is aimed to solve semantic segmentation using deep learning models and for this project  we are using the Cambridge labeled objects in video dataset. Which is an implementation of computer vision in autonomous vehicles for object detection.

<img width="396" alt="6" src="https://github.com/GiridharDhanapal/Semantic-Segmentation-of-Real-World-Road-Images/assets/117945886/b61c37cd-b1b0-45ed-8fc4-891b2a2e2df7">

# Dataset

The dataset consists of 101 images (960x720 pixel) in which each pixel was manually assigned to one of the following 32 object classes that are relevant in a driving environment. The "void" label indicates an area which ambiguous or irrelevant in this context. The colour/class association is given in the file label_colors.txt. Each line has the R G B values (between 0 and 255) and then the class name.

<img width="343" alt="1" src="https://github.com/GiridharDhanapal/Semantic-Segmentation-of-Real-World-Road-Images/assets/117945886/2caadcc8-49c9-4a79-9e18-a4d03ba36105">

<img width="335" alt="2" src="https://github.com/GiridharDhanapal/Semantic-Segmentation-of-Real-World-Road-Images/assets/117945886/c65b9c1d-a085-48c9-9c79-2b70aacaad61">

<img width="412" alt="5" src="https://github.com/GiridharDhanapal/Semantic-Segmentation-of-Real-World-Road-Images/assets/117945886/85cc472e-1879-4011-99fb-274624040f37">

# Description

We implemented three deep learning models for semantic segmentation,

**Fully Convolutional Network (FCN16)**

**Semantic Deep Segmentation (SDS)**

**U-Net**

The first model I implemented is the Fully Convolutional Network (FCN16), commonly used in object detection and medical image analysis. FCN16 employs convolution, pooling, and up-sampling layers. I defined the FCN16 architecture using a class, starting with the init method to set up the model structure. I used a pre-trained ResNet 101 model to help the network learn deep representations and avoid the vanishing gradient problem. I removed the fully connected and pooling layers, reducing the 2048 ResNet channels to 512 using convolutional layers. I applied batch normalization and ReLU activation for nonlinearity and included dropout layers to prevent overfitting. The forward function specifies how input data passes through the network, initializing necessary layers. I also created functions for initializing and updating weights, as well as for training and validation. These functions set parameters like epochs, optimizer, and loss functions, and calculate evaluation metrics such as Intersection over Union (IoU) and accuracy, along with plotting the results.

The SDS model, creates pixel-wise classification maps from input images. I started with a pre-trained ResNet101 model, locking its parameters and removing the fully connected and average pooling layers to make it fully convolutional. The remaining ResNet layers act as the feature extractor. I added a convolutional layer (kernel size 2, stride 1, padding 1) at the end of the feature extractor. The output then goes through batch normalization, ReLU activation, and dropout layers, followed by a 1x1 convolutional layer that maps the output to the correct number of classes. A transpose convolutional layer (kernel size 32, stride 16) upsamples the output to the original image size. Initially, the weights of the transpose convolutional layer are set for bilinear upsampling.

The third model I implemented is the U-Net architecture, which performs image segmentation using convolutional and transposed convolutional layers. The encoder downsamples the input image to capture its features, while the decoder upsamples these features to the original image resolution. The main component is the DoubleConv class, consisting of two convolutional layers with batch normalization and ReLU activation to learn complex information. The U-Net class creates an encoder-decoder structure using encoder and decoder blocks. Encoder blocks contain convolutional and max-pooling layers for downsampling, while decoder blocks use transposed convolutional layers for upsampling and DoubleConv blocks to combine upsampled features with corresponding encoder features. Skip connections maintain high-resolution details. The model also includes a bottleneck layer for feature extraction and a final convolutional layer to adjust the number of output channels. This symmetric structure with skip connections improves model accuracy by facilitating gradient propagation.

<img width="839" alt="4" src="https://github.com/GiridharDhanapal/Semantic-Segmentation-of-Real-World-Road-Images/assets/117945886/aa2a3643-d562-4f02-8a91-716864013b03">

# Hyperparameter tuning

We experimented with various hyperparameters like **optimizers, loss function, and suitable settings for learning rate, batch size, and epochs**.

<img width="434" alt="3" src="https://github.com/GiridharDhanapal/Semantic-Segmentation-of-Real-World-Road-Images/assets/117945886/ecd3abb0-5b35-457d-936e-6545409e1f23">

# Conclusion

This project offered valuable insights into deep learning techniques and image segmentation, presenting challenges and learning opportunities in model building and hyperparameter tuning.

# Tech Stack

**Editor/IDE:** VS Code (Visual Studio Code) for writing and debugging code.

**Notebook:** Jupyter Notebook for interactive experimentation and prototyping.

**Frameworks and Libraries** : Python, PyTorch, OpenCV, NumPy, Matplotlib, Albumentations, scikit-learn, torchvision, GridSearchCV, DataLoader


