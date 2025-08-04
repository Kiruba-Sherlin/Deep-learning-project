# DEEP LEARNING PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: KIRUBA SHERLIN. A

INTERN ID: CT04DH2074

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

DESCRIPTION:
To build and evaluate a deep learning model capable of classifying 32x32 color images into one of 10 object categories using the CIFAR-10 dataset.
📚 Dataset Used:
CIFAR-10 (available via TensorFlow's keras.datasets) Contains 60,000 color images (32x32 pixels) in 10 classes. 50,000 training images and 10,000 test images.
Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

🛠️ Tools & Technologies:
1) Python
2) TensorFlow and Keras (for building the CNN model)
3) Matplotlib and NumPy (for visualization and preprocessing)

This project focuses on implementing a deep learning-based image classification system using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. The CIFAR-10 dataset is a well-known benchmark in the field of computer vision, containing 60,000 color images sized 32x32 pixels across 10 different classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is divided into 50,000 training images and 10,000 test images. The objective of this project is to build a model that can accurately classify an unseen image into one of these ten categories. To accomplish this, the dataset is first loaded using TensorFlow’s built-in Keras API. The image pixel values are normalized to the range [-1, 1] to help improve model performance during training. A set of 10 images is displayed with their corresponding class labels to visualize and understand the dataset.

The core of the project lies in designing and training a CNN model using TensorFlow and Keras. The CNN architecture is composed of multiple layers: two convolutional layers followed by max-pooling layers, a flatten layer to convert the 2D feature maps into 1D, and finally, three dense layers including an output layer with 10 neurons and softmax activation. The model uses the ReLU activation function for hidden layers to introduce non-linearity, while the softmax activation function in the final layer outputs a probability distribution over the 10 classes. The model is compiled with the Adam optimizer and trained using sparse categorical crossentropy loss, which is suitable for multi-class classification problems with integer labels. The training process is carried out for 10 epochs, and validation is performed using the test dataset to monitor the model's performance after each epoch.

Once the model is trained, its performance is evaluated on the test set. The model’s summary is printed to show the total number of parameters and the structure of each layer. After training, the model is used to predict the class of a test image. A visualization function is included to display the test image along with a bar chart showing the model’s confidence for each of the 10 classes. This helps in understanding not only what the model predicted but also how confident it is in its prediction. Finally, the overall accuracy of the model on the test dataset is calculated and printed. This metric gives an estimate of how well the model is expected to perform on unseen data.

In conclusion, this project demonstrates a complete pipeline for image classification using deep learning techniques. It covers data preprocessing, model building, training, evaluation, and visualization. With an accuracy typically around 70–80%, the model performs reasonably well for a basic CNN architecture. There is potential to further enhance performance through advanced techniques such as data augmentation, dropout, batch normalization, or using deeper architectures like VGG16 or ResNet. This project not only provides hands-on experience with TensorFlow and convolutional neural networks but also lays the foundation for more complex computer vision applications in real-world scenarios.
