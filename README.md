# proyecto2-sistemas-inteligentes

Data normalization and pre-processing
Usually, the original images of the MNIST database are re-mixed, normalized, and pre-processed before fitting a model. This is an important operation because machine learning models won't work with this kind of samples without transforming the original data.

In this stage of the project, download all the available images in the Emojis database, and pre-process and normalize the images. You can try any normalization approach, but the recommended steps are the following for each image:

Binarize the image (white background, black lines).
Identify the bounding rect of the emoji.
Rescale the image inside the bounding rect to a size of 32 x 32.
Save the new small image.
Supervised learning
Train at least two types of classifiers (RB-SVM and KNN for example), and determine which classes can be identified by the models. Remember to use cross-validation to evaluate the performance of the classifiers. Additionally, train a convolutional neural network with this data, and compare the results obtained with the different models.

Unsupervised learning
Use K-Means and dendrograms to identify possible groups of similar images in the data set. Try different numbers of groups in the K-means method.

Are these groups organized according to the types of emojis?

Reinforcement learning (optional)
Train a convolutional neural network using reinforcement learning. How many training samples does the network need to predict correctly new observations?

Conclusions
Write some conclusions about this work. Let me know what you have learned, and what you would like to improve. Conclusions are individual. Each team member must write at least 400 words.
