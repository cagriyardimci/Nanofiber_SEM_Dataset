clear; clc;

% Load and preprocess the dataset
Dataset = imageDatastore('C:\Training_DataSet', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[Training_Dataset, Validation_Dataset] = splitEachLabel(Dataset, 0.7);

inputSize = [224 224 3];  % Adjusted input size


Resized_Training_Image = augmentedImageDatastore(inputSize(1:2), Training_Dataset);
Resized_Validation_Image = augmentedImageDatastore(inputSize(1:2), Validation_Dataset);

numClasses = 3;

layers = [
    imageInputLayer(inputSize, "Name", "input", "Normalization", "rescale-zero-one")
    convolution2dLayer(3, 32, "Name", "conv1", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm1")
    leakyReluLayer(0.1, "Name", "leaky1")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool1")
    convolution2dLayer(3, 64, "Name", "conv2", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm2")
    leakyReluLayer(0.1, "Name", "leaky2")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool2")
    convolution2dLayer(3, 128, "Name", "conv3", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm3")
    leakyReluLayer(0.1, "Name", "leaky3")
    dropoutLayer(0.4, "Name", "drop1")
    convolution2dLayer(3, 128, "Name", "conv4", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm4")
    leakyReluLayer(0.1, "Name", "leaky4")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool3")
    convolution2dLayer(3, 256, "Name", "conv5", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm5")
    leakyReluLayer(0.1, "Name", "leaky5")
    convolution2dLayer(3, 256, "Name", "conv6", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm6")
    leakyReluLayer(0.1, "Name", "leaky6")
    dropoutLayer(0.4, "Name", "drop2")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool4")
    convolution2dLayer(3, 512, "Name", "conv7", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm7")
    leakyReluLayer(0.1, "Name", "leaky7")
    convolution2dLayer(3, 512, "Name", "conv8", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm8")
    leakyReluLayer(0.1, "Name", "leaky8")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool5")
    convolution2dLayer(3, 1024, "Name", "conv9", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm9")
    leakyReluLayer(0.1, "Name", "leaky9")
    dropoutLayer(0.4, "Name", "drop3")
    convolution2dLayer(3, 1024, "Name", "conv10", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm10")
    leakyReluLayer(0.1, "Name", "leaky10")
    convolution2dLayer(1, numClasses, "Name", "conv11", "Padding", "same")
    globalAveragePooling2dLayer("Name", "avg1")
    softmaxLayer("Name", "softmax")
    classificationLayer("Name", "NanoFiberClassifier")
];

Size_of_Minibatch = 32;
Validation_Frequency = floor(numel(Resized_Training_Image.Files) / Size_of_Minibatch);

% Set training options
options = trainingOptions('adam', ...
    'MiniBatchSize', Size_of_Minibatch, ...
    'MaxEpochs', 20, ... % Increased epochs for more training
    'InitialLearnRate', 0.0001, ... % Adjusted initial learning rate
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Resized_Validation_Image, ...
    'ValidationFrequency', Validation_Frequency, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ... % Set to true for verbose output
    'ExecutionEnvironment', 'gpu'); % Adjust as needed

% Train the neural network
net = trainNetwork(Resized_Training_Image, layers, options);
