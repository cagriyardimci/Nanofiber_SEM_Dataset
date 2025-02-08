% Load the pre-trained network
load('nanofiber.mat');

% Test datasets and folder paths
folders = {'C:\Test_Dataset\SlightlyDefective', 'C:\Test_Dataset\Defective', 'C:\Test_Dataset\NonDefective'};
labels = {'AzKusurlu', 'Kusurlu', 'Kusursuz'};
displayLabels = {'Slightly Defective', 'Defective', 'Non-Defective'};

% Cells to store labels
Label_az = [];
Label_kusurlu = [];
Label_kusursuz = [];

% Variables to store total processing time and total number of files
total_time_ms = 0;
total_files = 0;

for i = 1:3
    folderpath = folders{i};
    filelist = dir(fullfile(folderpath, '*.tif')); % List only .tif files
    row = numel(filelist);
    total_files = total_files + row; % Increase the total file count
    
    % Temporary variable to store labels
    tmp_labels = [];
    
    for k = 1:row
        tmp = string(filelist(k).name);
        image = fullfile(folderpath, tmp);
        
        % Calculate processing time
        tic;
        [Label, Probability] = test_network(net, image, 1);
        elapsed_time = toc; % Stop timing and get the elapsed time
        total_time_ms = total_time_ms + (elapsed_time * 1000); % Convert to milliseconds and add to total time
        
        % Append labels to the temporary variable
        tmp_labels = [tmp_labels; string(Label)];
    end
    
    % Assign labels to the corresponding variable
    switch labels{i}
        case 'AzKusurlu'
            Label_az = tmp_labels;
        case 'Kusurlu'
            Label_kusurlu = tmp_labels;
        case 'Kusursuz'
            Label_kusursuz = tmp_labels;
    end
end

% Print processing time
average_time_ms = total_time_ms / total_files;

fprintf('Total number of files: %d\n', total_files);
fprintf('Total processing time: %.2f milliseconds\n', total_time_ms);
fprintf('Average processing time: %.2f milliseconds\n', average_time_ms);

% Create confusion matrix
confusion_matrix = zeros(3, 3);

confusion_matrix(1, :) = [numel(find(Label_az == 'AzKusurlu')), numel(find(Label_az == 'Kusurlu')), numel(find(Label_az == 'Kusursuz'))];
confusion_matrix(2, :) = [numel(find(Label_kusurlu == 'AzKusurlu')), numel(find(Label_kusurlu == 'Kusurlu')), numel(find(Label_kusurlu == 'Kusursuz'))];
confusion_matrix(3, :) = [numel(find(Label_kusursuz == 'AzKusurlu')), numel(find(Label_kusursuz == 'Kusurlu')), numel(find(Label_kusursuz == 'Kusursuz'))];

% Sensitivity, Specificity, Accuracy, Precision, F-measure
sensitivity = zeros(3, 1);
specificity = zeros(3, 1);
accuracy = zeros(3, 1);
precision = zeros(3, 1);
f_measure = zeros(3, 1);

for i = 1:3
    TP = confusion_matrix(i, i);                   % True Positives
    FN = sum(confusion_matrix(i, :)) - TP;         % False Negatives
    FP = sum(confusion_matrix(:, i)) - TP;         % False Positives
    TN = sum(confusion_matrix(:)) - (TP + FN + FP); % True Negatives
    
    sensitivity(i) = (TP / (TP + FN)) * 100;
    specificity(i) = (TN / (TN + FP)) * 100;
    accuracy(i) = ((TP + TN) / (TN + FP + FN + TP)) * 100;
    precision(i) = (TP / (TP + FP)) * 100;          
    f_measure(i) = ((2 * TP) / (2 * TP + FP + FN)) * 100;   
end

% Sensitivity, Specificity, Accuracy, Precision, F-measure tables
T = table(displayLabels', sensitivity, specificity, accuracy, precision, f_measure, ...
    'VariableNames', {'Class', 'Sensitivity', 'Specificity', 'Accuracy', 'Precision', 'FMeasure'});

disp(T);

% Traditional Axes-Based Confusion Matrix Plot
figure;
imagesc(confusion_matrix);
colormap(jet);
colorbar;

% Set axes labels
ax = gca;
ax.XTick = 1:3; 
ax.YTick = 1:3;
ax.XTickLabel = displayLabels;
ax.YTickLabel = displayLabels;
xlabel('Predicted');
ylabel('Actual');

% Set font size and bold for axes labels
ax.FontSize = 16;
ax.FontWeight = 'bold';

% Overlay text (cell values) with bold formatting
for i = 1:size(confusion_matrix, 1)
    for j = 1:size(confusion_matrix, 2)
        text(j, i, num2str(confusion_matrix(i, j)), 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'Color', 'white');
    end
end
