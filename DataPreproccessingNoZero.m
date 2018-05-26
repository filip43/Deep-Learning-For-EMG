function [yShuffled,emgShuffled] = DataShuffle(nameOfFile)
%DataShuffle is a function to correctly label and split the data
addpath('Datasets/s1/');
load(nameOfFile);
combined = [emg, restimulus];
number_of_classes = max(combined(:,11)) + 1;
[~,~,X] = unique(combined(:,11));
C = accumarray(X,1:size(combined,1),[],@(r){combined(r,:)});
counter = 2:number_of_classes;
trainingCombined = [];
testCombined = [];
for i = counter
    trainingCombined = [trainingCombined; C{i}(1:round(0.7*size(C{i},1)), :)];
    testCombined = [testCombined; C{i}((round(0.7*size(C{i},1))+1):end, :)];
end
shuffledTrainingCombined= trainingCombined(randperm(size(trainingCombined,1)),:);
shuffledTestCombined= testCombined(randperm(size(testCombined,1)),:);
emgShuffled = [shuffledTrainingCombined(:,1:10);shuffledTestCombined(:,1:10)];
disp(size(emgShuffled));
save('PreProccessed/s1NoZero/emgShuffled.mat', 'emgShuffled');
yShuffled = [shuffledTrainingCombined(:,11);shuffledTestCombined(:,11)];
disp(size(yShuffled));
save('PreProccessed/s1NoZero/yShuffled.mat', 'yShuffled');

end

