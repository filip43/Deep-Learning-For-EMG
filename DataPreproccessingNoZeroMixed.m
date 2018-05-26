function [yShuffled,emgShuffled] = DataShuffle(nameOfFile)
%DataShuffle is a function to correctly label and split the data
addpath('Datasets/s1/');
load(nameOfFile);
combined = [emg, restimulus];
number_of_classes = max(combined(:,11)) + 1;
[~,~,X] = unique(combined(:,11));
C = accumarray(X,1:size(combined,1),[],@(r){combined(r,:)});
counter = [2 3 6 10 11 12 13 15 18 19 22 23 24];
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
save('PreProccessed/test/emgShuffled.mat', 'emgShuffled');
yShuffled = [shuffledTrainingCombined(:,11);shuffledTestCombined(:,11)];

yShuffled(yShuffled == 1) = 1;
yShuffled(yShuffled == 2 | yShuffled == 5) = 2;
yShuffled(yShuffled == 9) = 3;
yShuffled(yShuffled == 10| yShuffled == 11| yShuffled == 12) = 4;
yShuffled(yShuffled == 14) = 5;
yShuffled(yShuffled == 17) = 6;
yShuffled(yShuffled == 18) = 7;
yShuffled(yShuffled == 21) = 8;
yShuffled(yShuffled == 22) = 9;
yShuffled(yShuffled == 23) = 10;


disp(size(yShuffled));
save('PreProccessed/TestTest/yShuffled.mat', 'yShuffled');

end

