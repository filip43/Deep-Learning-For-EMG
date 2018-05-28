
function [yShuffled,emgShuffled] = DataPreproccessingNoZeroMixedMyo(nameOfFile)
%DataShuffle is a function to correctly label and split the data
addpath('Datasets/s1Myo/');
load(nameOfFile);
combined = [emg, restimulus];
number_of_classes = max(combined(:,end)) + 1;
[~,~,X] = unique(combined(:,end));
C = accumarray(X,1:size(combined,1),[],@(r){combined(r,:)});
counter = [2 3 6 10 11 12 23 24];
trainingCombined = [];
testCombined = [];
for i = counter
    trainingCombined = [trainingCombined; C{i}(1:round(0.7*size(C{i},1)), :)];
    testCombined = [testCombined; C{i}((round(0.7*size(C{i},1))+1):end, :)];
end
shuffledTrainingCombined= trainingCombined(randperm(size(trainingCombined,1)),:);
shuffledTestCombined= testCombined(randperm(size(testCombined,1)),:);
emgShuffled = [shuffledTrainingCombined(:,1:end-1);shuffledTestCombined(:,1:end-1)];
disp(size(emgShuffled));
save('PreProccessed/FewerCombinedFunctionalMyo/emgShuffled.mat', 'emgShuffled');
yShuffled = [shuffledTrainingCombined(:,end);shuffledTestCombined(:,end)];

yShuffled(yShuffled == 2 | yShuffled == 5 | yShuffled == 1) = 1;
yShuffled(yShuffled == 9) = 2;
yShuffled(yShuffled == 10| yShuffled == 11| yShuffled == 12) = 3;
yShuffled(yShuffled == 22) = 4;
yShuffled(yShuffled == 23) = 5;


disp(size(yShuffled));
save('PreProccessed/FewerCombinedFunctionalMyo/yShuffled.mat', 'yShuffled');

end

