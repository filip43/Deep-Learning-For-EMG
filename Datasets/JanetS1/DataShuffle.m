function [yShuffled,emgShuffled] = DataShuffle(nameOfFile)
%DataShuffle is a function to correctly label and split the data
addpath('/Users/Filip/Dropbox/Programming/GitHub-Repos/Deep-Learning-For-EMG-Signals/Datasets/s1/');
load(nameOfFile);
combined = [emg, repetition];
disp(size(combined));
save('combined.mat', 'combined');
end

