clear; close all; clc;

% This code is modified from the baseline system for ASVspoof 2019

% add required libraries to the path
addpath('baseline/LFCC/');
addpath('baseline/CQCC_v1.0');

% set here the experiment to run (access and feature type)
access_type = 'LA'; 

% set paths to the wave files and protocols
pathToASVspoof2019Data = 'C:/Users/ÑîÇ§/Desktop/ÕæÎ±ÓïÒô¼ø±ð/';          %æ ¹è·¯å¾?
pathToFeatures = 'C:/Users/ÑîÇ§/Desktop/ÕæÎ±ÓïÒô¼ø±ð/Features/';

pathToDatabase = [pathToASVspoof2019Data, '/zju_deepfake'];          %zju_deepfakeæ–‡ä»¶å¤¹ï¼›LAæ–‡ä»¶å¤?; 
%trainProtocolFile = fullfile(pathToDatabase, train.txt);          %trainå¯¹åº”ç»“æžœçš„æ–‡ä»¶å¤¹
%devProtocolFile = fullfile(pathToDatabase, dev.txt);              %devå¯¹åº”ç»“æžœçš„æ–‡ä»¶å¤¹
%evalProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_cm_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.eval.trl.txt'));            %evalå¯¹åº”ç»“æžœçš„æ–‡ä»¶å¤¹

%{
% read train protocol
trainfileID = fopen(trainProtocolFile);
trainprotocol = textscan(trainfileID, '%s%s%s');
fclose(trainfileID);
trainfilelist = trainprotocol{1};

% read dev protocol
devfileID = fopen(devProtocolFile);
devprotocol = textscan(devfileID, '%s%s%s');
fclose(devfileID);
devfilelist = devprotocol{1};


% read eval protocol
evalfileID = fopen(evalProtocolFile);
evalprotocol = textscan(evalfileID, '%s%s%s%s%s');
fclose(evalfileID);
evalfilelist = evalprotocol{2};


%% Feature extraction for training data

% extract features for training data and store them
disp('Extracting features for training data...');
trainFeatureCell = cell(length(trainfilelist), 3);
for i=1:length(trainfilelist)
    filePath = fullfile(pathToDatabase,'train/flac',[trainfilelist{i} '.flac']);
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    LFCC = [stat delta double_delta]';
    filename_LFCC = fullfile(pathToFeatures, 'train', horzcat('LFCC_', trainfilelist{i}, '.mat'))
    parsave(filename_LFCC, LFCC)
    LFCC = [];
end
disp('Done!');

%% Feature extraction for development data

% extract features for training data and store them
disp('Extracting features for development data...');
for i=1:length(devfilelist)
    filePath = fullfile(pathToDatabase,'dev/flac',[devfilelist{i} '.flac']);
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    LFCC = [stat delta double_delta]';
    filename_LFCC = fullfile(pathToFeatures, 'dev', horzcat('LFCC_', devfilelist{i}, '.mat'))
    parsave(filename_LFCC, LFCC)
    LFCC = [];
end
disp('Done!');
%}


%% Feature extraction for evaluation data

% extract features for evaluation data and store them
disp('Extracting features for evaluation data...');
cd ../zju_deepfake/eval2/flac
filenames= dir();
%filenames = filenames_sum.name;
%disp(filenames(3).name);
%file = filenames(3);
%filename = filenames(3).name;
%new_filename = erase(filename,'.flac');

for i=1:(length(filenames) - 2)
    filename =  filenames(i+2).name;
    head_filename = erase(filename,'.flac');
    filePath = fullfile(pathToDatabase,'eval2/flac/',filename);
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    LFCC = [stat delta double_delta]';
    filename_LFCC = fullfile(pathToFeatures, 'eval2', horzcat('LFCC_', head_filename, '.mat'));
    parsave(filename_LFCC, LFCC)
    LFCC = [];
end
disp('Done!');


%% supplementary function
function parsave(fname, x)
    save(fname, 'x')
end
