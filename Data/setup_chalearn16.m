function imdb = setup_chalearn16( varargin)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2015-09-07

% SETUPVOC    Setup Stanford40 data
% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
% opts.dataset = 'ChaLearn';
% opts.dataDir = './data' ;
dataDir = '/mnt/data3/gaobb/image_data/image_faces/age_faces';
dataset = 'MTCNN_Chalearn16';
listDir = '/mnt/data3/gaobb/experiment/FB_Torch_SS_Age/train_val_list';

train_list = 'train_gt.csv';
val_list = 'valid_gt.csv';
test_list = 'test_gt.csv';

% get training list
data = importdata(fullfile(listDir, train_list));
train_name = fullfile(dataset, 'Align.5Train',data.rowheaders);

train_mu = data.data(:,1);
train_sigma = data.data(:,2);

% get validation list
data = importdata(fullfile(listDir, val_list));
val_name = fullfile(dataset, 'Align.5Valid',data.rowheaders);
val_mu = data.data(:,1);
val_sigma = data.data(:,2);


% get testint list
data = importdata(fullfile(listDir, test_list));
test_name = fullfile(dataset, 'Align.5Test',data.rowheaders);
test_mu = data.data(:,1);
test_sigma = data.data(:,2);
 
dataset = 'MTCNN_Google/Align_GoogleClean';
list = 'GoogleClean.txt';
data = importdata(fullfile(listDir, list));
name = fullfile(dataset, 'MTCNN_GoogleClean',data.rowheaders);
mu = data.data(:,1);
sigma = data.data(:,2);


opts.lite = false ;
% opts = vl_argparse(opts, varargin) ;

imdb.images.id = [] ;
imdb.images.set = uint8([]) ;
imdb.images.name = {} ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes =0:100;
imdb.imageDir = fullfile(dataDir) ; 
 
 
num_train = numel(train_name);
num_val = numel(val_name);
num_test = numel(test_name);
num_google = numel(name);

imdb.images.id = 1: num_train + num_val + num_test + num_google;
imdb.images.name = [train_name', val_name', test_name', name'];
imdb.images.label = [train_mu', val_mu', test_mu', mu';
                     train_sigma', val_sigma', test_sigma', sigma'];
imdb.images.set = [ones(1, num_train), ones(1, num_val), 2.*ones(1, num_test), 4.*ones(1, num_test)];

fprintf('num_train: %d, num_val: %d num_test:%d \n', num_train, num_val, num_test);

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

% opts.lite = false ;
% opts = vl_argparse(opts, varargin) ;
% 
% imdb.images.id = [] ;
% imdb.images.set = uint8([]) ;
% imdb.images.name = {} ;
% imdb.meta.sets = {'train', 'val', 'test'} ;
% imdb.meta.classes =1:85;
% imdb.imageDir = fullfile(opts.dataDir) ;
% 
% 
% 
% % ChaLearn faces
% label_path = fullfile(opts.dataDir, opts.dataset, 'Train.csv');
% if exist(label_path,'file')
%     data = importdata(label_path);
%     img_age = data.data;
%     img_name = data.textdata;
% else
%     error('ChaLearn data not found in %s', opts.dataDir) ;
% end
% % Construct image database imdb structure
% % training data
% % train_names = fullfile(opts.dataset,...
% %     [fullfile('Train/', img_name'),...
% %     fullfile('fTrain/', img_name'),...
% %     fullfile('GrayTrain/', img_name'),...
% %     fullfile('fGrayTrain/', img_name')]);
% train_names = fullfile(opts.dataset,...
%     [fullfile('Train/', img_name')]);
% 
% num_train = numel(train_names);
% 
% train_labels = repmat(img_age',1,1) ;
% train_set = ones(1,num_train);
% 
% % validation data
% clear img_age  img_name
% label_path = fullfile(opts.dataDir, opts.dataset, 'Validation.csv');
% if exist(label_path,'file')
%     data= importdata(label_path);
%     img_age = data.data;
%     img_name = data.textdata;
% end
% 
% val_names = fullfile(opts.dataset, 'Validation/',img_name' );
% val_labels = img_age';
% num_val = numel(val_names);
% val_set = 2*ones(1,num_val);
% 
% imdb.images.id = 1: num_train + num_val ;
% imdb.images.name = [train_names, val_names];
% imdb.images.label = [train_labels, val_labels];
% imdb.images.set = [train_set, val_set];
% 
% fprintf('num_train: %d, num_val: %d\n', num_train, num_val);

% if opts.lite
%     ok = {} ;
%     for c = 1:3
%         ok{end+1} = vl_colsubset(find(imdb.images.class == c & imdb.images.set == 1), 5) ;
%         ok{end+1} = vl_colsubset(find(imdb.images.class == c & imdb.images.set == 2), 5) ;
%         ok{end+1} = vl_colsubset(find(imdb.images.class == c & imdb.images.set == 3), 5) ;
%     end
%     ok = cat(2, ok{:}) ;
%     imdb.meta.classes = imdb.meta.classes(1:3) ;
%     imdb.images.id = imdb.images.id(ok) ;
%     imdb.images.name = imdb.images.name(ok) ;
%     imdb.images.set = imdb.images.set(ok) ;
%     imdb.images.class = imdb.images.class(ok) ;
% end



