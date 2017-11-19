function imdb = setup_chalearn15( varargin)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2015-09-07

% SETUPVOC    Setup Stanford40 data
% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.lite = false ;
opts.dataset = 'ChaLearn';
opts.dataDir = './data' ;

opts = vl_argparse(opts, varargin) ;

imdb.images.id = [] ;
imdb.images.set = uint8([]) ;
imdb.images.name = {} ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes =1:85;
imdb.imageDir = fullfile(opts.dataDir) ;



% ChaLearn faces
label_path = fullfile(opts.dataDir, opts.dataset, 'Train.csv');
if exist(label_path,'file')
    data = importdata(label_path);
    img_age = data.data;
    img_name = data.textdata;
else
    error('ChaLearn data not found in %s', opts.dataDir) ;
end
% Construct image database imdb structure
% training data
train_names = fullfile(opts.dataset,...
    [fullfile('Train/', img_name'),...
    fullfile('fTrain/', img_name'),...
    fullfile('GrayTrain/', img_name'),...
    fullfile('fGrayTrain/', img_name')]);

num_train = numel(train_names);

train_labels = repmat(img_age',1,4) ;
train_set = ones(1,num_train);

% validation data
clear img_age  img_name
label_path = fullfile(opts.dataDir, opts.dataset, 'Validation.csv');
if exist(label_path,'file')
    data= importdata(label_path);
    img_age = data.data;
    img_name = data.textdata;
end


val_names = fullfile(opts.dataset, 'Validation/',img_name' );
val_labels = img_age';
num_val = numel(val_names);
val_set = 2*ones(1,num_val);

imdb.images.id = 1: num_train + num_val ;
imdb.images.name = [train_names, val_names];
imdb.images.label = [train_labels, val_labels];
imdb.images.set = [train_set, val_set];

fprintf('num_train: %d, num_val: %d\n', num_train, num_val);

if opts.lite
    ok = {} ;
    for c = 1:3
        ok{end+1} = vl_colsubset(find(imdb.images.class == c & imdb.images.set == 1), 5) ;
        ok{end+1} = vl_colsubset(find(imdb.images.class == c & imdb.images.set == 2), 5) ;
        ok{end+1} = vl_colsubset(find(imdb.images.class == c & imdb.images.set == 3), 5) ;
    end
    ok = cat(2, ok{:}) ;
    imdb.meta.classes = imdb.meta.classes(1:3) ;
    imdb.images.id = imdb.images.id(ok) ;
    imdb.images.name = imdb.images.name(ok) ;
    imdb.images.set = imdb.images.set(ok) ;
    imdb.images.class = imdb.images.class(ok) ;
end



