function IMDB = setup_chalearn( varargin)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2015-09-07

% SETUPVOC    Setup Stanford40 data
% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.dataDir = fullfile('data','ChaLearn') ;
opts.lite = false ;
opts.sample_num = [];
opts = vl_argparse(opts, varargin) ;

imdb.images.id = [] ;
imdb.images.set = uint8([]) ;
imdb.images.name = {} ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes =[];
imdb.imageDir = fullfile(opts.dataDir) ;

% Download and unpack
% vl_xmkdir(datasetDir) ;
% ChaLearn faces
label_path = fullfile(opts.dataDir,'Train.csv');
if exist(label_path,'file')
    data = importdata(label_path);
    img_age = data.data;
    img_name = data.textdata;
    
    if ~isempty(opts.sample_num)
        rng(1)
        sel_id = randsample(1:numel(img_name),opts.sample_num);
        img_age = img_age(sel_id,:);
        img_name = img_name(sel_id,:);
    end
    
else
    error('ChaLearn data not found in %s', opts.dataDir) ;
end
% Construct image database imdb structure
% train data
% names = cellstr([fullfile('Train/', img_name'),fullfile('fTrain/', img_name'),...
%     fullfile('3GTrain/', img_name'),fullfile('GrayTrain/', img_name'),...
%     fullfile('SobelTrain/', img_name'),fullfile('fTrain/', img_name')]);

names = cellstr([fullfile('Train/', img_name'),fullfile('fTrain/', img_name'),...
    fullfile('GrayTrain/', img_name'),fullfile('fGrayTrain/', img_name')]);
% names = cellstr([fullfile('Train/', img_name'),fullfile('fTrain/', img_name')]);

labels = repmat(img_age',1,4) ;
set = ones(1,numel(names));
num_train = numel(names);

% validation data
clear img_age  img_name
label_path = fullfile(opts.dataDir,'Validation.csv');
if exist(label_path,'file')
    data= importdata(label_path);
    img_age = data.data;
    img_name = data.textdata;
end

val_iamges = fullfile('Validation/',img_name' );
names =[names, val_iamges];
labels = [labels, img_age'];
set = [set,2*ones(1,numel(val_iamges))];
num_val = numel(val_iamges);

% test data
clear img_age  image_names
img_name = struct2cell(dir(fullfile('Test','*.jpg')));
img_name(2:end,:) =[];

test_image = fullfile('Test/',img_name);
imdb.images.name=[names, test_image];
imdb.images.label = [labels,zeros(2,size(img_name,2))];
imdb.images.id = 1:numel(imdb.images.name) ;
imdb.images.set = [set,3*ones(1,numel(test_image))];
num_test = numel(test_image);

IMDB{1} = imdb;

fprintf('num_train: %d, num_val: %d, num_test: %d \n', num_train, num_val, num_test);
% ages distrbution statistic
% age = unique(imdb.meta.classes(1,:));
% for i = 1:length(age)
%     id = find(imdb.meta.classes(1,:) == age(1,i));
%     std(1,i) = mean(imdb.meta.classes(2,id));
% end
% 
% figure
% plot(age,std,'-o')
% hold
% pa = polyfit(age,std,3);
% x = 1:85;
% y = pa(1)*x.^3+pa(2)*x.^2+pa(3)*x.^1+pa(4);
% plot(x,y,'r');

% Adience faces
% dataset = 'Adience_faces';
% file = struct2cell(dir(fullfile(datasetDir,dataset)));
% file(2:end,:) = [];
% img_names = [];
% img_ages = [];
% 
% for f = 3:numel(file)
%     clear img_age img_name
%     img_name = [struct2cell( dir(fullfile(datasetDir,dataset,file{1,f},'*.jpg'))),struct2cell( dir(fullfile(datasetDir,file{1,f},dataset,'*.JPG')))];
%     img_name(2:end,:) = [];
%     img_age(1,:) = max(str2num(file{1,f}),1).*ones(1,numel(img_name));
%     img_age(2,:) = y(str2num(file{1,f})).*ones(1,numel(img_name));
%     
%     img_names = [ img_names fullfile(dataset,file{1,f},img_name)];
%     img_ages = [ img_ages img_age];
% end
% % 
% img_ages(2,:) = img_ages(2,:)+0.1.*randn(1,numel(img_names));
% imdb.images.name=[imdb.images.name,img_names];
% imdb.meta.classes = [imdb.meta.classes, img_ages];
% imdb.images.id = 1:numel(imdb.images.name) ;
% imdb.images.set = [imdb.images.set,ones(1,numel(img_names))];


% Adience faces
% dataset = 'bing-old';
% file = struct2cell(dir(fullfile(datasetDir,dataset)));
% file(2:end,:) = [];
% img_names = [];
% img_ages = [];
% 
% for f = 3:numel(file)
%     clear img_age img_name
%     img_name = [struct2cell( dir(fullfile(datasetDir,dataset,file{1,f},'*.jpg'))),struct2cell( dir(fullfile(datasetDir,file{1,f},dataset,'*.JPG')))];
%     img_name(2:end,:) = [];
%     img_age(1,:) = max(str2num(file{1,f}),1).*ones(1,numel(img_name));
%     img_age(2,:) = y(str2num(file{1,f})).*ones(1,numel(img_name));
%     
%     img_names = [ img_names fullfile(dataset,file{1,f},img_name)];
%     img_ages = [ img_ages img_age];
% end
% % 
% img_ages(2,:) = img_ages(2,:)+0.1.*randn(1,numel(img_names));
% imdb.images.name=[imdb.images.name,img_names];
% imdb.meta.classes = [imdb.meta.classes, img_ages];
% imdb.images.id = 1:numel(imdb.images.name) ;
% imdb.images.set = [imdb.images.set,ones(1,numel(img_names))];

% ChaLearn faces
% dataset = 'ChaLearn_Det'; 
% label_path =fullfile(datasetDir,'ChaLearn','Train.csv'); 
% if exist(label_path)
%     data= importdata(label_path); 
%     img_age = data.data;
%     image_names =data.textdata;
% else
%     error('Stanford40 data not found in %s', datasetDir) ;
% end % Construct image database imdb structure train data 
% img_names =cellstr(fullfile(dataset,'Train/', image_names'));
% imdb.images.name=[imdb.images.name,img_names]; 
% imdb.meta.classes =[imdb.meta.classes, img_age'];
% imdb.images.id = 1:numel(imdb.images.name);
% imdb.images.set = [imdb.images.set,ones(1,numel(img_names))];
% sel = 1:numel(imdb.images.name) ;
% selTrain = vl_colsubset(sel, 2400) ;
% selVal = vl_colsubset(setdiff(sel, selTrain), 76) ;
% imdb.images.set(selTrain) = 1;
% imdb.images.set(selVal) = 2;
% opts.numTrain = 100;
% opts.numTest = inf;
% opts.numVal = 0;
% 
% numClasses =numel(imdb.meta.classes);
% classes = imdb.images.class;
% for c = 1:numClasses
%   sel = find(classes == c) ;
%  % randn('state', 1) ;
%  % rand('state', 1) ;
%   selTrain = vl_colsubset(sel, opts.numTrain) ;
%   selVal = vl_colsubset(setdiff(sel, selTrain), opts.numVal) ;
%   selTest = vl_colsubset(setdiff(sel, [selTrain selVal]), opts.numTest) ;
%   Sets(selTrain) = 1 ;
%   Sets(selVal) = 2 ;
%   Sets(selTest) = 3 ;
% end
% 
% ok = find(Sets ~= 0) ;
% imdb.images.id =imdb.images.id(ok) ;
% imdb.images.name = imdb.images.name(ok) ;
% imdb.images.set = Sets(ok) ;
% imdb.images.class = classes(ok) ;

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



