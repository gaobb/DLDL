function [net, info] = PFDLDLNet(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

% run(fullfile(fileparts(mfilename('fullpath')),...
%   '../../../toolbox/matconvnet-1.0-beta18', 'matlab', 'vl_setupnn.m')) ;
% run(fullfile(fileparts(mfilename('fullpath')),...
%   './matconvnet-1.0-beta18', 'matlab', 'vl_setupnn.m')) ;
%% load edge-bbox model
addpath('./External/edges'); 
addpath(genpath('./External/Piotr-CV')); 
addpath('./External/Ncut_9'); 

data_rootnn = '/mnt/data3/gaobb/image_data/PASCAL/';
%model_rootnn = '/data/gaobb/DLDL_v2.0/models';
short_slide = 256;
opts.dataset = 'voc07';
opts.modelType = 'vgg16';
opts.gpus_id = [];
opts.PD = [];
opts.loss = 'klloss';
opts.trainval_set = '';
% opts.loss = 'l2loss';
opts.weightInitMethod = 'gaussian';
opts.dataDir = fullfile(data_rootnn,opts.dataset) ;

opts.networkType = 'simplenn' ;
opts.pool5Type = 'avg';
% opts.pool5Type = 'max';

opts.batchNormalization = false ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.data_type = 'ml';
model_rootnn = ['./Models', opts.dataset];

sfx =  ['PFDLDL', opts.modelType] ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
if ~isempty(opts.PD)
    sfx = [sfx '-PD_', num2str(opts.PD),'-epsilon_0.01'];
else
    opts.PD = 0.3;
end
    
sfx = [sfx '-' opts.networkType] ;

opts.expDir = fullfile(model_rootnn, opts.dataset) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
% opts.imdbPath = fullfile(opts.expDir, [opts.dataset,'_bbox_imdb.mat']);
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;


if ~exist(opts.expDir,'dir')
    mkdir(opts.expDir);
end
opts.diary = fullfile(opts.expDir, 'diary.txt');
fid = fopen(opts.diary,'a');
fprintf(fid,'date: %s dataset:%s  loss:%s \n',date,opts.dataset,opts.loss);


tic; 
opts.train.expDir = fullfile(opts.expDir ,[sfx,'-',opts.loss,'-',opts.pool5Type]);

opts.train.gpus = opts.gpus_id;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
prenet_path = sprintf('./SimModel/ifdldl_%s_%s_%s', opts.modelType, opts.pool5Type, opts.dataset);
load(prenet_path);
hcp_layer =  struct('type', 'hcp', 'name', 'hcp') ;
net.layers = [net.layers(1:end-2), hcp_layer, net.layers(end-1:end)];

% Meta parameters
net.meta.normalization.interpolation = 'bilinear' ;
net.meta.normalization.keepAspect = true ;
% net.meta.augmentation.rgbVariance = zeros(0,3) ;
% net.meta.augmentation.transformation = 'stretch' ;

if ~opts.batchNormalization
    switch  opts.loss
        case  {'sigmoid_cross_entropy_loss', 'ls_sigmoid_cross_entropy_loss'}
            lr = logspace(-4, -5, 20) ;
        case  'cross_entropy_loss'
            lr = logspace(-6, -6, 20) ;
        otherwise
            lr = logspace(-4, -6, 20) ;
    end
else
    lr = logspace(-2, -4, 20) ;
end

net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = 3 ;
net.meta.trainOpts.weightDecay = 0.0005 ;

% Fill in default values
net = vl_simplenn_tidy(net) ;                       
vl_simplenn_display(net);
% vl_simplenn_diagnose(net)
% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
if exist(opts.imdbPath)
    load(opts.imdbPath) ;
else
    %imdb = setupVoc(opts.dataDir, 'lite', opts.lite) ;
    imdb = setupVoc12(opts.dataDir, 'lite', opts.lite) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, 'imdb') ;
end
% imdb.imageDir = strrep(imdb.imageDir,'image_data','image_data/PASCAL');
labels_ld = imdb.images.labels;
switch opts.loss
    case 'klloss'
        ind = find(labels_ld ==0);
        labels_ld(ind) = opts.PD;
        ind = find(labels_ld ==-1);
        labels_ld(ind) = 0;
        labels_ld = bsxfun(@times,labels_ld,1./sum(labels_ld));
        imdb.images.labels_ld = (1-0.01).*labels_ld + 0.01/20;
    case 'baseline_klloss'    
        ind = find(labels_ld ==-1);
        labels_ld(ind) = 0;
        labels_ld = bsxfun(@times,labels_ld,1./sum(labels_ld));
        imdb.images.labels_ld = labels_ld;
    case 'l2loss'
        ind = find(labels_ld <1);
        labels_ld(ind) = 0;
        imdb.images.labels_ld = bsxfun(@times,labels_ld,1./sum(labels_ld));
    case {'sigmoid_cross_entropy_loss', 'cross_entropy_loss'}
        ind = find(imdb.images.labels == -1);
        labels_ld(ind) = 0; % trans GT label from {-1,1} to {0,1}
        imdb.images.labels_ld = labels_ld;
    case {'ls_sigmoid_cross_entropy_loss'}
        PD_ind = find(imdb.images.labels == 0);
        labels_ld(PD_ind) = 0.3; % trans GT label from {-1,1} to {0,1}
        PN_ind = find(imdb.images.labels == -1);
        labels_ld(PN_ind) = 0.1; % trans GT label from {-1,1} to {0,1}
        PP_ind = find(imdb.images.labels == 1);
        labels_ld(PP_ind) = 0.9; % trans GT label from {-1,1} to {0,1}
        imdb.images.labels_ld = labels_ld;
end
% Set the class names in the network
net.meta.classes.name = imdb.meta.classes ;
%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('./toolbox/edges/models/forest/modelBsds'); 
model=model.model;
model.opts.multiscale=0; 
model.opts.sharpen=2; 
model.opts.nThreads=4;
opts.edge_model = model;

% -------------------------------------------------------------------------
%                                          cnn_ml_net                           Learn
% -------------------------------------------------------------------------
switch opts.networkType
  case 'simplenn', trainFn = @dldl_train_ml ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end
switch opts.trainval_set
    case 'train+val'
        train_id = find(imdb.images.set ==1);
        val_id = find(imdb.images.set ==2);
    case 'trainval+test'
        train_id = find(imdb.images.set <=2);
        val_id = find(imdb.images.set ==3);    
    otherwise
        train_id = find(imdb.images.set <=2);
        val_id = NaN;
end

fprintf('train_num: %d, val_num: %d \n', numel(train_id), numel(val_id));
       
[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      'train',train_id,...
                       'val',val_id,...
                       'loss', opts.loss,...
                      opts.train) ;
time = toc;
fprintf(fid,'map: %f,map11: %f, time: %fs\n',info.val.error(:,end),time);
% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.train.expDir, 'net-deployed.mat');

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end
% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
% bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;
bopts.edge_model = opts.edge_model;

switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;
  case 'dagnn'
    fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;
end

% -------------------------------------------------------------------------
function [im,labels] = getSimpleNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = getPFDLDLbatch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'flip', true) ;
else
  % validation: disable data augmentation
  im = getPFDLDLbatch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'flip', false) ;
end

if nargout > 0
   labels.label = imdb.images.labels(:,batch) ;
   labels.label_ld = imdb.images.labels_ld(:,batch) ;
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_hcp_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_hcp_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  labels = imdb.images.label(batch) ;
  inputs = {'input', im, 'label', imdb.images.label(batch)} ;
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = [find(imdb.images.set == 1), find(imdb.images.set == 2)];
train = train(1: 1: end);
bs = 256 ;
opts.networkType = 'simplenn' ;
fn = getBatchFn(opts, meta) ;
avg = {}; rgbm1 = {}; rgbm2 = {};

for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
  temp = fn(imdb, batch) ;
  z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{end+1} = mean(temp, 4) ;
  rgbm1{end+1} = sum(z,2)/n ;
  rgbm2{end+1} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
