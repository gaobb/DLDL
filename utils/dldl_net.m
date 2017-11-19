function [net, info] = dldl_net(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.
opts.dataset = 'chalearn15'; 
opts.dataDir = 'data/Chalearn';

opts.modelType = 'izfnet' ;
opts.network = [] ;
opts.networkType = 'dagnn' ;
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
opts.loss = 'klloss';
opts.gpus = [3];


[opts, varargin] = vl_argparse(opts, varargin) ;
model_rootnn = ['./Models/', opts.dataset];

sfx = [opts.modelType] ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.loss, '-', opts.networkType] ;
opts.expDir = fullfile(model_rootnn, [opts.dataset, '-', sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 30 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.gpus = opts.gpus ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    switch opts.dataset
        case 'ChaLearn'
            imdb = setup_chalearn15('dataDir', opts.dataDir, 'dataset', opts.dataset) ;
            imdb.images.realLabel = imdb.images.label;
            mu = imdb.images.label(1,:);
            sigma = imdb.images.label(2,:);
            imdb.images.ld = genAgeLd(mu, sigma, opts.loss);
            imdb.images.labelset = (1:85)';
        case 'chalearn16'
            imdb = setup_chalearn16('dataDir', opts.dataDir, 'dataset', opts.dataset) ;
            imdb.images.ld = genAgeLd(imdb.images.reaLabel, imdb.images.sigma);
            imdb.images.ld = genAgeLd(mu, sigma, opts.loss);
            imdb.images.labelset =  (1:85)';
        case 'morph'
            imdb = setup_morph('dataDir', opts.dataDir, 'dataset', opts.dataset) ;
            imdb.images.ld = genAgeLd(imdb.images.reaLabel, imdb.images.sigma);
            imdb.images.ld = genAgeLd(mu, sigma, opts.loss);
            imdb.images.labelset =  (1:85)';
        case 'aflw'
            imdb = setup_aflw('dataDir', opts.dataDir, 'dataset', opts.dataset) ;
            imdb = imdb{1};
            real_label =  round(imdb.images.label(1:2,:));
            real_label(1,:) = -1*real_label(1,:);
            
            cordin_yaw = repmat(-90:3:90,61,1);
            cordin_pitch = repmat([90:-3:-90]',1,61);
            cordin = [reshape(cordin_yaw,[],1),reshape(cordin_pitch,[],1)];
            
            imdb.images.realLabel = real_label;
            sigma = 3;
            imdb.images.ld = genPoseLd(real_label, sigma, cordin);
            imdb.images.labelset = cordin;
            for i = 1:size(cordin,1)
                imdb.meta.classes{1,i} = [num2str(cordin(i,1)),'_',num2str(cordin(i,2))];
            end
        case 'point04'
            imdb = setup_point('dataDir', opts.dataDir, 'dataset', opts.dataset) ;
            imdb = imdb{1};
            cordstruct = load('./data/cordin_angle.mat');
            cordin = cordstruct.cordin*15;
            real_label = cordin(imdb.images.label',:)';
            imdb.images.realLabel = real_label;
            sigma = 15;
            imdb.images.ld = genPoseLd(real_label, sigma, cordin);
            imdb.images.labelset = cordin;
        case 'bjut'
            imdb = setup_bjut('dataDir', opts.dataDir, 'dataset', opts.dataset) ;
            imdb = imdb{1};
            cordstruct = load('./data/cordin.mat');
            cordin = cordstruct.cordin*10;
            real_label = cordin(imdb.images.label',:)';
            imdb.images.realLabel =  real_label;
            sigma = 5;
            imdb.images.ld = genPoseLd(real_label, sigma, cordin);
            imdb.images.labelset = cordin;
            for i = 1:size(cordin,1)
                imdb.meta.classes{1,i} = [num2str(cordin(i,1)),'_',num2str(cordin(i,2))];
            end
    end
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
 % train = find((imdb.images.set == 1 & ~isnan(imdb.images.label(1,:)))==1) ;
 train = find((imdb.images.set == 1)) ;
 %train = 1:numel(imdb.images.name);
 images = fullfile(imdb.imageDir, imdb.images.name(train(1:1:end))) ;
 [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
                                                    'imageSize', [224 224], ...
                                                    'numThreads', opts.numFetchThreads, ...
                                                    'gpus', opts.train.gpus) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
[v,d] = eig(rgbCovariance) ;
rgbDeviation = v*sqrt(d) ;
clear v d ;
% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
switch opts.dataset
    case {'ChaLearn', 'chalearn16', 'morph'}
        opts.task = 'age';
    case {'point04', 'aflw', 'bjut'}
        opts.task = 'pose';
    otherwise
        fprintf('unknow dataest...\n');
end
opts.outdim = size(imdb.images.labelset,1);
if isempty(opts.network)
    net = dldl_net_init('model', opts.modelType, ...
        'loss', opts.loss,...
        'task', opts.task,...
        'outdim', opts.outdim, ...
        'batchNormalization', opts.batchNormalization, ...
        'weightInitMethod', opts.weightInitMethod, ...
        'networkType', opts.networkType, ...
        'averageImage', rgbMean, ...
        'classNames', imdb.meta.classes,...
        'colorDeviation', rgbDeviation) ;
    opts.derOutputs = net.meta.trainOpts.derOutputs;
    
else
    net = opts.network ;
    opts.network = [] ;
end
    
    % -------------------------------------------------------------------------
    %                                                                     Learn
    % -------------------------------------------------------------------------
    
    switch opts.networkType
        case 'simplenn', trainFn = @cnn_train ;
        case 'dagnn', trainFn = @dldl_train_dag ;
    end
    
    [net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
        'expDir', opts.expDir, ...
        'derOutputs', opts.derOutputs,...
        net.meta.trainOpts, ...
        opts.train) ;
    
    % -------------------------------------------------------------------------
    %                                                                    Deploy
    % -------------------------------------------------------------------------
    
    net = dldl_net_deploy(net) ;
    modelPath = fullfile(opts.expDir, 'net-deployed.mat');
    
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

if numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
                meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', meta.normalization.cropSize, ...
  'subtractAverage', mu) ;

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
for f = fieldnames(meta.augmentation)'
  f = char(f) ;
  bopts.train.(f) = meta.augmentation.(f) ;
end

fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end

data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.realLabel(:,batch);
  lds = imdb.images.ld(:,batch);
  labelset = imdb.images.labelset;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels, 'ld', lds, 'labelset', labelset} ;
  end
end

