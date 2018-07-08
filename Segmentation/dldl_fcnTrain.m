function dldl_fcnTrain(varargin)
%FNCTRAIN Train FCN model using MatConvNet

% run matconvnet/matlab/vl_setupnn ;
% addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = 'training-models/fl_conv_ldl_nols_fcn32s-m0.9-voc11' ;
% opts.expDir = 'models/fl_conv_ldl_nols_fcn16s-m0.9-voc11' ;
% opts.expDir = 'models/vl_conv_ldl_nols_fcn32s-m0.9-voc11' ;
% opts.expDir = 'models/vl_conv_ldl_nols_fcn16s-m0.9-voc11' ;
% opts.expDir = 'models/vl_conv_ldl_nols_fcn16s-m0.9-voc11' ;
% opts.expDir = 'models/vvl_conv_ldl_nols_fcn16s-m0.9-voc11-33' ;
% opts.expDir = 'models/vfl_conv_ldl_nols_fcn16s-m0.9-voc11-33' ;
% opts.expDir = 'models/vffl_conv_ldl_nols_fcn8s-m0.9-voc11-33' ;

opts.dataDir = '/home/gaobb/mywork/SV3/image_data/VOC2011_ALL/VOCdevkit/VOC2011' ;

opts.modelType = 'fcn32s' ;
% opts.modelType = 'fcn16s' ;
% opts.modelType = 'fcn8s' ;

% opts.sourceModelPath = 'data/models/imagenet-vgg-verydeep-16.mat' ;
opts.sourceModelPath = '/home/gaobb/mywork/vgg_net/imagenet-vgg-verydeep-16.mat' ;
% opts.sourceModelPath = 'models/fl_conv_ldl_nols_fcn32s-m0.9-voc11/net-epoch-50.mat' ;
% opts.sourceModelPath = 'models/vl_conv_ldl_nols_fcn32s-m0.9-voc11/net-epoch-50.mat' ;
% opts.sourceModelPath = 'models/vl_conv_ldl_nols_fcn32s-m0.9-voc11/net-epoch-33.mat' ;
% opts.sourceModelPath = 'models/vfl_conv_ldl_nols_fcn16s-m0.9-voc11-33/net-epoch-27.mat' ;

[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = true ;

opts.numFetchThreads = 1 ; % not used yet

% training options (SGD)
% opts.train = struct([]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.batchSize = 20 ;
opts.train.numSubBatches = 20 ;
opts.train.continue = true ;
opts.train.gpus = [16] ;
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 0.0001 * ones(1,50) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Get PASCAL VOC 12 segmentation dataset plus Berkeley's additional
% segmentations/home/gaobb/mywork/SV3/image_data/PASCAL
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
%   imdb.paths.classSegmentation = '/home/gaobb/mywork/SV3/image_data/PASCAL/VOC2011_ALL/VOCdevkit/VOC2011/SegmentationClassExt/%s.png';
%   imdb.paths.image = '/home/gaobb/mywork/SV3/image_data/PASCAL/VOC2011_ALL/VOCdevkit/VOC2011/JPEGImages/%s.jpg';
%   imdb.paths.classSegmentation_ld = '/home/gaobb/mywork/SV3/image_data/PASCAL/VOC2011_ALL/VOCdevkit/VOC2011/SegmentationClassExt_LD/%s.mat';

else
  imdb = vocSetup('dataDir', opts.dataDir, ...
    'edition', opts.vocEdition, ...
    'includeTest', false, ...
    'includeSegmentation', true, ...
    'includeDetection', false) ;
  if opts.vocAdditionalSegmentations
    imdb = vocSetupAdditionalSegmentations(imdb, 'dataDir', opts.dataDir) ;
  end
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end


% Get training and test/validation subsets
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;

% Get dataset statistics
if exist(opts.imdbStatsPath)
  stats = load(opts.imdbStatsPath) ;
else
  stats = getDatasetStatistics(imdb) ;
  save(opts.imdbStatsPath, '-struct', 'stats') ;
end

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
% Get initial model from VGG-VD-16
if any(strcmp(opts.modelType, {'fcn32s'}))
   net = dldl_fcnInitializeModel('sourceModelPath', opts.sourceModelPath) ;
end

if any(strcmp(opts.modelType, {'fcn16s'}))
   model_FCN = load(opts.sourceModelPath) ;
   net = dagnn.DagNN.loadobj(model_FCN.net);
  % upgrade model to FCN16s
   net = dldl_fcnInitializeModel16s(net) ;
end

if strcmp(opts.modelType, 'fcn8s')
   model_FCN = load(opts.sourceModelPath) ;
   net = dagnn.DagNN.loadobj(model_FCN.net);
  % upgrade model fto FCN8s
  net = dldl_fcnInitializeModel8s(net) ;
end

net.meta.normalization.rgbMean = stats.rgbMean ;
net.meta.classes = imdb.classes.name ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup data fetching options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,21,'single') ;
bopts.rgbMean = stats.rgbMean ;
% opts.train.gpus = opts.train.gpus;
bopts.useGpu = numel(opts.train.gpus) > 0 ;

% Launch SGD
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), ...
                     opts.train, ....
                     'train', train, ...
                     'val', val, ...
                     'momentum',0.9,...
                     opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) dldl_getBatch(imdb,batch,opts,'prefetch',nargout==0) ;
