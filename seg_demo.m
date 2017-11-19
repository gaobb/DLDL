%% Deep label distribution learning for semantic segmentation
addpath('./utils')
run('./External/matconvnet-1.0-beta18/matlab/vl_setupnn.m');
opts.modelPath = 'DLDLModel/dldl8s.mat';
opts.modelFamily = 'matconvnet' ;

% experiment setup
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = true ;
opts.vocAdditionalSegmentationsMergeMode = 2 ;
opts.gpus = [11] ;

if ~isempty(opts.gpus)
    gpuDevice(opts.gpus(1))
end

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------
% Get PASCAL VOC 11/12 segmentation dataset plus Berkeley's additional
% segmentations
imdb.classes.id = uint8(1:20) ;
imdb.classes.name = {...
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', ...
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', ...
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'} ;
imdb.classes.images = cell(1,20) ;


% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net) ;
net.mode = 'test' ;
for name = {'objective', 'accuracy'}
    net.removeLayer(name) ;
end
net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
predVar = net.getVarIndex('prediction') ;
% predVar = net.getVarIndex('softMax_prediction') ;  % add to crf

inputVar = 'input' ;
imageNeedsToBeMultiple = true ;
if ~isempty(opts.gpus)
    gpuDevice(opts.gpus(1)) ;
    net.move('gpu') ;
end
net.mode = 'test' ;

% -------------------------------------------------------------------------
% Test
% -------------------------------------------------------------------------
img_name = '2007_001311.jpg';
img_path = fullfile('./data/voc11', img_name);


% Load an image and gt segmentation
rgb = vl_imreadjpeg({img_path}) ;
rgb = rgb{1} ;

imt = single(rgb);


if imageNeedsToBeMultiple
    sz = [size(imt,1), size(imt,2)] ;
    sz_ = round(sz / 32)*32 ;
    im = imresize(imt, sz_) ;
else
    im = imt ;
end
% Subtract the mean (color)
im = bsxfun(@minus, imt, net.meta.normalization.averageImage) ;
% Soome networks requires the image to be a multiple of 32 pixels
if imageNeedsToBeMultiple
    sz = [size(im,1), size(im,2)] ;
    sz_ = round(sz / 32)*32 ;
    im_ = imresize(im, sz_) ;
else
    im_ = im ;
end

if ~isempty(opts.gpus)
    im_ = gpuArray(im_) ;
end

net.eval({inputVar, im_}) ;
scores_ = gather(net.vars(predVar).value) ;
[~,pred_] = max(scores_,[],3) ;
sm_scores_ = vl_nnsoftmax(scores_); % softmax

if imageNeedsToBeMultiple
    pred = imresize(pred_, sz, 'method', 'nearest') ;
else
    pred = pred_ ;
end

% Print segmentation
figure(1) ;clf ;
drawnow ;
image(rgb/255) ;
axis image ;
axis off
title('source image') ;


N=21;
cmap = zeros(N,3);
for i=1:N
    id = i-1; r=0;g=0;b=0;
    for j=0:7
        r = bitor(r, bitshift(bitget(id,1),7 - j));
        g = bitor(g, bitshift(bitget(id,2),7 - j));
        b = bitor(b, bitshift(bitget(id,3),7 - j));
        id = bitshift(id,-3);
    end
    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
figure(2)
image(uint8(pred-1)) ;
axis image ;
title('predicted') ;
colormap(cmap) ;
axis off

