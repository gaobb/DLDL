%% Deep Label Distribution Learning for Multi-label Classification
% add the MatConvNet toolbox to MATLAB path
run(fullfile(fileparts(mfilename('fullpath')),...
  './External/matconvnet-1.0-beta18', 'matlab', 'vl_setupnn.m')) ;
addpath('./External/edges'); 
addpath(genpath('./External/Piotr-CV'));
addpath(genpath('./External/Ncut_9'));
addpath('./utils')
gpu_id = 12;
% load pre-trained edge detection model
model=load('./External/edges/models/forest/modelBsds');
model=model.model;
model.opts.multiscale=0;
model.opts.sharpen=2; 
model.opts.nThreads=4;
edge_model = model;

% load PFDLDL model
modelPath = sprintf('./SimModel/pfdldl_%s_%s_%s', 'vgg16','max','voc07');
fprintf('load model from %s\n', modelPath);
tic;
load(modelPath, 'net', 'info');
toc;
for l = 1:numel(net.layers)
    if strcmp(net.layers{l}.name, 'hcp')
        break;
    end
end
net.layers(l:end) = [];
if ~isempty(gpu_id)
    gpuDevice(gpu_id);
    net = vl_simplenn_move(net, 'gpu') ;
end
vl_simplenn_display(net);


rgbmean = net.meta.normalization.averageImage;
tic;
% load image
img_name = '2007_001311.jpg';
img_path = fullfile('./data/voc11', img_name);
im = imread(img_path);

im_patch = genImgProposal(im, edge_model, rgbmean);
patch_size = size(im_patch, 4);
batch_size = 250;
out_score = zeros(20, patch_size);
for t = 1:batch_size:patch_size
    batch = t:min(t+batch_size-1, patch_size);
    input_data = im_patch(:,:,:,batch);
    
    if ~isempty(gpu_id)
        input_data = gpuArray(input_data) ;
    end
    res = dldl_simplenn(net, input_data, [], [], ...
        'accumulate', 0, ...
        'mode', 'test', ...
        'conserveMemory', true, ...
        'backPropDepth', 0, ...
        'sync', false, ...
        'cudnn', true) ;
    out_score(:,batch) = squeeze(gather(res(end).x));
end
max_score = max(out_score, [], 2);
pro_score = bsxfun(@times, exp(max_score), 1./sum(exp(max_score)));

%% visialization
classname  = net.meta.classes.name;
[prob, id] = sort(pro_score, 'descend');
figure(1)
imshow(im)
strs = sprintf('Top1: %s, %.2f Top2: %s, %.2f Top3: %s, %.2f',...
    classname{id(1)}, prob(1),classname{id(2)}, prob(2),classname{id(3)}, prob(3));
title(strs, 'color', 'r');
figure(2)
bar(pro_score)
colormap([255 165 0]./255)
grid on
xlim([1 20])
set(gca,'XTick',1:1:20);
set(gca,'XTickLabel',[]);
set(gca,'ycolor','black','FontSize',10)
set(gca,'xcolor','black','FontSize',10)
for i=1:20
    text(i,-0.005,classname(i),'color','blue','HorizontalAlignment','right','rotation',45,'FontSize',12);
end
ylabel('Probability','color','blue','FontSize',12)
grid on

