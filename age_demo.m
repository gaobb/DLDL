%% Deep label distribution learning for age estimation
% add the MatConvNet toolbox to MATLAB path
addpath('./utils')
run('./External/matconvnet-1.0-beta18/matlab/vl_setupnn.m');
%% load pre-trained model
% dldl_izfnet_chalearn_1.mat is avlaible at https://pan.baidu.com/s/1eSKWELO
modelPath = './DLDLModel/dldl_izfnet_chalearn_1.mat';
load(modelPath, 'net') ;

clear vl_tmove vl_imreadjpeg ;
opts.gpus = 12;
gpuDevice(opts.gpus)
net = dldl_simplenn_move(net, 'gpu');

rgbMean = net.meta.normalization.averageImage;

%% load image
img_name = 'image_525.jpg';
img_path = fullfile('./data/chalearn', img_name);
imt = imread(img_path);
im = imresize(imt, [224,224]);

data = bsxfun(@minus, single(imresize(im, net.meta.normalization.imageSize(1:2))),...
    reshape(net.meta.normalization.averageImage, [1,1,3]));
data(:,:,:,2) = data(:,end:-1:1,:);
%% forward 
res = dldl_simplenn(net, gpuArray(data), [], [], ...
            'accumulate', 0, ...
            'mode', 'test', ...
            'conserveMemory', 1) ;
pred_score = squeeze(mean(gather(res(end).x),4));
pred_age = (1:85)*pred_score;

%% visialization
close all
figure(1)
imshow(imt)
figure(2)
plot(pred_score,'--ro','LineWidth',2,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',5)
grid on
axis([0 85 0 0.12])
ylabel('Probability','color','blue','FontSize',12)
xlabel('Age','color','blue','FontSize',12)
set(gca,'YTick',0:0.01:max(pred_score));
set(gca,'XTick',0:10:85);
set(gca,'FontSize',10)
axis([1, 85 0 max(max(pred_score), max(pred_score))])
grid on
title(sprintf('PredAge: %.2f', pred_age))
% print(gcf,'-depsc',['figure/chalearn15',img_name,'.eps'])
