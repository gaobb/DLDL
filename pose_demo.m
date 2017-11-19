%% Deep Label Distribution Learning for Head Pose Estimation
addpath('./utils')
run('./External/matconvnet-1.0-beta18/matlab/vl_setupnn.m');
%% load pre-trained model
% dldl_izfnet_aflw_1.mat can be downloaded at 
modelPath = './DLDLModel/dldl_izfnet_aflw_1.mat';
load(modelPath, 'net') ;

clear vl_tmove vl_imreadjpeg ;
opts.gpus = 12;
gpuDevice(opts.gpus)
net = dldl_simplenn_move(net, 'gpu');
rgbMean = net.meta.normalization.averageImage;

%% load image
img_name = '2-image06575_jpg_7788.jpg';
img_path = fullfile('./data/aflw_images', img_name);
imt = imread(img_path);
im = imresize(imt, [224,224]);

data = bsxfun(@minus, single(imresize(im, net.meta.normalization.imageSize(1:2))),...
    reshape(net.meta.normalization.averageImage, [1,1,3]));
%% forward 
res = dldl_simplenn(net, gpuArray(data), [], [], ...
            'accumulate', 0, ...
            'mode', 'test', ...
            'conserveMemory', 1) ;
pred_score = squeeze(gather(res(end).x));
[~,mu] = max(pred_score);
cordin_yaw = repmat(-90:3:90,61,1);
cordin_pitch = repmat([90:-3:-90]',1,61);
cordin = [reshape(cordin_yaw,[],1),reshape(cordin_pitch,[],1)];
pre_pose = cordin(mu',:);
%% visialization
figure(1)
imshow(imt)

figure(2)
ribbon(reshape(pred_score,61,61))
%imagesc(reshape(pred_score,61,61))
colormap jet
set(gca,'XTick',1:6:61);
set(gca,'XTickLabel',{'-90',' -72',' -54',...
    ' -36',   '-18',   ' 0',  '18',  '  36', ...
    ' 54  ', '72  ', '  90'});
set(gca,'YTick',1:6:61);
set(gca,'YTickLabel',{'90',' 72',  ' 54', ...
    ' 36',  '18',  ' 0',   '-18',  '-36', ...
    '-54  ', '-72  ','-90'});
xlabel('Yaw','color','blue','FontSize',12);
ylabel('Pitch','color','blue','FontSize',12);
zlabel('Probability','color','blue','FontSize',12);
axis([1 61 1 61 0 max(pred_score).*(1+0.01)])
set(gca,'FontSize',12)
% print(gcf,'-depsc',['figure/aflw_ld',img_name,'.eps'])
