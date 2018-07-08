function dldl_pose(varargin)
% CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%   This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%   VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

opts.dataset = 'point04';
opts.loss = 'klloss';
opts.fine_tune = false;

opts.modelType = 'izfnet' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
opts.sigma =1;
opts.dataDir = '';
opts.gpus_id = [];
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.dataDir = fullfile(opts.dataDir,opts.dataset) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
% opts.loss = 'jdloss';

opts.expDir = fullfile('./training-models', [opts.dataset,sprintf('-%s-%s', ...
    sfx, opts.networkType)]) ;
mkdir(opts.expDir);
diary =  fullfile(opts.expDir, [opts.dataset,opts.loss,'_diary.txt']);
fid = fopen(diary,'a');
fprintf(fid,'dataset: %s  loss: %s\n',opts.dataset,opts.loss);
fprintf(fid,'        |  pitch  | yaw | pitch+yaw |  pitch |  yaw  | pitch+yaw | time |\n');
fclose(fid);


sigma = opts.sigma;
% indices = crossvalind('KFold',imdb.images.label,20);
% train = find(indices ==1);
% val = find(indices == 10);
%
% imdb.images.name = imdb.images.name([train;val]');
% imdb.images.label = imdb.images.label([train;val]');
% imdb.images.set = [ones(1,numel(train)),2*ones(1,numel(val))];
% imdb.images.id = 1:numel(imdb.images.name);
% opts.imdbPath = 'model_v2/bjut-3d-izfnet-simplenn/val_imdb.mat';
% save(opts.imdbPath,'imdb') ;

for k_fold = 1:5
    tic;
    [opts, varargin] = vl_argparse(opts, varargin) ;
    
    opts.numFetchThreads = 12 ;
    opts.lite = false ;
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
    switch opts.dataset
        case 'point04'
            opts.train.batchSize = 32 ; %128
        case 'bjut-3d'
            opts.train.batchSize = 128 ; %128  finetune
        case 'aflw_det'
            opts.train.batchSize = 128 ; %128  finetune
    end
    
    opts.train.numSubBatches = 1 ;
    opts.train.continue = true ;
    opts.train.gpus = opts.gpus_id ;
    opts.train.prefetch = true ;
    opts.train.sync = false ;
    opts.train.cudnn = true ;
    if opts.fine_tune == true
        opts.train.expDir = fullfile(opts.expDir ,[opts.loss,'vgg_fine_tune'],['fold',num2str(k_fold)]);
    else
        opts.train.expDir = fullfile(opts.expDir ,['ls-',opts.loss],['fold',num2str(k_fold)]);%,'-label',
    end
    [opts, varargin] = vl_argparse(opts, varargin) ;
    switch opts.dataset
        case  'bjut-3d'
            opts.data_type = 'pose';
            temp = logspace(-2, -4, 60) ;
            opts.train.learningRate = temp(1:20);
        case  'aflw_det'
            opts.data_type = 'pose';
            temp = logspace(-2, -4, 60) ;
            opts.train.learningRate = temp(1:20);
        case 'point04'
            opts.data_type = 'pose';
            temp = logspace(-2, -4, 60) ;
            opts.train.learningRate = temp(1:20);
    end
    opts.train.numEpochs = numel(opts.train.learningRate) ;
    opts = vl_argparse(opts, varargin) ;
    
    % -------------------------------------------------------------------------
    %                                                   Database initialization
    % -------------------------------------------------------------------------
    if exist(opts.imdbPath)
        imdbstruct = load(opts.imdbPath) ;
        imdb = imdbstruct.IMDB{1,k_fold};
    else
        mkdir(opts.expDir) ;
        switch opts.dataset
            case 'point04'
                IMDB = setup_point('dataDir', opts.dataDir, 'lite', 0) ;
            case 'bjut-3d'
                IMDB = setup_bjut('dataDir', opts.dataDir, 'lite', 0) ;
            case 'aflw_det'
                IMDB = setup_aflw('dataDir', opts.dataDir, 'lite', 0) ;
        end
        save(opts.imdbPath,'IMDB') ;
        imdb = IMDB{1,1};
    end
    switch opts.loss
        case {'l2loss','l1loss','epsilonloss','tukeyloss'}
            opts.out_dim = 2;
            switch opts.dataset
                case 'point04'
                    cordstruct = load('cordin_angle.mat');
                    cordin = cordstruct.cordin;
                    temp = cordin(imdb.images.label',:)'*15;
                    [label,opts.ps] = mapminmax(temp,-1,1);
                    %                         sigma0 = mapminmax('apply',[15;15],opts.ps);
                    imdb.images.label = label;
                    imdb.images.class =  temp;
                case 'bjut-3d'
                    cordstruct = load('cordin.mat');
                    cordin = cordstruct.cordin;
                    temp = cordin(imdb.images.label',:)'*10;
                    [label,opts.ps] = mapminmax(temp,-1,1);
                    imdb.images.label = label;
                    imdb.images.class =  temp;
                case 'aflw_det'
                    true_label =  round(imdb.images.label(1:2,:));
                    true_label(1,:) = -1*true_label(1,:);
                    [label,opts.ps] = mapminmax(true_label,-1,1);
                    imdb.images.label = label;
                    imdb.images.class =  true_label;
                otherwise
            end
            
        case {'klloss','hellingerloss'}
            switch opts.dataset
                case 'point04'
                    cordstruct = load('cordin_angle.mat');
                    cordin = cordstruct.cordin;
                    true_label = cordin(imdb.images.label',:);
                    imdb.images.class =  true_label'*15;
                    opts.out_dim = 93;
                case 'bjut-3d'
                    cordstruct = load('cordin.mat');
                    cordin = cordstruct.cordin;
                    true_label = cordin(imdb.images.label',:);
                    imdb.images.class =  true_label'*10;
                    opts.out_dim = 93;
                case 'aflw_det'
                    true_label =  round(imdb.images.label(1:2,:));
                    true_label(1,:) = -1*true_label(1,:);
                    opts.out_dim = 3721;
                otherwise
            end
            switch opts.dataset
                case {'point04','bjut-3d'}
                    labels = [];
                    for i =1 : numel(imdb.images.label)
                        
                        dif_cor = bsxfun(@minus, true_label(i,:) , cordin);
                        t = zeros(93,1);
                        if sigma ~= 0
                            t = exp(-0.5*(dif_cor(:,1).*dif_cor(:,1)+dif_cor(:,2).*dif_cor(:,2))./sigma^2);
                        else
                            t(imdb.images.label(i)) = 1;
                        end
                        labels(:,i) = t./sum(t);
                        imdb.images.label =  max(min(labels,1-10^-15),10^-15);
                    end
                case 'aflw_det'
                    cordin_yaw = repmat(-90:3:90,61,1);
                    cordin_pitch = repmat([90:-3:-90]',1,61);
                    cordin = [reshape(cordin_yaw,[],1),reshape(cordin_pitch,[],1)];  
                    
                    for i =1 : numel(true_label(1,:))
                        dif_cor = bsxfun(@minus, true_label(:,i)' ,cordin);
                        if sigma ~= 0
                            %sigma = 2;
                            t = exp(-0.5*(dif_cor(:,1).*dif_cor(:,1)./(2*sigma^2)+dif_cor(:,2).*dif_cor(:,2)./sigma^2));
                        else
                            t(imdb.images.label(i)) = 1;
                        end
                        labels(:,i) = t./sum(t);
                    end
                    imdb.images.label =  max(min(labels,1-10^-15),10^-15);
                    imdb.images.class =  true_label;

            end
        case 'softmaxloss'
            switch opts.dataset 
                case 'point04'
                    cordstruct = load('cordin_angle.mat');
                    cordin = cordstruct.cordin;
                    true_label= cordin(imdb.images.label',:);
                    imdb.images.class =  true_label'*15;
                    opts.out_dim = 93;
                case 'bjut-3d'
                    cordstruct = load('cordin.mat');
                    cordin = cordstruct.cordin;
                    true_label = cordin(imdb.images.label',:);
                    imdb.images.class =  true_label'*10;
                    opts.out_dim = 93;
                case 'aflw_det'
                    true_label =  round(imdb.images.label(1:2,:));
                    true_label(1,:) = -1*true_label(1,:);
                    cordin_yaw = repmat(-90:3:90,61,1);
                    cordin_pitch = repmat([90:-3:-90]',1,61);
                    
                    cordin = [reshape(cordin_yaw,[],1),reshape(cordin_pitch,[],1)];
                   
                    for i = 1:numel(imdb.images.name)
                        res = bsxfun(@minus,true_label(:,i), cordin');
                        [~,id ] = min(res(1,:).^2+res(2,:).^2);
                        label(1,i) = id;
                    end
                    imdb.images.label = label;
                    imdb.images.class =  true_label;
                    opts.out_dim = 3721;
            end
    end
    % -------------------------------------------------------------------------
    %                                                    Network initialization
    % -------------------------------------------------------------------------
    
    if opts.fine_tune == false
        net = dldl_init('model', opts.modelType, ...
            'batchNormalization', opts.batchNormalization, ...
            'weightInitMethod', opts.weightInitMethod,...
            'loss',opts.loss,...
            'out_dim', opts.out_dim) ;
        
    else
        %         load('/home/gaobb/mywork/experiment/Pose_Net/model/bjut-3d-izfnet-simplenn/klloss/fold1/net-epoch-20.mat');
        load('./Pre-trainedModels/vgg_face.mat');
        moveop = @(x) gather(x);
        for l=1:numel(net.layers)
            switch net.layers{l}.type
                case {'prelu'}
                    for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
                        f = char(f) ;
                        if isfield(net.layers{l}, f)
                            net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
                        end
                    end
                    for f = {'weights', 'momentum'}
                        f = char(f) ;
                        if isfield(net.layers{l}, f)
                            for j=1:numel(net.layers{l}.(f))
                                net.layers{l}.(f){j} = moveop(net.layers{l}.(f){j}) ;
                            end
                        end
                    end
                otherwise
                    % nothing to do ?
            end
        end
        net.normalization = rmfield(net.normalization,'averageImage');
        %     opts.train.learningRate = logspace(-3, -5, 10);
        opts.train.learningRate = logspace(-3, -4, 10);
        
        net.layers(end-2:end) =[];
        net = add_block(net, opts, '8', 1, 1, 4096, 93, 1, 0) ;% 85 and 93 for age and pose, respectively.
        net.layers(end) = [] ;
        
        switch opts.loss
            case 'klloss'
                net.layers{end+1} = struct('type', 'softmax', 'name', 'softmax') ;
                net.layers{end+1} = struct('type', 'klloss', 'name', 'loss') ;
            case 'hellingerloss'
                net.layers{end+1} = struct('type', 'softmax', 'name', 'softmax') ;
                net.layers{end+1} = struct('type', 'hellingerloss', 'name', 'loss') ;
            case 'jdloss'
                net.layers{end+1} = struct('type', 'softmax', 'name', 'softmax') ;
                net.layers{end+1} = struct('type', 'jdloss', 'name', 'loss') ;
            case 'l2loss'
                net.layers{end+1} = struct('type', 'tanh', 'name', 'tanh') ;
                net.layers{end+1} = struct('type', 'l2loss', 'name', 'loss') ;
            case 'l1loss'
                net.layers{end+1} = struct('type', 'tanh', 'name', 'tanh') ;
                net.layers{end+1} = struct('type', 'l1loss', 'name', 'loss') ;
            case 'epsilonloss'
                net.layers{end+1} = struct('type', 'tanh', 'name', 'tanh') ;
                net.layers{end+1} = struct('type', 'epsilonloss', 'name', 'loss') ;
            otherwise
                net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
        end
    end
    
    vl_simplenn_display(net);
    bopts = net.normalization ;
    bopts.numThreads = opts.numFetchThreads ;
    bopts.loss = opts.loss;
    % compute image statistics (mean, RGB covariances etc)
    imageStatsPath = fullfile(opts.train.expDir, 'imageStats.mat') ;
    if ~exist(opts.train.expDir)
        mkdir(opts.train.expDir);
    end
    if exist(imageStatsPath)
        load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    else
        [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
        save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    end
    
    % One can use the average RGB value, or use a different average for
    % each pixel
    %net.normalization.averageImage = averageImage ;
    net.normalization.averageImage = rgbMean ;
    
    switch lower(opts.networkType)
        case 'simplenn'
        case 'dagnn'
            net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
            net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
                {'prediction','label'}, 'top1error') ;
        otherwise
            error('Unknown netowrk type ''%s''.', opts.networkType) ;
    end
    
    % -------------------------------------------------------------------------
    %                                               Stochastic gradient descent
    % -------------------------------------------------------------------------
    
    [v,d] = eig(rgbCovariance) ;
    % bopts.transformation = 'stretch' ;
    bopts.transformation = 'none' ;
    bopts.averageImage = rgbMean ;
    bopts.rgbVariance = 0.1*sqrt(d)*v' ;
    useGpu = numel(opts.train.gpus) > 0 ;
    %opts.data_type = 'pose';
    bopts.loss = opts.loss;
    
    switch lower(opts.networkType)
        case 'simplenn'
            fn = getBatchSimpleNNWrapper(bopts) ;
            [net,info] = dldl_pose_train(net, imdb, fn, opts.train, 'conserveMemory', true,'data_type',opts.data_type,'dataset',opts.dataset,'loss',opts.loss) ;
        case 'dagnn'
            fn = getBatchDagNNWrapper(bopts, useGpu) ;
            opts.train = rmfield(opts.train, {'sync', 'cudnn'}) ;
            info = cnn_train_dag(net, imdb, fn, opts.train) ;
    end
    time = toc;
    fid = fopen(diary,'a');
    fprintf(fid,'fold-%d,sigma %.4f:  %.4f   %.4f   %.4f  %.4f   %.4f   %.4f   %.4f s\n',k_fold,sigma,info.val.error(1:3,end)',info.val.error(4:6,end)',time);
    fclose(fid);
    
end

% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = pose_get_batch(images, opts, ...
    'prefetch', nargout == 0) ;
labels.ld = imdb.images.label(:,batch) ;
labels.gt = imdb.images.class(:,batch) ;

% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
    'prefetch', nargout == 0) ;
if nargout > 0
    if useGpu
        im = gpuArray(im) ;
    end
    inputs = {'input', im, 'label', imdb.images.label(batch)} ;
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 512: end);
bs = 256 ;
fn = getBatchSimpleNNWrapper(opts) ;
for t=1:bs:numel(train)
    batch_time = tic ;
    batch = train(t:min(t+bs-1, numel(train))) ;
    fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
    temp = fn(imdb, batch) ;
    z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
    n = size(z,2) ;
    avg{t} = mean(temp, 4) ;
    rgbm1{t} = sum(z,2)/n ;
    rgbm2{t} = z*z'/n ;
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
