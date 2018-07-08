function dldl_age(varargin)
% CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%   This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%   VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.
opts.dataset = 'MORPH_Album2';
opts.modelType = 'izfnet' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
opts.loss = 'klloss';
opts.fine_tune = false;
opts.data_type = 'age';
opts.sigma = 2;
opts.sample_num = [];
opts.gpus_id = [];
opts.label_set = [];
opts.dataDir = '';
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.dataDir = fullfile(opts.dataDir,opts.dataset) ;
opts.numFetchThreads = 12;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end

if ~isempty(opts.sample_num), sfx = [sfx '-sample-', num2str(opts.sample_num)];end
if ~isempty(opts.label_set)
    ld_num = numel(opts.label_set);
    sfx = [sfx, '-out_num-', num2str(ld_num)];
else
    opts.label_set = 1:85;
    ld_num = numel(opts.label_set);
end

switch opts.dataset
    case  'MORPH_Album2'
        num_fold = 10;%
    otherwise
        num_fold = 1;
end

opts.expDir = fullfile('./training-models', [opts.dataset,sprintf('-%s-%s', ...
    sfx, opts.networkType)]) ;
mkdir(opts.expDir);
diary =  fullfile(opts.expDir,[opts.dataset,opts.loss, '.txt']);
fid = fopen(diary,'a');
fprintf(fid,'dataset: %s  loss: %s\n',opts.dataset,opts.loss);
fprintf(fid,'        |  mae | ChaLearnError | time |\n');
for k_fold =1:num_fold
     fprintf(fid,'flod-%d:',k_fold);
    fclose(fid);
    
    tic;
    [opts, varargin] = vl_argparse(opts, varargin) ;
    
    opts.numFetchThreads = 12 ;
    opts.lite = false ;
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
    switch opts.dataset
        case {'ChaLearn'}
            opts.train.batchSize = 32 ; %128
        case 'MORPH_Album2'
            opts.train.batchSize = 128 ; %64  finetune
    end
    opts.train.numSubBatches = 1 ;
    opts.train.continue = true ;
    opts.train.gpus = opts.gpus_id  ;
    opts.train.prefetch = true ;
    opts.train.sync = false ;
    opts.train.cudnn = true ;
    % opts.train.expDir = opts.expDir ;
    if opts.fine_tune == false
        opts.train.expDir = fullfile(opts.expDir , opts.loss,['fold',num2str(k_fold)]);
    else
        opts.train.expDir = fullfile(opts.expDir ,[opts.loss,'vgg_fine_tune'],['fold',num2str(k_fold)]);
    end
    
    switch opts.dataset
        case  'MORPH_Album2'
            temp = logspace(-2, -4, 60) ;
            opts.train.learningRate = temp(1:20);
        otherwise
            if ld_num >= 169
                temp = logspace(-3, -4, 60) ;% for chalearn softmax
            else
                temp = logspace(-2, -4, 60) ;% for chalearn softmax
            end
            opts.train.learningRate = temp(1:20);
    end
    
    [opts, varargin] = vl_argparse(opts, varargin) ;
    
    opts.train.numEpochs = numel(opts.train.learningRate) ;
    opts = vl_argparse(opts, varargin) ;
    
    % -------------------------------------------------------------------------
    %                                                   Database initialization
    % -------------------------------------------------------------------------
    
    if exist(opts.imdbPath)
        imdbstruct = load(opts.imdbPath) ;
        imdb = imdbstruct.IMDB{1,k_fold};
    else
        switch opts.dataset
            case 'MORPH_Album2'
                IMDB = setup_morph('dataDir', opts.dataDir, 'lite', opts.lite) ;
            case 'ChaLearn'
                IMDB = setup_chalearn('dataDir', opts.dataDir, 'lite', opts.lite, 'sample_num', opts.sample_num) ;
        end
        mkdir(opts.expDir) ;
        save(opts.imdbPath, 'IMDB') ;
        imdb = IMDB{1,1};
    end
    switch opts.dataset
        case 'MORPH_Album2'
            imdb.images.label(2,:) = opts.sigma;
        otherwise
    end
    
    label = imdb.images.label;
    switch opts.loss
        case 'softmaxloss'
            imdb.images.label=  label(1,:);
            imdb.images.delta = label(2,:);
            out_dim = 85;
        case {'l2loss','l1loss','tukeyloss','epsilonloss'}
            [cordin,ps] = mapminmax([1:85],-1,1);
            % cvpr16
            imdb.images.label = cordin(min(max(label(1,:),1),85));
            imdb.images.delta = label(2,:);
            out_dim = 1;
        case {'klloss','hellingerloss'}
            label = imdb.images.label;
            img_num = numel(label(1,:));
            img_label = zeros(ld_num, img_num);
            if opts.sigma == 0
                pos = [label(1,:); 1: numel(label(1,:))]';
                pos  = (pos(:,2)-1)*85+pos(:,1);
                img_label(pos) = 1;
            else
                dif_age =  bsxfun(@minus,opts.label_set',repmat(label(1,:),ld_num,1));
                switch opts.dataset
                    case 'MORPH_Album2'
                        img_label  = 1./(sqrt(2*pi)*opts.sigma).*exp(-dif_age.^2./(2*opts.sigma.^2));
                    case 'ChaLearn'
                        sigma = label(2,:);
                        img_label =1./repmat(sqrt(2*pi)*sigma,ld_num,1).*...
                            exp(-(bsxfun(@minus,opts.label_set',repmat(label(1,:),ld_num,1))).^2./repmat(2*sigma.^2,ld_num,1));
                end
            end
            imdb.images.label=  max(min(img_label,1-10^-15),10^-15);
            imdb.images.delta = label(2,:);
            out_dim = 85;
        case 'ls-klloss'
            q =85;
            img_label = zeros(q,numel(imdb.images.name));
            for i = 1:numel(imdb.images.name)
                img_label(label(1,i),i)  = 1;
            end
            
            imdb.images.label= img_label*(1-0.1)+0.1/85;
            imdb.images.delta = label(2,:);
            out_dim = 85;
            
    end
    % -------------------------------------------------------------------------
    %                                                    Network initialization
    % -------------------------------------------------------------------------
    if opts.fine_tune == false
        net = dldl_init('model', opts.modelType, ...
            'batchNormalization', opts.batchNormalization, ...
            'weightInitMethod', opts.weightInitMethod,...
            'loss',opts.loss,...
            'out_dim', out_dim);
    else
        load('./VGG-Face-model/vgg_face.mat');
        vl_simplenn_display(net)
        net.normalization = rmfield(net.normalization,'averageImage');
        opts.train.learningRate = logspace(-3, -5, 10);
        net.layers(end-2:end) =[];
        net.layers{end+1} = struct('type', 'conv', 'name', 'fc8', ...
            'weights', {{randn(1, 1, 4096, out_dim, 'single')*0.01, zeros(out_dim, 1, 'single')}}, ...
            'stride', 1, ...
            'pad', 0, ...
            'learningRate', [10 20], ...
            'weightDecay', [0.005  0]) ;
    
             
        switch opts.loss
            case {'klloss','ls-klloss'}
                net.layers{end+1} = struct('type', 'softmax', 'name', 'softmax') ;
                net.layers{end+1} = struct('type', 'klloss', 'name', 'loss') ;
            case 'helingerloss'
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
    net.normalization.border = 224 - net.normalization.imageSize(1:2) ;
    net.normalization.interpolation = 'bilinear';
    bopts = net.normalization ;
    bopts.numThreads = opts.numFetchThreads ;
    bopts.loss = opts.loss;
    opts.train.numEpochs = numel(opts.train.learningRate) ;
    
    % compute image statistics (mean, RGB covariances etc)
    %     imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
    imageStatsPath = fullfile(opts.train.expDir, 'imageStats.mat') ;
    if ~exist(opts.train.expDir, 'dir')
        mkdir(opts.train.expDir);
    end
    if exist(imageStatsPath,'file')
        load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    else
        [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
        save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    end
    
    % One can use the average RGB value, or use a different average for
    % each pixel
    % net.normalization.averageImage = averageImage ;
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
    
    switch lower(opts.networkType)
        case 'simplenn'
            fn = getBatchSimpleNNWrapper(bopts) ;
            [net,info] = dldl_age_train(net, imdb, fn, opts.train, ...
                                          'conserveMemory', true,...
                                          'data_type',opts.data_type,...
                                          'label_set', opts.label_set,...
                                          'loss',opts.loss) ;
        case 'dagnn'
            fn = getBatchDagNNWrapper(bopts, useGpu) ;
            opts.train = rmfield(opts.train, {'sync', 'cudnn'}) ;
            info = cnn_train_dag(net, imdb, fn, opts.train) ;
    end
    time = toc;
    fid = fopen(diary,'a');
    fprintf(fid,'sigma-%f:  %.4f  %.4f  %.4f s\n',opts.sigma,info.val.error(1:2,end),time);
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
im = age_get_batch(images, opts, ...
    'prefetch', nargout == 0) ;%
labels.label = imdb.images.label(:,batch) ;
labels.delta = imdb.images.delta(:,batch) ;


% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_age_get_batch(images, opts, ...
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
    fprintf('collecting image stats: batch starting with image %d ...',batch(1)) ;
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
