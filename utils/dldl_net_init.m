function net = dldl_net_init(varargin)
% CNN_IMAGENET_INIT  Initialize a standard CNN for ImageNet

opts.scale = 1 ;
opts.initBias = 0 ;
opts.weightDecay = 1 ;
%opts.weightInitMethod = 'xavierimproved' ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts.loss = 'softmaxklloss';
opts.outdim = 85;
opts.task = 'age';%or 'pose'
opts = vl_argparse(opts, varargin) ;

% Define layers
switch opts.model
    case  'izfnet'
        net.meta.normalization.imageSize = [224, 224, 3] ;
        net = izfnet(net, opts) ;
        bs = 32 ;
    case  'thinizfnet'
        net.meta.normalization.imageSize = [224, 224, 3] ;
        net = thinizfnet(net, opts) ;
        bs = 32 ;
    case 'vggm' 
        net.meta.normalization.imageSize = [224, 224, 3] ;
        net = vgg_m(net, opts);   
        bs = 32 ;
    case 'vgg-face'
        opts.vgg_face_path = '/mnt/data3/gaobb/experiment/deep_semi_age/vgg_face_net/vgg_face.mat';
        net.meta.normalization.imageSize = [224, 224, 3] ;
        net = vgg_face_net(net, opts) ;
        bs = 32 ;
    otherwise
        error('Unknown model ''%s''', opts.model) ;
end

switch opts.model
    case {'vgg-face'}
        lr = logspace(-3, -5, 20) ;
    otherwise
        if ~opts.batchNormalization
            lr = logspace(-2, -4, 20) ;
        else
            lr = logspace(-1, -4, 20) ;
        end
end

% final touches
switch lower(opts.weightInitMethod)
    case {'xavier', 'xavierimproved'}
        net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end
% net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

% Meta parameters
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 224 ;
net.meta.normalization.averageImage = opts.averageImage ;
net.meta.classes.name = opts.classNames ;
% net.meta.classes.description = opts.classDescriptions;
net.meta.augmentation.jitterLocation = false ;
net.meta.augmentation.jitterFlip = false ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
net.meta.augmentation.jitterAspect = [1, 1] ;

net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = bs ;
net.meta.trainOpts.weightDecay = 0.0005 ;

% Fill in default values
net = vl_simplenn_tidy(net) ;
vl_simplenn_display(net);
% Switch to DagNN if requested
switch lower(opts.networkType)
    case 'simplenn'
        % done
    case 'dagnn'
        %net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net = fromSimpleNN(net, 'canonicalNames', false) ;
        net.renameVar(net.layers(1).inputs{1}, 'input') ;
        net.conserveMemory = false;
        outname = char(net.getOutputs());
        net.vars(end).precious = 1;
        
        switch opts.model
            case {'izfnet', 'thinizfnet', 'vgg-face'}
                % Add loss layer
                switch opts.loss
                    case 'smloss'
                        net.addLayer('smloss', ...
                            lib.SMLoss(), ...
                            {outname,  'ld'}, 'smloss') ;
                    case 'klloss'
                        net.addLayer('klloss', ...
                            lib.KLLoss(), ...
                            {outname,  'ld'}, 'klloss') ;
                end
            otherwise
                assert(false) ;
        end
        % Add Error layer
        switch opts.task
            case 'age'
                net.addLayer('AgeError', ...
                    lib.AgeError(), ...
                    {outname,'label','labelset'}, {'Error'}) ;%, 'ExpMae', 'MaxError', 'ExpError'
            case 'pose'
                net.addLayer('AgeError', ...
                    lib.PoseError(), ...
                    {outname,'label','labelset'}, {'Error'}) ;%, 'ExpMae', 'MaxError', 'ExpError'
        end
        net.meta.trainOpts.derOutputs = {opts.loss, 1}; 
end

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
    name = 'fc' ;
else
    name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
    'weights', {{init_weight(opts, h, w, in, out, 'single'), ...
    ones(out, 1, 'single')*opts.initBias}}, ...
    'stride', stride, ...
    'pad', pad, ...
    'dilate', 1, ...
    'learningRate', [1 2], ...
    'weightDecay', [opts.weightDecay 0], ...
    'opts', {convOpts}) ;
if opts.batchNormalization
    net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
        'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), ...
        zeros(out, 2, 'single')}}, ...
        'epsilon', 1e-4, ...
        'learningRate', [2 1 0.1], ...
        'weightDecay', [0 0]) ;
end
% net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
net.layers{end+1} = struct('type', 'prelu', 'name', sprintf('prelu%s',id),...
                            'weights', {{0.25*rand(out, 1, 'single')}},...
                            'learningRate', 1) ;

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
    case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
    case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
    case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
    otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
    net.layers{end+1} = struct('type', 'normalize', ...
        'name', sprintf('norm%s', id), ...
        'param', [5 1 0.0001/5 0.75]) ;
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
    net.layers{end+1} = struct('type', 'dropout', ...
        'name', sprintf('dropout%s', id), ...
        'rate', 0.5) ;
end

function net = vgg_face_net(net, opts) 
out_dim = opts.outdim;

load(opts.vgg_face_path)
net = vl_simplenn_tidy(net);
net.layers(end-1:end) =[]; % remove fc8 and softmax layer

net.layers{end+1} = struct('type', 'conv', 'name', 'fc8', ...
    'weights', {{randn(1, 1, 4096, out_dim, 'single')*0.01, zeros(out_dim, 1, 'single')}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [10 20], ...
    'weightDecay', [0.005  0]) ;
net = vl_simplenn_tidy(net);
vl_simplenn_display(net)


function net = izfnet(net, opts) % improve zf-net
% --------------------------------------------------------------------
out_dim = opts.outdim;

net.layers = {} ;

net = add_block(net, opts, '1', 7, 7, 3, 96, 2, 1) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;


net = add_block(net, opts, '2', 5, 5, 96, 256, 2, 0) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;


net = add_block(net, opts, '3', 3, 3, 256, 384, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 384, 384, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 384, 256, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 256, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, out_dim, 1, 0) ;% 85 and 93 for age and pose, respectively.
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

function net = thinizfnet(net, opts) % improve zf-net
% --------------------------------------------------------------------
out_dim = opts.outdim;

net.layers = {} ;

net = add_block(net, opts, '1', 7, 7, 3, 32, 2, 1) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;


net = add_block(net, opts, '2', 5, 5, 32, 64, 2, 0) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;


net = add_block(net, opts, '3', 3, 3, 64, 128, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 128, 128, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 128, 256, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 256, 512, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 512, 512, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 512, out_dim, 1, 0) ;% 85 and 93 for age and pose, respectively.
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end


% --------------------------------------------------------------------
function net = vgg_m(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1', 7, 7, 3, 96, 2, 0) ;
%net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '2', 5, 5, 96, 256, 2, 1) ;
%net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

net = add_block(net, opts, '3', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 512, 512, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 512, 512, 1, 0) ;
net = add_dropout(net, opts, '6') ;


bottleneck = 1024 ;

net = add_block(net, opts, '7', 1, 1, 512, bottleneck, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, bottleneck, 101, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end