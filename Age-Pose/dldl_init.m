function net = dldl_init(varargin)
% CNN_IMAGENET_INIT  Initialize a standard CNN for ImageNet

opts.scale = 1 ;
opts.initBias = 0.1 ;
% opts.weightDecay = 0.005  ; %v1
opts.weightDecay = 1  ;           %v2
%opts.weightInitMethod = 'xavierimproved' ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false ;
opts.loss = 'softmaxloss';
opts.out_dim = 85;
opts = vl_argparse(opts, varargin) ;

% Define layers
switch opts.model
    case 'alexnet'
        net.normalization.imageSize = [227, 227, 3] ;
        net = alexnet(net, opts) ;
    case 'zfnet'
        net.normalization.imageSize = [224, 224, 3] ;
        net = zfnet(net, opts) ;
    case  'izfnet'
        net.normalization.imageSize = [224, 224, 3] ;
        net = izfnet(net, opts) ;
    case 'vgg-f'
        net.normalization.imageSize = [224, 224, 3] ;
        net = vgg_f(net, opts) ;
    case 'vgg-m'
        net.normalization.imageSize = [224, 224, 3] ;
        net = vgg_m(net, opts) ;
    case 'vgg-s'
        net.normalization.imageSize = [224, 224, 3] ;
        net = vgg_s(net, opts) ;
    case 'vgg-vd-16'
        net.normalization.imageSize = [224, 224, 3] ;
        net = vgg_vd(net, opts) ;
    case 'vgg-vd-19'
        net.normalization.imageSize = [224, 224, 3] ;
        net = vgg_vd(net, opts) ;
    otherwise
        error('Unknown model ''%s''', opts.model) ;
end

% final touches
switch lower(opts.weightInitMethod)
  case {'xavier', 'xavierimproved'}
    net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end
% net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

switch opts.loss
    case {'klloss','ls-klloss'}
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
        net.layers{end+1} = struct('type', 'l1loss', 'name','loss');
    case 'epsilonloss'
        net.layers{end+1} = struct('type', 'tanh', 'name', 'tanh') ;
        net.layers{end+1} = struct('type', 'epsilonloss', 'name','loss');     
     otherwise
        net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
end

net.normalization.border = 224 - net.normalization.imageSize(1:2) ;
net.normalization.interpolation = 'bicubic' ;
net.normalization.averageImage = [] ;
net.normalization.keepAspect = true ;
 
% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0]) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single')}}, ...
                             'learningRate', [2 1], ...
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


% --------------------------------------------------------------------
function net = alexnet(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;

net = add_block(net, opts, '1', 11, 11, 3, 96, 4, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '2', 5, 5, 48, 256, 1, 2) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '3', 3, 3, 256, 384, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 192, 384, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 192, 256, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 256, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, 85, 1, 0) ;% 85 and 93 for age and pose, respectively.
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end
% --------------------------------------------------------------------
function net = zfnet(net, opts)
% --------------------------------------------------------------------
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

net = add_block(net, opts, '8', 1, 1, 4096, opts.out_dim, 1, 0) ;% 85 and 93 for age and pose, respectively.
net.layers(end) = [] ;
% if opts.batchNormalization, net.layers(end) = [] ; end

function net = izfnet(net, opts) % improve zf-net
% --------------------------------------------------------------------
% switch opts.data_type
%     case 'age'
%         switch opts.loss
%             case {'l2loss','l1loss','epsilonloss'}
%                 out_dim = 1;
%             otherwise
%         out_dim = opts.out_dim;
%         end
%     case 'pose'
%         switch opts.loss
%             case {'l2loss','l1loss','epsilonloss'}
%                 out_dim = 2;
%             case 'klgloss'
%                 out_dim = 3;
%             otherwise
%         out_dim = opts.out_dim;
%         end
% end
out_dim = opts.out_dim;
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
% if opts.batchNormalization, net.layers(end) = [] ; end


function net = fcnzfnet(net, opts) % improve zf-net
% --------------------------------------------------------------------
switch opts.data_type
    case 'age'
        switch opts.loss
            case {'l2loss','l1loss','tukeyloss','epsilonloss'}
                out_dim = 1;
            otherwise
        out_dim = 85;
        end
    case 'pose'
        switch opts.loss
            case {'l2loss','l1loss','tukeyloss','epsilonloss'}
                out_dim = 2;
            case 'klgloss'
                out_dim = 3;
            otherwise
        out_dim = 93;
        end
    case 'pose3'
        switch opts.loss
            case {'l2loss','l1loss','tukeyloss','epsilonloss'}
                out_dim = 3;
            otherwise
        out_dim = 93;
        end    
end

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
net = add_block(net, opts, '5', 3, 3, 384, 512, 1, 1) ;

net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 1, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 512, 1024, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'avg', ...
                           'pool', [6 6], ...
                           'stride', 1, ...
                           'pad', 0) ;

% 
% net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
% net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 1024, out_dim, 1, 0) ;% 85 and 93 for age and pose, respectively.
net.layers(end) = [] ;
% if opts.batchNormalization, net.layers(end) = [] ; end


function net = izf_snet(net, opts) % improve zf-net
% --------------------------------------------------------------------
switch opts.data_type
    case 'age'
        switch opts.loss
            case {'l2loss','l1loss','tukeyloss','epsilonloss'}
                out_dim = 1;
            otherwise
        out_dim = 85;
        end
    case 'pose'
        switch opts.loss
            case {'l2loss','l1loss','tukeyloss','epsilonloss'}
                out_dim = 2;
            otherwise
        out_dim = 3721;
        end
    case 'pose3'
        switch opts.loss
            case {'l2loss','l1loss','tukeyloss','epsilonloss'}
                out_dim = 2;
            otherwise
                out_dim = 93;
        end
end

net.layers = {} ;

net = add_block(net, opts, '1', 7, 7, 3, 48, 2, 1) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;


net = add_block(net, opts, '2', 5, 5, 48, 128, 2, 0) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;


net = add_block(net, opts, '3', 3, 3, 128, 192, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 192, 192, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 192, 128, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 128, 2048, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 2048, 2048, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 2048, out_dim, 1, 0) ;% 85 and 93 for age and pose, respectively.
net.layers(end) = [] ;
% if opts.batchNormalization, net.layers(end) = [] ; end



% --------------------------------------------------------------------
function net = vgg_s(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1', 7, 7, 3, 96, 2, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 3, ...
                           'pad', [0 2 0 2]) ;

net = add_block(net, opts, '2', 5, 5, 96, 256, 1, 0) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

net = add_block(net, opts, '3', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 512, 512, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 3, ...
                           'pad', [0 1 0 1]) ;

net = add_block(net, opts, '6', 6, 6, 512, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

% --------------------------------------------------------------------
function net = vgg_m(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1', 7, 7, 3, 96, 2, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '2', 5, 5, 96, 256, 2, 1) ;
net = add_norm(net, opts, '2') ;
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

net = add_block(net, opts, '6', 6, 6, 512, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

% --------------------------------------------------------------------
function net = vgg_f(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1', 11, 11, 3, 64, 4, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

net = add_block(net, opts, '2', 5, 5, 64, 256, 1, 2) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '3', 3, 3, 256, 256, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 256, 256, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 256, 256, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 256, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

% --------------------------------------------------------------------
function net = vgg_vd(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1_1', 3, 3, 3, 64, 1, 1) ;
net = add_block(net, opts, '1_2', 3, 3, 64, 64, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '2_1', 3, 3, 64, 128, 1, 1) ;
net = add_block(net, opts, '2_2', 3, 3, 128, 128, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '3_1', 3, 3, 128, 256, 1, 1) ;
net = add_block(net, opts, '3_2', 3, 3, 256, 256, 1, 1) ;
net = add_block(net, opts, '3_3', 3, 3, 256, 256, 1, 1) ;
if strcmp(opts.model, 'vgg-vd-19')
  net = add_block(net, opts, '3_4', 3, 3, 256, 256, 1, 1) ;
end
net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '4_1', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '4_2', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '4_3', 3, 3, 512, 512, 1, 1) ;
if strcmp(opts.model, 'vgg-vd-19')
  net = add_block(net, opts, '4_4', 3, 3, 512, 512, 1, 1) ;
end
net.layers{end+1} = struct('type', 'pool', 'name', 'pool4', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '5_1', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5_2', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5_3', 3, 3, 512, 512, 1, 1) ;
if strcmp(opts.model, 'vgg-vd-19')
  net = add_block(net, opts, '5_4', 3, 3, 512, 512, 1, 1) ;
end
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 7, 7, 512, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end