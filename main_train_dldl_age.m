run(fullfile('/home/gaobb/mywork/toolbox/matconvnet-1.0-beta18', 'matlab', 'vl_setupnn.m')) ;
addpath(genpath('./Age-Pose'))
% Compile the MatConvNet toolbox
% vl_compilenn('EnableGpu', false); % cpu
% vl_compilenn('EnableGpu', true);  % gpu
% vl_compilenn('enableGpu', true, ...
%                'cudaRoot', '/usr/local/cuda-7.5', ...
%                'cudaMethod', 'nvcc', ...
%                'enableCudnn', true, ...
%                'cudnnRoot', '/usr/local/cuda-7.5-cudnn-v5') ;

%% training from scratch: izfnet
opts.dataset = 'MORPH_Album2';
opts.dataset = 'ChaLearn';
opts.dataDir = './datasets';

loss = {'l1loss','klloss','l2loss','softmaxloss','ls-klloss','hellingerloss','epsilonloss',};


opts.modelType = 'izfnet' ;
opts.fine_tune = false;


opts.modelType = 'vggnet' ;
opts.fine_tune = true;
for l = 1:6
    opts.loss = loss{1,l};
    opts.networkType = 'simplenn' ;
    sfx = opts.modelType ;
    for sigma = 2
        opts.sigma = sigma;
        dldl_age('dataset',opts.dataset,...
            'dataDir', opts.dataDir,...
            'loss',opts.loss,...
            'fine_tune',opts.fine_tune,...
            'modelType',opts.modelType,...
            'sigma',opts.sigma,...
            'gpus_id',11);
    end
end