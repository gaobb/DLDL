run(fullfile('/home/gaobb/mywork/toolbox/matconvnet-1.0-beta18', 'matlab', 'vl_setupnn.m')) ;
% Compile the MatConvNet toolbox
% vl_compilenn('EnableGpu', false); % cpu
% vl_compilenn('EnableGpu', true);  % gpu
% vl_compilenn('enableGpu', true, ...
%                'cudaRoot', '/usr/local/cuda-7.5', ...
%                'cudaMethod', 'nvcc', ...
%                'enableCudnn', true, ...
%                'cudnnRoot', '/usr/local/cuda-7.5-cudnn-v5') ;

opts.dataset = 'bjut-3d';
opts.dataset = 'point04';
opts.dataset = 'aflw_det';
loss = {'softmaxloss','l2loss','l1loss','epsilonloss','hellingerloss','klloss'};
opts.modelType = 'izfnet' ;
opts.fine_tune = false;
opts.dataDir = './datasets';
% sigma = 0:0.5:3
for  i  = 1:6
    sigma = 3;
    opts.loss = loss{1,i};
    opts.fine_tune = false;
 
    opts.modelType = 'izfnet' ;
    opts.networkType = 'simplenn' ;
  
    
    sfx = opts.modelType ;
    opts.sigma = sigma;
    dldl_pose('dataset',opts.dataset,...
                'loss',opts.loss,...
                'dataDir', opts.dataDir,...
                'fine_tune',opts.fine_tune,...
                'modelType',opts.modelType,...
                'sigma',opts.sigma,...
                'gpus_id',11);
end
