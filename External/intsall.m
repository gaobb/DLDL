
%% intsall matconvnet
currpath  = pwd;
cd ./matconvnet-1.0-beta18/
addpath matlab
run('vl_setupnn')
% cpu:
% vl_compilenn
% gpu:
% vl_compilenn('enableGpu', true)
% cuda8.0 + cudnnv4:
vl_compilenn( 'enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-8.0', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', 'true', ...
               'cudnnRoot', '/home/gaobb/mywork/toolbox/cudnn-rc4');
           
vl_compilenn( 'enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-8.0', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', 'true', ...
               'cudnnRoot', '/usr/local/cudnn/cudnn-rc4');

% vl_testnn('gpu', true)
% modify 'matlab/src/bits/impl/pooling_gpu.cu(163)' as follows 
% #include <cuda.h>
% #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
% #else
% // CUDA: atomicAdd is not defined for doubles
% ***************************
% #endif
cd ../

%% intsall edges
mex private/edgesDetectMex.cpp -outdir private 
mex private/edgesNmsMex.cpp    -outdir private 
mex private/spDetectMex.cpp    -outdir private 
mex private/edgeBoxesMex.cpp   -outdir private

%% install drtoolbox
run('mexall')

%% install Ncut9
run('compileDir_simple')

