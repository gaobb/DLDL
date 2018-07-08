
addpath(genpath('Segmentation'))
% DLDL-32S
opts.expDir = 'training-models/fcn32s-voc11' ;
opts.dataDir = '/mnt/data3/gaobb/image_data/PASCAL/VOC2011_ALL/VOCdevkit/VOC2011' ;
opts.modelType = 'fcn32s' ;
opts.sourceModelPath = './Pre-trainedModels/imagenet-vgg-verydeep-16.mat' ;

dldl_fcnTrain('expDir', opts.expDir,...
         'dataDir', opts.dataDir,...
          'modelType', opts.modelType,...
          'sourceModelPath', opts.sourceModelPath);
% DLDL-16S
opts.expDir = 'training-models/fcn32s-voc11' ;
opts.dataDir = '/mnt/data3/gaobb/image_data/PASCAL/VOC2011_ALL/VOCdevkit/VOC2011' ;
opts.modelType = 'fcn16s' ;
opts.sourceModelPath = '/mnt/data3/gaobb/PaperCode/ORI_DLDL_TIP17/DLDL_v2.0/matconvnet-fcn/models/vl_conv_ldl_nols_fcn32s-m0.9-voc11/net-epoch-50.mat' ;

dldl_fcnTrain('expDir', opts.expDir,...
         'dataDir', opts.dataDir,...
          'modelType', opts.modelType,...
          'sourceModelPath', opts.sourceModelPath);
% DLDL-8S
opts.expDir = 'training-models/fcn8s-voc11' ;
opts.dataDir = '/mnt/data3/gaobb/image_data/PASCAL/VOC2011_ALL/VOCdevkit/VOC2011' ;
opts.modelType = 'fcn8s' ;
opts.sourceModelPath = '/mnt/data3/gaobb/PaperCode/ORI_DLDL_TIP17/DLDL_v2.0/matconvnet-fcn/models/vfl_conv_ldl_nols_fcn16s-m0.9-voc11-33/net-epoch-27.mat' ;

dldl_fcnTrain('expDir', opts.expDir,...
         'dataDir', opts.dataDir,...
          'modelType', opts.modelType,...
          'sourceModelPath', opts.sourceModelPath);
