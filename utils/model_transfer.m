izfnet_path= '/mnt/data3/gaobb/experiment/CVPR16_DLDL/model_v2/ChaLearn-izfnet-simplenn/klloss/fold1/net-epoch-20.mat';
vggacenet_path= '/mnt/data3/gaobb/experiment/CVPR16_DLDL/model_v2/ChaLearn-izfnet-simplenn/kllossvgg_fine_tune_v1/fold1/net-epoch-10.mat';

clear
dataset = 'aflw_det';
dataset = 'point04';
dataset = 'bjut-3d';
dataset = 'MORPH'
dataset = 'MORPH'

root_path = ['/mnt/data3/gaobb/experiment/CVPR16_DLDL/model_v2/', dataset, '-izfnet-simplenn'];
losses = {'softmaxloss','l2loss','l1loss','epsilonloss','ls-klloss','hellingerloss','klloss-3721'};
% losses = {'softmaxloss','l2loss','l1loss','epsilonloss','ls-klloss','hellingerloss','klloss'};
load(fullfile(root_path,'imdb.mat'));
for i = 1:10
%    IMDB{i}.images.label(1,:) = -1*(IMDB{i}.images.label(1,:));
   IMDB{i}.imageDir =  '/mnt/data3/gaobb/image_data/image_faces/age_faces/Morph/MORPH_Album2_align';
end


imdb = imdb.IMDB{1};
% for i = 1:3
%    IMDB{i}.images.label(1,:) = -1*(IMDB{i}.images.label(1,:));
%    IMDB{i}.imageDir =  '/mnt/data3/gaobb/image_data/image_faces/head_pose/aflw_det';
% end
imdb.imageDir = '/mnt/data3/gaobb/image_data/image_faces/head_pose/point04';

for f = 1:10
izfnet_path= fullfile(root_path, 'klloss', ['fold',num2str(f)], '/net-epoch-20.mat');

net = load(izfnet_path);
% net = load(vggacenet_path);

% a = net;
% net = a;
net = vl_simplenn_tidy(net.net);
net = dldl_net_deploy(net);
vl_simplenn_display(net)

b = net;
net = fromSimpleNN(net, 'canonicalNames', true) ;
net = net.saveobj() ;
% save(['./pretrainModel/izfnet_point04_',num2str(f), '.mat'], 'net') ;
% save(['./pretrainModel/izfnet_bjut-3d_',num2str(f), '.mat'], 'net') ;
save(['./pretrainModel/izfnet_morph_',num2str(f), '.mat'], 'net') ;
end


net = vl_simplenn_tidy(net.net);


% net = load(izfnet_path);
net = dldl_net_deploy(net);
vl_simplenn_display(net)
net = dldl_simplenn_move(net, 'cpu');






clear
dataset = 'aflw_det';
dataset = 'point04';
dataset = 'bjut-3d';
dataset = 'morph'
dataset = 'chalearn'

root_path = ['/mnt/data3/gaobb/experiment/CVPR16_DLDL/model_v2/', dataset, '-izfnet-simplenn'];


losses = {'softmaxloss','l2loss','l1loss','epsilonloss','ls-klloss','hellingerloss','klloss-3721'};
% losses = {'softmaxloss','l2loss','l1loss','epsilonloss','ls-klloss','hellingerloss','klloss'};
load(fullfile(root_path,'imdb.mat'));
for i = 1:10
%    IMDB{i}.images.label(1,:) = -1*(IMDB{i}.images.label(1,:));
   IMDB{i}.imageDir =  '/mnt/data3/gaobb/image_data/image_faces/age_faces/Morph/MORPH_Album2_align';
end


imdb = imdb.IMDB{1};
% for i = 1:3
%    IMDB{i}.images.label(1,:) = -1*(IMDB{i}.images.label(1,:));
%    IMDB{i}.imageDir =  '/mnt/data3/gaobb/image_data/image_faces/head_pose/aflw_det';
% end
imdb.imageDir = '/mnt/data3/gaobb/image_data/image_faces/head_pose/point04';

for f = 1:10
% net_path= fullfile(root_path, 'klloss', ['fold',num2str(f)], '/net-epoch-20.mat');
net_path= fullfile(root_path, 'klloss', ['fold',num2str(f)], '/net-epoch-20.mat');

net = load(net_path);
info = net.info;

% star_path = fullfile(root_path, 'klloss', ['fold',num2str(f)], '/imageStats.mat');
star_path = fullfile(root_path, 'klloss', ['fold',num2str(f)], '/imageStats.mat');
imgStats = load(star_path);
% net = load(vggacenet_path);

net = vl_simplenn_tidy(net.net);
% net = load(izfnet_path);
net = dldl_net_deploy(net);
vl_simplenn_display(net)
net = dldl_simplenn_move(net, 'cpu');
save(['./SimModel/dldl_izfnet_',dataset,'_', num2str(f), '.mat'], 'net','info','imgStats') ;
% save(['./SimModel/dldl_vggface_',dataset,'_', num2str(f), '.mat'], 'net','info','imgStats') ;
end
