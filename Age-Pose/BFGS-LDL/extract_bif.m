
addpath('./bif_xiaodong_wang')

load('bifpara.mat')

opts.dataset = 'MORPH_Album2';
opts.dataset = 'ChaLearn';
% opts.dataset = 'lamda_6';
opts.fine_tune = false;
% opts.sigma = 2;
% loss = {'hellingerloss','epsilonloss','l1loss','klloss','l2loss','softmaxloss'};
% opts.loss = loss{4};
opts.modelType = 'izfnet' ;
opts.networkType = 'simplenn' ;
sfx = opts.modelType ;

opts.expDir = fullfile('model_v2', [opts.dataset,sprintf('-%s-%s', ...
        sfx, opts.networkType)]) ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
load(opts.imdbPath)
imdb = IMDB{1,1};



val = find(imdb.images.set ==2);
for i =1 :numel(val)
    img_path = fullfile(imdb.imageDir,imdb.images.name{val(i)});
    
    im = imread(img_path);
    % imshow(im)
    im = rgb2gray(im);
    im = imresize(im, [60, 60]);
    val_feat(i,:) = bif(double(im), para);
end
save([opts.expDir ,'/bfgs_ldl/bif_feat_chalearn_val.mat'],'val_feat');

train = 1:2476;
for i =1 :numel(train)
    img_path = fullfile(imdb.imageDir,imdb.images.name{train(i)});
    fprintf('%s\n',img_path);
    im = imread(img_path);
    % imshow(im)
    im = rgb2gray(im);
    im = imresize(im, [60, 60]);
    train_feat(i,:) = bif(double(im), para);
end

save([opts.expDir,'/bfgs_ldl/bif_feat_chalearn_train.mat'],'fea');


load([opts.expDir ,'/bfgs_ldl/bif_feat_chalearn_val.mat']);

num = size(raw,1);



mu = mean(feats);
feats = feats - repmat(mu,size(feats,1),1); % center all features

[temp , ~, latent] = princomp(feats);
score = cumsum(latent)/sum(latent);

PCA_dim = min(find(score>=opt.PCA_energy));
codes.lf = temp(:,1:opt.PCA_dim); % PCA load factors

fprintf('# PCA features reserved energy: %.2f%% with %d features\n',score(opt.PCA_dim)*100,opt.PCA_dim);

temp = (feats*codes.lf)'; % do PCA to the features