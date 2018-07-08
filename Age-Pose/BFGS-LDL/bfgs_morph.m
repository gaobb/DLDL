%BFGSLLDDEMO	The example of BFGSLLD algorithm.
%
%	Description
%   In order to optimize the IIS-LLD algorithm, we follow the idea of an
%   effective quasi-Newton method BFGS to further improve IIS-LLD. 
%   Here is an example of BFGSLLD algorithm.
%	
%	See also
%	LLDPREDICT, BFGSLLDTRAIN
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
clear;
clc;
% Load the trainData and bfgslldTestData.
imdbstruct = load('model_v2/MORPH_Album2-izfnet-simplenn/imdb.mat') ;

exp_path = 'model_v2/MORPH_Album2-izfnet-simplenn/bfgs_ldl';

imdb = imdbstruct.IMDB{1,1};
imdb.imageDir = '/home/gaobb/mywork/SV3/image_data/age_faces/MORPH_Album2_align';
% load(fullfile(root_path,'mappedX.mat'))
load('xinchao/morph_bif.mat')

% map_id = zeros(1,55134);
% for i =1:numel(imdb.images.name)
%     for j = 1:numel(morph.name)
%         if strcmp(imdb.images.name{i},morph.name{j})
%             map_id(i) = j;
%            break;           
%         end
%     end
%     fprintf('%d: %s == %d: %s \n',i,imdb.images.name{i},j,morph.name{j});
% end
load(fullfile(exp_path,'map_id.mat'));

find(map_id==0)
labels = imdb.images.label;
labels(:,16) = [];
map_id(16) = [];
find(map_id==0)
labels(:,65) = [];
map_id(65) = [];
% labels(1,:) - morph.age(map_id)
all_feat = morph.feat_dr(map_id',:);

% feat = morph.feat;
% label = morph.age;
% q =85;
% sigma = 2;
% age_range = [1:q]';
% img_label = zeros(q,numel(label(1,:)));
% dif_age =  bsxfun(@minus,age_range,repmat(label(1,:),q,1));
% %     switch opts.dataset
% %         case 'MORPH_Album2'
% img_label  = 1./(sqrt(2*pi)*sigma).*exp(-dif_age.^2./(2*sigma.^2));
% %         case 'ChaLearn'
% %             img_label =1./repmat(sqrt(2*pi)*label(2,:),q,1).*...
% %                 exp(-(bsxfun(@minus,age_range,repmat(label(1,:),q,1))).^2./repmat(2*label(2,:).^2,q,1));
% %     end
% 
% img_ld =  max(min(img_label,1-10^-15),10^-15);
% indices = crossvalind('Kfold',label(1,:),10);

% imdb.images.set(sel_train') = 1;
% imdb.images.set(sel_val') = 2;
val_bif = testFeature;
val_label =  imdb.images.label(1,val_id);

for fold = 1:10
    imdb = imdbstruct.IMDB{1,fold};
    imdb.images.set(16) = [];
    imdb.images.set(65) = [];
    imdb.images.label(:,16) =[];
    imdb.images.label(:,65) = [];
    
    label = imdb.images.label;
    q =85;
    sigma = 2;
    age_range = [1:q]';
    img_label = zeros(q,numel(label(1,:)));
    dif_age =  bsxfun(@minus,age_range,repmat(label(1,:),q,1));
    %     switch opts.dataset
    %         case 'MORPH_Album2'
    img_label  = 1./(sqrt(2*pi)*sigma).*exp(-dif_age.^2./(2*sigma.^2));
    %         case 'ChaLearn'
    %             img_label =1./repmat(sqrt(2*pi)*label(2,:),q,1).*...
    %                 exp(-(bsxfun(@minus,age_range,repmat(label(1,:),q,1))).^2./repmat(2*label(2,:).^2,q,1));
    %     end
   
    img_ld =  max(min(img_label,1-10^-15),10^-15);
    
    
     train_id = find(imdb.images.set == 1);
     val_id = find(imdb.images.set ==2);
%     val_id = (indices ==fold)';
%     train_id = ~val_id;


    trainFeature = all_feat(train_id',:);
    trainDistribution = img_ld(:,train_id)';
    
    testFeature = all_feat(val_id',:);
    testDistribution = img_ld(:,val_id)';
    save('trainData.mat','trainFeature','trainDistribution');

    % Initialize the model parameters.
    prediction=[];
    geneNum = size(testFeature,1);
    
    item=eye(size(trainFeature,2),size(trainDistribution,2));
    
    % The training part of BFGSLLD algorithm.
    % The function of bfgsprocess provides a target function and the gradient.
    [weights,fval] = bfgslldtrain(@bfgsprocess,item);
 
    distribution = lldpredict(weights,testFeature);
    
    [~,pred_age_max] = max(transpose(distribution));
    pred_age_ep = (distribution*[1:85]')';
    
    
    
    % evaluation
    
    clear real_age
    real_age = label(1,val_id);
    dif_age_max  = abs(pred_age_max-real_age);
    dif_age_ep  = abs(pred_age_ep-real_age);
    
    err_max = mean(dif_age_max);
    %     err(2,:) = mean(1-exp(-dif_age.^2./(2.*(imdb.images.label(2,val).^2))));
    sum_acc_max = zeros(100,1);
    for theta = 1:100
        sum_acc_max(theta,1)  = mean(dif_age_max(1,:) <=theta);
    end
    sum_acc_max = sum_acc_max.*100;
    
    err_ep = mean(dif_age_ep);
    %     err(2,:) = mean(1-exp(-dif_age.^2./(2.*(imdb.images.label(2,val).^2))));
    sum_acc_ep = zeros(100,1);
    for theta = 1:100
        sum_acc_ep(theta,1)  = mean(dif_age_ep(1,:) <=theta);
    end
    sum_acc_ep = sum_acc_ep.*100;
    
    
    fprintf('MAE(MAX):%.2f, MAE(EP):%.2f\n',err_max,err_ep);
    
    
    model_path = fullfile(exp_path,['fold',num2str(fold)]);
    mkdir(model_path)
    save(fullfile(model_path,'weights.mat'),'weights');
    save(fullfile(model_path,'results.mat'),'distribution','sum_acc_ep','sum_acc_max','err_ep','err_max');
    
    result{fold}.Mae = [err_ep,err_max];
    result{fold}.acc = [sum_acc_ep,sum_acc_max];
end

for i=1:10
    mae_ep(i,:) = result{1,i}.Mae(1);
    mae_max(i,:) = result{1,i}.Mae(2);
    acc_ep(i,:) = result{1,i}.acc(:,1)';
    acc_max(i,:) = result{1,i}.acc(:,2)';
end

fprintf('mae(ep):%.2f+-%.2f,mae(max):%.2f+-%.2f \n',mean(mae_ep),std(mae_ep), mean(mae_max),std(mae_ep));
save(['model_v2/MORPH_Album2-izfnet-simplenn/bfgs_ldl/','result.mat'],'result')






%% chalearn
addpath('/home/gaobb/SoftToolBox/LDLPackage_2.0')
clc;
clear;
% Load the trainData and bfgslldTestData.
imdbstruct = load('model_v2/ChaLearn-izfnet-simplenn/imdb.mat') ;

exp_path = 'model_v2/ChaLearn-izfnet-simplenn/bfgs_ldl';

imdb = imdbstruct.IMDB{1,1};
% imdb.imageDir = '/home/gaobb/mywork/SV3/image_data/age_faces/MORPH_Album2_align';
% load(fullfile(root_path,'mappedX.mat'))
% load('xinchao/morph_bif.mat')
load([exp_path,'/bif_feat_chalearn_train.mat']);
trainFeature = fea;

load([exp_path,'/bif_feat_chalearn_val.mat']);
testFeature = fea;

feats = trainFeature;
mu = mean(feats);
va = std(feats);

trainFeature = (trainFeature - repmat(mu,size(trainFeature,1),1))./repmat(va,size(trainFeature,1),1);
% 
% 
testFeature = (testFeature - repmat(mu,size(testFeature,1),1))./repmat(va,size(testFeature,1),1); 

trainFeature = trainFeature - repmat(mu,size(trainFeature,1),1);
testFeature = testFeature - repmat(mu,size(testFeature,1),1);

label = [imdb.images.label(:,1:2476)];
real_age = imdb.images.label(:,end-1135:end);


q =85;
sigma = 2;
age_range = [1:q]';
img_label = zeros(q,numel(label(1,:)));
dif_age =  bsxfun(@minus,age_range,repmat(label(1,:),q,1));

img_label =1./repmat(sqrt(2*pi)*label(2,:),q,1).*...
    exp(-(bsxfun(@minus,age_range,repmat(label(1,:),q,1))).^2./repmat(2*label(2,:).^2,q,1));


img_ld =  max(min(img_label,1-10^-15),10^-15);
trainDistribution = img_ld';


save('trainData.mat','trainFeature','trainDistribution');

% global trainFeature;
% global trainDistribution;

% feats = trainFeature;
% mu = mean(feats);
% feats = feats - repmat(mu,size(feats,1),1); % center all features
% 
% [temp , ~, latent] = princomp(feats);
% score = cumsum(latent)/sum(latent);
% PCA_dim = 200;
% codes.lf = temp(:,1:PCA_dim); % PCA load factors
% fprintf('# PCA features reserved energy: %.2f%% with %d features\n',score(PCA_dim)*100,PCA_dim);
% 
% trainFeature = feats*codes.lf; % do PCA to the features
% testFeature = (testFeature-repmat(mu,size(testFeature,1),1))*codes.lf;

% Initialize the model parameters.
prediction=[];
geneNum = size(testFeature,1);

item=eye(size(trainFeature,2),size(trainDistribution,2));

% The training part of BFGSLLD algorithm.
% The function of bfgsprocess provides a target function and the gradient.
[weights,fval] = bfgslldtrain(@(w) bfgsprocess(w, trainFeature, trainDistribution), item);
distribution = lldpredict(weights,testFeature);



[weights,fval] = bfgslldtrain(@bfgsprocess,item);

distribution = lldpredict(weights,testFeature);


distribution = predictions';
[~,pred_age_max] = max(transpose(distribution));
pred_age_ep = (distribution*[1:85]')';

dif_age_max  = abs(pred_age_max-real_age(1,:));
dif_age_ep  = abs(pred_age_ep-real_age(1,:));

err_max = mean(dif_age_max);
%     err(2,:) = mean(1-exp(-dif_age.^2./(2.*(imdb.images.label(2,val).^2))));
sum_acc_max = zeros(100,1);
for theta = 1:100
    sum_acc_max(theta,1)  = mean(dif_age_max(1,:) <=theta);
end
sum_acc_max = sum_acc_max.*100;

err_ep = mean(dif_age_ep);
%     err(2,:) = mean(1-exp(-dif_age.^2./(2.*(imdb.images.label(2,val).^2))));
sum_acc_ep = zeros(100,1);
for theta = 1:100
    sum_acc_ep(theta,1)  = mean(dif_age_ep(1,:) <=theta);
end
sum_acc_ep = sum_acc_ep.*100;

epsilon_error_max = mean(1-exp(-(pred_age_max-real_age(1,:)).^2./(2.*(real_age(2,:).^2))));
epsilon_error_ep = mean(1-exp(-(pred_age_ep-real_age(1,:)).^2./(2.*(real_age(2,:).^2))));
fprintf('MAE(MAX):%.2f, MAE(EP):%.2f\n',err_max,err_ep);
fprintf('ME(MAX):%.2f, ME(EP):%.2f\n',epsilon_error_max,epsilon_error_ep);

save(fullfile(exp_path,'weights.mat'),'weights');
save(fullfile(exp_path,'results.mat'),'distribution','sum_acc_ep','sum_acc_max','err_ep','err_max','epsilon_error_max','epsilon_error_ep');


% MAE(MAX):7.78, MAE(EP):6.79
% ME(MAX):0.57, ME(EP):0.53

% pca 200
% MAE(MAX):9.55, MAE(EP):10.62
% ME(MAX):0.62, ME(EP):0.71
% center all features

% for i=1:geneNum
%     testfeature = testFeature(i,:);
%     % Use LDL model to predict the distribution.
%     distribution = lldpredict(weights,testfeature);
%     % Show the comparisons between the real distribution and the predicted distribution.
%     dist(1)=kldist(testDistribution(i,:),distribution);
%     dist(2)=euclideandist(testDistribution(i,:),distribution);  
%     dist(3)=sorensendist(testDistribution(i,:),distribution);
%     dist(4)=squaredxdist(testDistribution(i,:),distribution);
%     dist(5)=fidelity(testDistribution(i,:),distribution);
%     dist(6)=intersection(testDistribution(i,:),distribution);
%     % Draw the picture of the real and prediced distribution.
%     drawdistribution(testDistribution(i,:),distribution,dist);
%     sign=input('Press any key to continue:');
% end