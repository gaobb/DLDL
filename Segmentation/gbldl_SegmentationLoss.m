classdef gbldl_SegmentationLoss < dagnn.Loss
    
  methods
    function outputs = forward(obj, inputs, params)
      groundtruth = inputs{2};
      groundtruth = groundtruth(:, :, 1:21,:);

      t =  groundtruth.* log(inputs{1}); % KL
      Y =  -sum(t, 3) ; 
      [zeros_value, ~] = max(groundtruth, [], 3);
      instanceWeights = single(zeros_value(:,:,1,:) ~= 0) ;
      max_class = max(groundtruth, [], 3);
      mass = sum(sum(max_class > 0,2),1) + 1 ;
      instanceWeights = bsxfun(@times, instanceWeights, 1./mass) ;
      outputs{1} = instanceWeights(:)' * Y(:);
%       mass = sum(sum(inputs{2} > 0,2),1) + 1 ;   
%       outputs{1} = LDL_vl_nnloss(inputs{1}, inputs{2}, [], ...
%                              'loss', obj.loss, ...
%                              'instanceWeights', 1./mass) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      groundtruth = inputs{2};
      groundtruth = groundtruth(:, :, 1:21,:);
      [zeros_value, ~] = max(groundtruth, [], 3);
      instanceWeights = single(zeros_value(:,:,1,:) ~= 0) ;
      max_class = max(groundtruth, [], 3);
      mass = sum(sum(max_class > 0,2),1) + 1 ;
      instanceWeights = bsxfun(@times, instanceWeights, 1./mass) ;
      dzdy = derOutputs{1};
      dzdy = dzdy * instanceWeights;
      dzdy = repmat(dzdy, 1, 1, 21, 1);
      derInputs{1} = -1./inputs{1}.*(dzdy.*groundtruth); %
      
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = gbldl_SegmentationLoss(varargin)
      obj.load(varargin) ;
    end
  end
end