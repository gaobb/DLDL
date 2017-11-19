function ld = genPoseLd(label, sigma, cordin, loss)
% genernate label distribution
switch loss
    case 'klloss'
        for i =1 : size(label, 2)
            diff = bsxfun(@minus, label(:,i)' , cordin);
            t = zeros(size(cordin,1),1);
            if sigma ~= 0
                t = exp(-0.5*(diff(:,1).*diff(:,1)+diff(:,2).*diff(:,2))./sigma^2);
            else
                t(imdb.images.label(i)) = 1;
            end
            ld(:,i) = t./sum(t);
        end
    case 'smloss'
        
    case {'l1', 'l2'}
end