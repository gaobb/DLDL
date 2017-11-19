function ld = genAgeLd(label, sigma, loss)
% genernate label distribution
label_set = 1:85;
switch loss
    case 'klloss'
        ld_num = length(label_set);
        
        dif_age =  bsxfun(@minus,label_set',repmat(label,ld_num,1));
        
        ld = 1./repmat(sqrt(2*pi)*sigma,ld_num,1).*exp(-(dif_age).^2./repmat(2*sigma.^2,ld_num,1));
        
        ld = bsxfun(@times, ld, 1./sum(ld));
    case 'smloss'
        ld = round(label);
    case {'l1', 'l2'}
        ld = 2/84.*(label-1) -1;
end