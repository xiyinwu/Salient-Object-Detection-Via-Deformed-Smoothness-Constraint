function [ObjectMap,sup_saliency1,sup_saliency] = ObjSal(im,bbssel,spnum,inds)
% objectness score for each superpixel
[w,h,~]=size(im); 
ObjectMap=zeros(w,h);
bbssel(:,5) = (bbssel(:,5)-min(bbssel(:,5)))/(max(bbssel(:,5))-min(bbssel(:,5)));
for idx = 1:size(bbssel,1)
    xmin = uint16(bbssel(idx,1));
    ymin = uint16(bbssel(idx,2));
    xmax = uint16(bbssel(idx,3));
    ymax = uint16(bbssel(idx,4));
    score = bbssel(idx,5);
    temp=zeros(w,h);
    temp(ymin:ymax,xmin:xmax)=score;
    ObjectMap=ObjectMap+temp;
end
ObjectMap = mat2gray(ObjectMap);

% foreground mask map
% thresh = 0.8*mean(ObjectMap(:));
% ObjectMap(ObjectMap>=thresh)=1;

% find negative instance
sup_saliency = zeros(spnum,1);  
for i=1:spnum
    sup_saliency(i) = mean(ObjectMap(inds{i}));
end
neg_thresh = 0.4; 
sup_saliency1=sup_saliency;
sup_saliency1(sup_saliency<= neg_thresh)=0;