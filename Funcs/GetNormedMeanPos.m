function [meanPos,Area] = GetNormedMeanPos(pixelList, height, width)
% averaged x(y) coordinates of each superpixel, normalized with respect to
% image dimension
% return N*2 vector, row i is superpixel i's coordinate [y x]

spNum = length(pixelList);
meanPos = zeros(spNum, 2);
Area = zeros(spNum, 1);

for n = 1 : spNum
    [rows, cols] = ind2sub([height, width], pixelList{n});    
    meanPos(n,1) = mean(rows) / height;
    meanPos(n,2) = mean(cols) / width;
    Area(n)=length(pixelList{n});
end