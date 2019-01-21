function Map = SaveSaliencyMap(feaVec, pixelList, frameRecord, imgName, doNormalize, fill_value, saveMap)
% Fill back super-pixel values to image pixels and save into .png images

if (~iscell(pixelList))
    error('pixelList should be a cell');
end

if (~ischar(imgName))
    error('imgName should be a string');
end

if (nargin < 5)
    doNormalize = true;
end

if (nargin < 6)
    fill_value = 0;
end

if (nargin < 7)
    saveMap = 1;
end
h = frameRecord(1);
w = frameRecord(2);

top = frameRecord(3);
bot = frameRecord(4);
left = frameRecord(5);
right = frameRecord(6);

partialH = bot - top + 1;
partialW = right - left + 1;
partialImg = CreateImageFromSPs(feaVec, pixelList, partialH, partialW, doNormalize);


if partialH ~= h || partialW ~= w
    feaImg = ones(h, w) * fill_value;
    feaImg(top:bot, left:right) = partialImg;
    Map = feaImg;
else
    Map = partialImg;
end
    if saveMap == true
        Map=imgaussfilt(Map);
        imwrite(Map, imgName);
    end