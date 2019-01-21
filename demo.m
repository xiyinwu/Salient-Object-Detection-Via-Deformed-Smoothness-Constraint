%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is for [1], and can only be used for non-comercial purpose. If
% you use our code, please cite [1].
% 
% Code Author: Xiyin Wu
% Email: xiyinwu1990@gmail.com
% Date: 21/01/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% demo.m shows how to use [1].

% [1] Xiyin Wu, Xiaodi Ma, Jinxia Zhang, Andong Wang, Zhong Jin.
% Salient Object Detection Via Deformed Smoothness Constraint. 
% 25th IEEE International Conference on Image Processing (ICIP). 
% 2018: 2815-2819.

%% start
clear; clc;
addpath(genpath('Funcs'));
addpath(genpath('Feat'));

%% 1. Parameter Settings
doFrameRemoving = true;
useSP = true;           %You can set useSP = false to use regular grid for speed consideration

SRC = '.\img';     %Path of input images
RES = '.\result';     %Path for saving saliency maps
srcSuffix = '.jpg';     %suffix of input image

if ~exist(RES, 'dir')
    mkdir(RES);
end

%% 2. Saliency Map Calculation
files = dir(fullfile(SRC, strcat('*', srcSuffix)));
for k=1:length(files)
    disp(k);
    srcName = files(k).name;
    noSuffixName = srcName(1:end-length(srcSuffix));
    %% Pre-Processing: Remove Image Frames
    srcImg = imread(fullfile(SRC, srcName));
    if doFrameRemoving
        [noFrameImg, frameRecord] = removeframe(srcImg, 'sobel');
        [h, w, chn] = size(noFrameImg);
    else
        noFrameImg = srcImg;
        [h, w, chn] = size(noFrameImg);
        frameRecord = [h, w, 1, h, 1, w];
    end
    if chn==1
        img = zeros(h,w,3);
        for i=1:3
            img(:,:,i) = noFrameImg;
        end
        noFrameImg = uint8(img);
        [rows, cols, chn] = size(noFrameImg);
    end
    %% Segment input rgb image into patches (SP/Grid)
    pixNumInSP = 600;                           % pixels in each superpixel
    spnumber = round( h * w / pixNumInSP );     % super-pixel number for current image
    
    if useSP
        [idxImg, adjcMatrix, pixelList] = SLIC_Split(noFrameImg, spnumber);
    else
        [idxImg, adjcMatrix, pixelList] = Grid_Split(noFrameImg, spnumber);        
    end
    
    %% Get super-pixel properties
    spNum = size(adjcMatrix, 1);  
    meanRgbCol = GetMeanColor(noFrameImg, pixelList);
    meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255); 
    meanPos = GetNormedMeanPos(pixelList, h, w);
    bdIds = GetBndPatchIds(idxImg,2);   % Bounndary nodes index
    colDistM = GetDistanceMatrix(meanLabCol);
    colDistM_nom = (colDistM-min(min(colDistM)))/(max(max(colDistM))-min(min(colDistM))+eps);
    posDistM = GetDistanceMatrix(meanPos);
    CentralDis = SineDis(meanPos, spNum); % geometrical distance in sine space of edge weight matrix W
    [cluIdx,uniIdx]=apclustering(colDistM_nom); % use APC to calculate the clusters of nodes
    
    %% Generate object proposals
    [model,opts] = ParEdgeBox;  % parameter of EdgeBox
    bbs = edgeBoxes(noFrameImg,model,opts);
    
    %% Proposals selecting
    [clipVal, geoSigma] = EstimateDynamicParas(adjcMatrix, colDistM); 
    bgProb = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma);
    bbssel = ProposalSel(noFrameImg,bbs,1-bgProb,idxImg,pixelList);
     [ObjectMap,sup_saliency1,obj_saliency] = ObjSal(noFrameImg,bbssel,spNum,pixelList);

    %% Salient object detection via Deformed Graph Laplacian
    sal = DeformedGraphLaplacian(adjcMatrix, idxImg, bdIds, cluIdx,colDistM,CentralDis,obj_saliency); 
    
    smapName=fullfile(RES, strcat(noSuffixName, '_DGL.png'));   % save result
    Map=SaveSaliencyMap(sal, pixelList, frameRecord, smapName, true);
end