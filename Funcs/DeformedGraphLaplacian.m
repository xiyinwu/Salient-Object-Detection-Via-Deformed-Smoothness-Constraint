function [sal_ref,sal_temp] = DeformedGraphLaplacian(adjcMatrix, idxImg, bdIds, cluIdx,colDistM, CentralDis,obj_saliency)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Salient Object Detection Via Deformed Smoothness Constraint
% 
% Code Author: Xiyin Wu
% Email: xiyinwu1990@gmail.com
% Date: 21/01/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% reference paper:
% [1] Xiyin Wu, Xiaodi Ma, Jinxia Zhang, Andong Wang, Zhong Jin.
% Salient Object Detection Via Deformed Smoothness Constraint. 
% 25th IEEE International Conference on Image Processing (ICIP). 
% 2018: 2815-2819.


%% parameter setting
alpha=0.99;
beta=0.5;
theta=10;
spNum = size(adjcMatrix, 1);

%% Construct Super-Pixel Graph
adjcMatrix_nn = LinkNNAndBoundary2(adjcMatrix, bdIds);  % add edges by Rule1 and Rule 2
adjcMatrix_nn = LinkNN(adjcMatrix_nn,cluIdx);  % % add edges by Rule 3

W = SetSmoothnessMatrix(colDistM,CentralDis,adjcMatrix_nn,theta);  % compute edge weight matrix

D = diag(sum(W));
V = sum(sum(W));
I = eye(spNum);
optAff =(D-alpha*W+beta*(I-D/V))\eye(spNum);
optAff(1:spNum+1:end) = 0;  %set diagonal elements to be zero

%% Label propagation based on deformed smoothness
% top
Yt=zeros(spNum,1);
bst=unique(idxImg(1, :));
Yt(bst)=1;
bsalt=optAff*Yt;
bsalt=(bsalt-min(bsalt(:)))/(max(bsalt(:))-min(bsalt(:)));
bsalt=1-bsalt;
% bottom
Yb=zeros(spNum,1);
bsb=unique(idxImg(end, :));
Yb(bsb)=1;
bsalb=optAff*Yb;
bsalb=(bsalb-min(bsalb(:)))/(max(bsalb(:))-min(bsalb(:)));
bsalb=1-bsalb;
% left
Yl=zeros(spNum,1);
bsl=unique(idxImg(:, 1));
Yl(bsl)=1;
bsall=optAff*Yl;
bsall=(bsall-min(bsall(:)))/(max(bsall(:))-min(bsall(:)));
bsall=1-bsall;
% right
Yr=zeros(spNum,1);
bsr=unique(idxImg(:, end));
Yr(bsr)=1;
bsalr=optAff*Yr;
bsalr=(bsalr-min(bsalr(:)))/(max(bsalr(:))-min(bsalr(:)));
bsalr=1-bsalr;
% combine
sal_temp=(bsalt.*bsalb.*bsall.*bsalr);
Dist2=GetDistanceMatrix(sal_temp); 
sal_temp=optAff*((sal_temp-min(sal_temp(:)))/(max(sal_temp(:))-min(sal_temp(:))));

%% Map Refinement based on deformed smoothness
adjcMatrix_n2 = LinkNN(adjcMatrix,cluIdx);
W2 = SetSmoothnessMatrixNew(Dist2,adjcMatrix_n2, theta); % new weight matrix computed by Mc
D2 = diag(sum(W2));
V2 = sum(sum(W2));
D_obj = diag(exp(-obj_saliency));
sal_ref = (0.5*(I-D2/V2)+D2-W2+D_obj)\sal_temp;  % map refinment result

function W = SetSmoothnessMatrix(colDistM,CentralDis,adjcMatrix_nn, theta)
allDists = colDistM(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);

colDistM(adjcMatrix_nn == 0) = Inf;
colDistM = (colDistM - minVal) / (maxVal - minVal + eps);

allDists = CentralDis(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);
CentralDis(adjcMatrix_nn == 0) = Inf;
CentralDis = (CentralDis - minVal) / (maxVal - minVal + eps);

W = exp(-(0.55*colDistM+0.45*CentralDis) * theta); 

function W = SetSmoothnessMatrixNew(DistM,adjcMatrix_nn, theta)
allDists = DistM(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);

DistM(adjcMatrix_nn == 0) = Inf;
DistM = (DistM - minVal) / (maxVal - minVal + eps);

W = exp(-DistM * theta); 

function adjcMatrix = LinkNNAndBoundary2(adjcMatrix, bdIds)
%link boundary SPs
adjcMatrix(bdIds, bdIds) = 1;

%link neighbor's neighbor
adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
adjcMatrix = double(adjcMatrix);

spNum = size(adjcMatrix, 1);
adjcMatrix(1:spNum+1:end) = 0;  %diagnal elements set to be zero