function backgroundIds = BoundaryAnalysis(colDistM, posDistM, bdIds)
% 1-D saliency analysis for boundary SPs, using  method in the CVPR10
% paper: S.Goferman, L.manor, and A.Tal. Context-aware saliency
% detection. In CVPR, 2010.

spNum = size(colDistM, 1);
neighborNum = round(spNum / 200 * 5);
c = 3;

colDist_bnd = colDistM(bdIds, bdIds);
colDist_bnd(1:length(bdIds) + 1:end) = inf;
posDist_bnd = posDistM(bdIds, bdIds);
cmbDist_bnd = colDist_bnd ./ (1 + c * posDist_bnd);
cmbDist_bnd = sort(cmbDist_bnd, 2, 'ascend');
meanDist_bnd = mean(cmbDist_bnd(:, 1:neighborNum), 2);
minDist_bnd = min(meanDist_bnd);
maxDist_bnd = max(meanDist_bnd);

if (maxDist_bnd - minDist_bnd > 1)
    meanDist_bnd = ( meanDist_bnd - minDist_bnd ) / (maxDist_bnd - minDist_bnd);
    backgroundIds = bdIds(meanDist_bnd <= 0.5);
else
    backgroundIds = bdIds;
end