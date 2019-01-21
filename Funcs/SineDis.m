function CentralDis = SineDis(meanPos, spNum)

CentralDis = sin(pi*abs( repmat(meanPos(:,1), [1, spNum]) - repmat(meanPos(:,1)', [spNum, 1]) )).^2 + sin(pi*abs( repmat(meanPos(:,2), [1, spNum]) - repmat(meanPos(:,2)', [spNum, 1]) )).^2;
CentralDis = sqrt(CentralDis);