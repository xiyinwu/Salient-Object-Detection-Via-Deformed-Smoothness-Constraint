function adjcMatrixF = LinkNN(adjcMatrix,cluIdx)
adjcMatrixF=adjcMatrix; 
for i=1:length(cluIdx)
    adjcMatrixF(cluIdx{i},cluIdx{i})=1;
end
adjcMatrixF(adjcMatrixF>1)=1;
spNum = size(adjcMatrix, 1);
adjcMatrixF(1:spNum+1:end) = 0;