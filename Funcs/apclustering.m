function [cluIdx,uniIdx]=apclustering(A,varargin)
% affinity propagation
% reference paper: Frey B J, Dueck D. Clustering by passing messages between data points[J]. science, 2007, 315(5814): 972-976.

N=size(A,1);
A((1:N+1:end))=-1;
A=A(:);
A(A==-1)=[];
order=[];
for i=1:N
    tmp1=zeros(N-1,2);
    tmp1(:,1)=i;
    tmp2=1:N;
    tmp2(i)=[];
    tmp1(:,2)=tmp2;
    order=[order;tmp1];
end
s=[order,-A];
p1=mean(-A);
if strcmp(varargin,'plot')
    [idx,netsim,dpsim,expref]=apcluster(s,p1,'plot');
else
    [idx,netsim,dpsim,expref]=apcluster(s,p1);
end
uniIdx=unique(idx);
cluIdx=cell(1,length(uniIdx));
cluLen=zeros(1,N);
for i=1:length(uniIdx)
    cluIdx{i}=find(idx==uniIdx(i));
    cluLen(cluIdx{i})=length(cluIdx{i});
end
