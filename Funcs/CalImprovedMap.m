function saliencymap = CalImprovedMap(psal, PrI_sal, PrI_bg, PrO_sal, PrO_bg, In_Ind, Out_Ind)
%%
% Calculate Improved Saliency Map Using Bayesian Model
psal = mynormalize(double(psal), 0, 1);
psal_I = psal(In_Ind);
psal_O = psal(Out_Ind);
Pr_I = (PrI_sal .* psal_I) ./ (PrI_sal .* psal_I + PrI_bg .* (1 - psal_I));
Pr_O = (PrO_sal .* psal_O) ./ (PrO_sal .* psal_O + PrO_bg .* (1-psal_O));

saliencymap = 0 * psal;
saliencymap(In_Ind) = Pr_I;
saliencymap(Out_Ind) = Pr_O;
saliencymap = (saliencymap - min(saliencymap(:)))/(max(saliencymap(:)) - min(saliencymap(:)));

end