function [Obj,MSE] = SurrogatePredict(Model,Dec)
%SurrogatePredict - Batch prediction of objectives and MSE by Kriging models.
%
%   [Obj,MSE] = SurrogatePredict(Model,Dec) predicts the objective values
%   and the estimated mean squared errors for all rows of Dec. The DACE
%   predictor is called once per objective on the whole matrix (the DACE
%   implementation natively supports batch sites), which is orders of
%   magnitude faster than point-wise loops.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N   = size(Dec,1);
    M   = length(Model);
    Obj = zeros(N,M);
    MSE = zeros(N,M);
    for j = 1 : M
        [y,mse] = predictor(Dec,Model{j});
        Obj(:,j) = real(y(:,1));
        MSE(:,j) = abs(real(mse(:,1)));
    end
    % Guard against tiny negative MSE from numerical errors
    MSE = max(MSE,1e-300);
end
