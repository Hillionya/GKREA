function [Next,Diag] = APDSelection(PopObj,N,W,pen)
%APDSelection - Reference-vector guided APD environmental selection.
%
%   Next = APDSelection(PopObj,N,W) selects N solutions from PopObj by the
%   angle-penalized distance (APD) criterion with respect to the reference
%   vectors W: every solution is assigned to its closest reference vector,
%   the solution with the smallest APD in each vector is preferred, and the
%   remaining slots are filled by globally ascending APD. This replaces the
%   pure angle-truncation selection of the original DISK, which loses
%   convergence pressure in many-objective spaces.
%
%   Next = APDSelection(PopObj,N,W,pen) uses the angle-penalty coefficient
%   pen (default: number of objectives). Smaller pen shifts pressure from
%   diversity towards convergence.
%
%   [Next,Diag] = APDSelection(...) additionally returns a read-only
%   diagnostic struct (assign/theta/apd/nActive/nFill) used by the MON
%   monitor. It does not affect the selection.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [NP,M] = size(PopObj);
    if nargin < 4
        pen = M;
    end
    Diag = struct('pen',pen,'assign',ones(1,NP),'theta',zeros(1,NP), ...
                  'apd',zeros(1,NP),'nActive',0,'nPerVector',0,'nFill',0);
    if NP <= N
        Next = true(1,NP);
        return;
    end

    %% Normalization
    zmin = min(PopObj,[],1);
    zmax = max(PopObj,[],1);
    F    = (PopObj - zmin)./max(zmax - zmin,1e-12);

    %% Associate each solution to its closest reference vector
    Wn   = W./max(sqrt(sum(W.^2,2)),1e-12);
    Fn   = F./max(sqrt(sum(F.^2,2)),1e-12);
    cosA = Fn*Wn.';
    cosA = min(max(cosA,-1),1);
    Angle = acos(cosA);
    [theta,assign] = min(Angle,[],2);

    %% Angle-penalized distance
    APD  = (1 + pen.*theta).*sqrt(sum(F.^2,2));

    %% Select the best solution per reference vector, then fill by APD
    Next  = false(1,NP);
    [~,order] = sort(APD);
    for v = 1 : size(W,1)
        cand = order(assign(order)==v);
        if ~isempty(cand)
            Next(cand(1)) = true;
        end
        if sum(Next) >= N
            break;
        end
    end
    Diag.assign     = assign;
    Diag.theta      = theta;
    Diag.apd        = APD;
    Diag.nPerVector = sum(Next);              % before the APD fill
    Diag.nActive    = numel(unique(assign(Next)));
    if sum(Next) < N
        rest = order(~Next(order));
        Next(rest(1:N-sum(Next))) = true;
        Diag.nFill = N - Diag.nPerVector;
    end
end
