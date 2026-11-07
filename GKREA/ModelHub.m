function [Model,THETA,Info] = ModelHub(A2,Model,THETA,maxTrain)
%ModelHub - Windowed Kriging training and batch prediction management.
%
%   [Model,THETA] = ModelHub(A2,Model,THETA,maxTrain) trains one Kriging
%   model per objective. If the archive exceeds maxTrain points, a training
%   subset is built by keeping (i) all non-dominated solutions, (ii) the
%   most recently evaluated solutions, and (iii) a farthest-point subset of
%   the remainder, so that the cubic training cost stays bounded.
%
%   See also SurrogatePredict, dacefit.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Dec = A2.decs;
    Obj = A2.objs;
    n   = size(Dec,1);
    D   = size(Dec,2);

    %% Window management: keep the training set bounded
    if n > maxTrain
        idx     = false(1,n);
        [F,~]   = NDSort(Obj,inf);
        idx(F<=2) = true;                                   % elitist part
        nRecent   = min(ceil(maxTrain/3),n);
        idx(end-nRecent+1:end) = true;                      % recent part
        rest      = find(~idx);
        nPick     = maxTrain - sum(idx);
        if nPick > 0 && ~isempty(rest)
            idx(rest(farthestSubset(Dec(rest,:),Dec(idx,:),nPick))) = true;
        end
        Dec = Dec(idx,:);
        Obj = Obj(idx,:);
    end

    %% Duplicate removal (same policy as the original DISK)
    [~,distinct] = unique(round(Dec*1e100)/1e100,'rows');
    Dec = Dec(distinct,:);
    Obj = Obj(distinct,:);

    %% Train one Kriging model per objective with warm-started theta
    %  Trend selection: dacefit with 'regpoly1' builds F = [1, S] with
    %  p = D+1 regression terms and requires p <= m (m training sites);
    %  otherwise it raises "least squares problem is underdetermined".
    %  When the archive holds fewer than D+1 distinct samples (small N, high
    %  decision dimension) we fall back to ordinary Kriging with a constant
    %  trend ('regpoly0', p = 1), which is always well posed and numerically
    %  much better conditioned. theta still has D entries, so the warm start
    %  remains valid.
    if size(Dec,1) < D + 1
        regr = 'regpoly0';
    else
        regr = 'regpoly1';
    end
    for i = 1 : size(Obj,2)
        dmodel     = dacefit(Dec,Obj(:,i),regr,'corrgauss',THETA(i,:),1e-5.*ones(1,D),100.*ones(1,D));
        Model{i}   = dmodel;
        THETA(i,:) = dmodel.theta;
    end

    %% [MON] Diagnostic info (read-only)
    if nargout > 2
        Info.nArchive = n;
        Info.nTrain   = size(Dec,1);
        Info.nDropped = n - size(Dec,1);
        Info.regr     = regr;
        Info.theta    = THETA;
    end
end

function pick = farthestSubset(Cand,Base,nPick)
% Greedy farthest-point selection of nPick candidates w.r.t. Base ∪ picked.
    if isempty(Base)
        % start from the candidate farthest from the centroid
        [~,first] = max(pdist2(Cand,mean(Cand,1)));
        picked    = first;
    else
        picked = [];
    end
    pool = true(1,size(Cand,1));
    if ~isempty(picked)
        pool(picked) = false;
    end
    while length(picked) < nPick && any(pool)
        idx  = find(pool);
        dmin = min(pdist2(Cand(idx,:),[Base;Cand(picked,:)]),[],2);
        [~,k]   = max(dmin);
        picked  = [picked,idx(k)];
        pool(idx(k)) = false;
    end
    pick = picked;
end
