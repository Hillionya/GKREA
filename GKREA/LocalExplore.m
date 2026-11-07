function [C2,Info] = LocalExplore(P,Model,Dist,A2,Problem,State,wmax,nOut)
%LocalExplore - Budgeted local exploration along the sparsest direction.
%
%   C2 = LocalExplore(P,Model,Dist,A2,Problem,State,wmax) is triggered by
%   the state monitor upon stagnation. It identifies the reference
%   direction that is worst covered by the current non-dominated front,
%   runs a short surrogate search (at most floor(wmax/3) generations with
%   early stopping) maximizing a scalarized LCB along that direction, and
%   expensively evaluates up to two well-separated candidates. Compared
%   with the original DISK local search (fixed 60 generations, single
%   output point), this version is budgeted, batch-output and early-stopped.
%
%   C2 = LocalExplore(...,nOut) caps the number of evaluated candidates at
%   nOut (default 2). When the surrogate is very accurate the caller may
%   request a larger polish batch to densify the front coverage.
%
%   NOTE on basin hopping: a multi-start "basin hopping" variant of this
%   routine (optimistic multi-restart search for multimodal landscapes) was
%   implemented and measured; it is preserved as
%   experimental/LocalExplore_basinhopping.m but is NOT active, because on
%   DTLZ1/DTLZ3 it produced statistically indistinguishable results (10
%   runs, Wilcoxon p = 0.91 / 0.08 with a slight regression trend) while
%   adding a large trajectory variance. See 02_GKREA算法总结文档 §7.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    C2   = [];
    Info = struct('dirIdx',NaN,'gens',0,'stopGen',0,'earlyStop',false, ...
                  'nGS',0,'nOut',0,'nCand',0,'fitHist',[]);
    if isempty(P) || isempty(P.decs)
        return;
    end
    if nargin < 8 || isempty(nOut)
        nOut = 2;
    end
    nOut = max(1,round(nOut));
    Info.nOut = nOut;
    M  = Problem.M;
    D  = Problem.D;
    NI = Problem.N;

    %% Identify the sparsest reference direction and the shifted ideal
    [F,~] = NDSort(A2.objs,1);
    Front = A2(F==1).objs;
    nadir = max(Front,[],1);
    ideal = min(Front,[],1);
    ideal = ideal - (nadir-ideal)/10 - 0.1;
    V     = UniformPoint(10*NI,M);
    Fc    = Front - ideal;
    Ang   = acos(min(max(1-pdist2(V,Fc,'cosine'),-1),1));
    [~,idx] = max(min(Ang,[],2));
    Wt    = V(idx,:);
    Wt    = max(Wt,1e-6);

    %% Normalization frame from the archive
    zmin = min(A2.objs,[],1);
    zmax = max(A2.objs,[],1);
    rng_ = max(zmax-zmin,1e-12);

    %% Short budgeted search on the surrogates
    Q.decs = P.decs;
    N      = size(Q.decs,1);
    gens   = max(10,floor(wmax/3));
    bestF  = inf;
    stall  = 0;
    nGS    = ceil(N/4);
    fitHist = nan(gens,1);
    for w = 1 : gens
        Off1 = OperatorDE_rand_1(Problem,Q.decs);
        Off2 = OperatorDE_current_rand_1(Problem,Q.decs);
        GS   = repmat(Dist.mu,nGS,1) + (randn(nGS,D).*repmat(Dist.s,nGS,1))*Dist.V.';
        if gkrea_opt('localIso')
            % [FIX F6] 谱塌缩后（effRank 可降至 2.8/10）高斯样本被压扁在
            % 前沿切空间；按秩亏程度自适应地混入各向同性分量：
            % effRankRel 越低，法向探索越强（权重 0.02~0.20）。
            er     = min(max(Dist.effRank/D,0),1);
            epsIso = min(0.20,max(0.02,0.15*(1-er)));
            GS = GS + epsIso*sqrt(mean(Dist.s.^2)).*randn(nGS,D);
        end
        GS   = min(max(GS,Problem.lower),Problem.upper);
        Q.decs = unique([Q.decs;Off1;Off2;GS],'rows');

        [Q.objs,Q.mse] = SurrogatePredict(Model,Q.decs);
        lcbN = ((Q.objs - State.kappa.*sqrt(Q.mse)) - zmin)./rng_;
        if gkrea_opt('localAbs')
            % [FIX F9 已撤回] 原判断「lcbN 非负、abs 是错的」不成立：
            % 这里的 zmin 是档案真实最小值（不是 InfillSelect 里平移过的
            % 理想点），故优于档案的候选 lcbN<0，abs() 是刻意的信赖域——
            % 把局部抛光约束在当前前沿附近。去掉后局部搜索会推向极端折中，
            % 实测 DTLZ3/M=6 恶化 +50%。保留开关仅用于消融。
            fit = max(lcbN.*repmat(Wt,size(lcbN,1),1),[],2) ...
                  - mean(sqrt(Q.mse)./rng_,2);
        else
            fit = max(abs(lcbN).*repmat(Wt,size(lcbN,1),1),[],2) ...
                  - mean(sqrt(Q.mse)./rng_,2);
        end

        [fit,ord] = sort(fit,'ascend');
        Q.decs = Q.decs(ord(1:min(N,size(Q.decs,1))),:);
        fitHist(w) = fit(1);
        if w == 1
            f0 = fit(1);
            if gkrea_opt('localStop')
                % [FIX F6] 相对停机阈值：原判据用绝对 1e-8，在归一化目标下
                % 几乎不可能触发，于是每次都跑满 gens 轮（实测早停率 0~37%）。
                % 改为相对量后，停滞即释放预算给下一轮代理搜索。
                tol      = 1e-5*max(1,abs(f0));
                maxStall = 4;
            else
                tol      = 1e-8;
                maxStall = 5;
            end
        end
        if fit(1) < bestF - tol
            bestF = fit(1);
            stall = 0;
        else
            stall = stall + 1;
        end
        if stall >= maxStall
            break;
        end
    end
    Info.dirIdx   = idx;
    Info.gens     = gens;
    Info.stopGen  = w;
    Info.earlyStop= (stall >= maxStall);
    Info.nGS      = nGS;
    Info.fitHist  = fitHist(1:w);
    Info.gain     = (f0 - bestF)./max(1,abs(f0));   % 相对收益，供批量自适应使用

    %% Batch output: up to nOut well-separated novel candidates
    [Q.objs,Q.mse] = SurrogatePredict(Model,Q.decs);
    lcbN = ((Q.objs - State.kappa.*sqrt(Q.mse)) - zmin)./rng_;
    if gkrea_opt('localAbs')
        fit = max(lcbN.*repmat(Wt,size(lcbN,1),1),[],2) ...
              - mean(sqrt(Q.mse)./rng_,2);
    else
        fit = max(abs(lcbN).*repmat(Wt,size(lcbN,1),1),[],2) ...
              - mean(sqrt(Q.mse)./rng_,2);
    end
    [~,ord] = sort(fit,'ascend');

    span  = Problem.upper - Problem.lower;
    An    = (A2.decs - Problem.lower)./span;
    cand  = [];
    for k = 1 : length(ord)
        x  = Q.decs(ord(k),:);
        xn = (x - Problem.lower)./span;
        if min(pdist2(xn,An)) > 1e-6
            cand = [cand;x];
            An   = [An;xn];
            if size(cand,1) >= nOut
                break;
            end
        end
        if k >= 50   % do not scan the whole population
            break;
        end
    end
    if ~isempty(cand)
        C2 = Problem.Evaluation(cand);
    end
    Info.nCand = size(cand,1);
end
