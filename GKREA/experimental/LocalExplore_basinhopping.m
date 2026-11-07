function C2 = LocalExplore(P,Model,Dist,A2,Problem,State,wmax,nOut)
%LocalExplore - Budgeted local exploration: refinement or basin hopping.
%
%   C2 = LocalExplore(P,Model,Dist,A2,Problem,State,wmax) is triggered by
%   the state monitor upon stagnation. Its behaviour depends on the
%   reliability of the surrogate:
%
%   (a) REFINEMENT mode (accurate surrogate, State.err < 0.10): identifies
%       the reference direction that is worst covered by the current
%       non-dominated front, runs a short surrogate search (at most
%       floor(wmax/3) generations with early stopping) maximizing a
%       scalarized LCB along that direction, and expensively evaluates up
%       to nOut well-separated candidates. This densifies the front
%       coverage and is the mode that serves degenerate/smooth fronts.
%
%   (b) BASIN-HOPPING mode (unreliable surrogate, State.err >= 0.10, and
%       stagnation State.stag >= 2): the model is known to mislead the
%       search on multimodal landscapes, and the population has stopped
%       improving while sitting in a local basin. The routine therefore
%       performs a multi-start optimistic search: the decision space is
%       re-seeded with several random restart clusters plus distribution
%       samples, the candidates are ranked by an OPTIMISTIC LCB (kappa
%       lifted to at least 3) scalarized along the direction of the best
%       converged archive point, further rewarded for prediction
%       uncertainty and for novelty (decision-space distance to the
%       archive). Only novel candidates are emitted, so the evaluated
%       points land in unexplored basins instead of re-sampling the
%       current one.
%
%   C2 = LocalExplore(...,nOut) caps the number of evaluated candidates at
%   nOut (default 2).

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    C2 = [];
    if isempty(P) || isempty(P.decs)
        return;
    end
    if nargin < 8 || isempty(nOut)
        nOut = 2;
    end
    nOut = max(1,round(nOut));
    M  = Problem.M;
    D  = Problem.D;
    NI = Problem.N;

    %% Mode selection: refinement vs. basin hopping
    % Basin hopping is a last-resort escape and only engages under DEEP
    % stagnation on an unreliable (multimodal) landscape: the cumulative
    % stagnation counter State.stagTot is not reset by triggers, so this
    % condition means "repeatedly stalled for many iterations". Shallow
    % stagnation is handled by the cheap refinement mode; engaging the hop
    % too early wastes evaluations on problems whose difficulty is coverage
    % rather than multimodality (measured: it degraded DTLZ3/DTLZ4).
    hop = (State.err >= 0.10) && (State.stagTot >= 6);   % v11g: combined mode
    % Observation counter (optional; used by the test scripts to verify how
    % often each mode is engaged).
    global GKREAHOP
    if isempty(GKREAHOP)
        GKREAHOP = struct('refine',0,'hop',0);
    end
    if hop
        GKREAHOP.hop = GKREAHOP.hop + 1;
    else
        GKREAHOP.refine = GKREAHOP.refine + 1;
    end

    %% Archive front and shifted ideal
    [F,~]  = NDSort(A2.objs,1);
    Front  = A2(F==1).objs;
    nadir  = max(Front,[],1);
    ideal  = min(Front,[],1);
    ideal  = ideal - (nadir-ideal)/10 - 0.1;
    V      = UniformPoint(10*NI,M);
    Fc     = Front - ideal;
    Ang    = acos(min(max(1-pdist2(V,Fc,'cosine'),-1),1));

    if hop
        % Hop towards the direction of the best converged archive point:
        % on multimodal problems the objective is to reach a better basin
        % in a promising direction, not to fill a coverage gap.
        [~,bs]  = min(max((Front-ideal)./max(nadir-ideal,1e-12),[],2));
        bBest   = Front(bs,:) - ideal;
        [~,idx] = max(min(acos(min(max(1-(V*bBest')./...
                    max(sqrt(sum(V.^2,2)).*norm(bBest),1e-12),-1),1)),[],2));
        Wt      = max(V(idx,:),1e-6);
        kappa   = max(State.kappa,3);
    else
        [~,idx] = max(min(Ang,[],2));      % sparsest direction
        Wt      = max(V(idx,:),1e-6);
        kappa   = State.kappa;
    end

    %% Normalization frame from the archive
    zmin = min(A2.objs,[],1);
    zmax = max(A2.objs,[],1);
    rng_ = max(zmax-zmin,1e-12);
    span = Problem.upper - Problem.lower;
    An   = (A2.decs - Problem.lower)./span;

    %% Initial population
    if hop
        % Multi-start re-seeding: several random restart clusters plus
        % distribution samples, so that the search starts outside the
        % current basin.
        nRS   = 5;
        nSub  = max(2,ceil(NI/nRS));
        Q.decs = repmat(Problem.lower,nRS*nSub,1) + ...
                 rand(nRS*nSub,D).*repmat(span,nRS*nSub,1);
        nGS   = max(2,ceil(NI/4));
        GS    = repmat(Dist.mu,nGS,1) + (randn(nGS,D).*repmat(Dist.s,nGS,1))*Dist.V.';
        Q.decs = [Q.decs;min(max(GS,Problem.lower),Problem.upper)];
    else
        Q.decs = P.decs;
    end
    N = size(Q.decs,1);

    %% Short budgeted search on the surrogates
    gens  = max(10,floor(wmax/3));
    bestF = inf;
    stall = 0;
    for w = 1 : gens
        Off1 = OperatorDE_rand_1(Problem,Q.decs);
        Off2 = OperatorDE_current_rand_1(Problem,Q.decs);
        nGS  = ceil(N/4);
        GS   = repmat(Dist.mu,nGS,1) + (randn(nGS,D).*repmat(Dist.s,nGS,1))*Dist.V.';
        GS   = min(max(GS,Problem.lower),Problem.upper);
        Q.decs = unique([Q.decs;Off1;Off2;GS],'rows');

        [fit,~]   = LocalFitness(Q,Model,Wt,zmin,rng_,kappa,Problem,span,An,D,hop);

        [fit,ord] = sort(fit,'ascend');
        Q.decs = Q.decs(ord(1:min(N,size(Q.decs,1))),:);
        if fit(1) < bestF - 1e-8
            bestF = fit(1);
            stall = 0;
        else
            stall = stall + 1;
        end
        if stall >= 5
            break;
        end
    end

    %% Batch output
    [fit,~]   = LocalFitness(Q,Model,Wt,zmin,rng_,kappa,Problem,span,An,D,hop);
    [~,ord]   = sort(fit,'ascend');

    cand = [];
    if hop
        % COMBINED mode: a hop replacing the whole batch measured as a coin
        % flip (either a much better basin or a wasted batch), so one slot
        % is always reserved for the classical refinement pick and the rest
        % go to the hop cloud.
        [rf,~]   = LocalFitness(P,Model,Wt,zmin,rng_,State.kappa,Problem,span,An,D,false);
        [~,rord] = sort(rf,'ascend');
        for k = 1 : min(numel(rord),20)
            xn = (P.decs(rord(k),:) - Problem.lower)./span;
            if min(pdist2(xn,An)) > 1e-6
                cand = [cand;P.decs(rord(k),:)];
                An   = [An;xn];
                break;
            end
        end
    end

    for k = 1 : length(ord)
        if size(cand,1) >= nOut
            break;
        end
        x  = Q.decs(ord(k),:);
        xn = (x - Problem.lower)./span;
        dA = min(pdist2(xn,An));
        % In hop mode only clearly novel candidates are emitted; in
        % refinement mode duplicates of the archive are the only exclusion.
        if dA > max(1e-6, 0.05.*double(hop))
            cand = [cand;x];
            An   = [An;xn];
        end
        if k >= 50   % do not scan the whole population
            break;
        end
    end
    if ~isempty(cand)
        C2 = Problem.Evaluation(cand);
    end
end

function [fit,asf] = LocalFitness(Q,Model,Wt,zmin,rng_,kappa,Problem,span,An,D,hop)
%LocalFitness - Scalarized LCB fitness; hop mode adds normalized rewards.
%   Refinement mode: fit = ASF(LCB along Wt) - uncertainty reward, i.e. the
%   search drives convergence and is mildly attracted by unexplored areas.
%   Hop mode: the three terms are min-max normalized so that the optimistic
%   predicted quality stays the leading criterion (weight 1), novelty (0.4)
%   and uncertainty (0.15) act as tie-breakers. This prevents the
%   uncertainty reward from dominating and flooding the batch with
%   high-variance but poor regions, which was the failure mode of the
%   first hop implementation. The raw optimistic ASF is also returned so
%   that the caller can decide whether a hop is worth evaluating at all.
    [Q.objs,Q.mse] = SurrogatePredict(Model,Q.decs);
    lcbN = ((Q.objs - kappa.*sqrt(Q.mse)) - zmin)./rng_;
    asf  = max(abs(lcbN).*repmat(Wt,size(lcbN,1),1),[],2);
    unc  = mean(sqrt(Q.mse)./rng_,2);
    if hop
        Qn  = (Q.decs - Problem.lower)./span;
        nov = min(pdist2(Qn,An),[],2)./sqrt(D);
        nrm = @(v) (v - min(v))./max(max(v)-min(v),1e-12);
        fit = nrm(asf) - 0.4.*nrm(nov) - 0.15.*nrm(unc);
    else
        fit = asf - unc;
    end
end
