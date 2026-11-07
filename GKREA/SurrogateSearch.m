function [P,S] = SurrogateSearch(Population,wmax,Model,Dist,Problem,W,State)
%SurrogateSearch - Hybrid reference-vector guided search on the surrogates.
%
%   P = SurrogateSearch(Population,wmax,Model,Dist,Problem,W,State) runs
%   wmax generations of a hybrid search on the Kriging surrogates. Each
%   generation generates offspring by a mixture of
%     (i)   DE/current-to-rand/1  - global differential search,
%     (ii)  GA (SBX + polynomial mutation) - local refinement,
%     (iii) Gaussian sampling from the learned elite distribution
%           (generative use of the distribution information).
%   Survival uses the APD criterion (reference vectors W) on the penalized
%   lower confidence bound LCB = mu - kappa*sigma, where candidates beyond
%   the Mahalanobis threshold of the elite distribution are penalized to
%   suppress model-driven outliers.
%
%   The output P is a struct with fields decs, objs, objmse, lcb, dmahal.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N      = size(Population.decs,1);
    D      = Problem.D;
    Lower  = Problem.lower;
    Upper  = Problem.upper;
    nGA    = N;                 % GA-driven search proved most sample-efficient
    nDE    = 0;
    nGS    = 0;
    kappaS = 0;                 % search on predicted means; uncertainty is
                                % handled by the infill criterion instead
    %% --- optional tuning override (for ablation experiments only) ---
    global GKREATUNE
    % Adaptive angle penalty: an accurate surrogate (err < 0.1) allows a
    % strong convergence pressure (pen = 1), while an unreliable one needs
    % the full diversity-preserving penalty (pen = M).
    apdPen = 1 + (Problem.M-1).*double(State.err >= 0.1);
    if ~isempty(GKREATUNE)
        if isfield(GKREATUNE,'kappaS'), kappaS = GKREATUNE.kappaS; end
        if isfield(GKREATUNE,'wmax'),   wmax   = GKREATUNE.wmax;   end
        if isfield(GKREATUNE,'apdPen'), apdPen = GKREATUNE.apdPen; end
        if isfield(GKREATUNE,'mix')
            nGA = round(N*GKREATUNE.mix(1));
            nDE = round(N*GKREATUNE.mix(2));
            nGS = N - nGA - nDE;
        end
    end
    P.decs = Population.decs;
    [P.objs,P.objmse] = SurrogatePredict(Model,P.decs);
    P.diag = struct('wmax',wmax,'nGA',nGA,'nDE',nDE,'nGS',nGS, ...
                    'apdPen',apdPen,'gen',zeros(0,7),'distImpact',0, ...
                    'dmOutFrac',0,'gsPerIter',nGS.*wmax);

    for w = 1 : wmax
        %% Hybrid offspring generation
        Off = zeros(N,D);
        if nGA > 0
            OffGA = OperatorGA(Problem,P.decs);
            Off(1:nGA,:) = OffGA(1:nGA,:);
        end
        if nDE > 0
            OffDE = OperatorDE_current_rand_1(Problem,P.decs);
            Off(nGA+1:nGA+nDE,:) = OffDE(1:nDE,:);
        end
        if nGS > 0
            GS = repmat(Dist.mu,nGS,1) + (randn(nGS,D).*repmat(Dist.s,nGS,1))*Dist.V.';
            GS = min(max(GS,Lower),Upper);
            Off(nGA+nDE+1:end,:) = GS;
        end

        %% Batch prediction
        P.decs = [P.decs;Off];
        [P.objs,P.objmse] = SurrogatePredict(Model,P.decs);

        %% Penalized LCB
        [Fl,dm] = penalizedLCB(P.objs,P.objmse,P.decs,Dist,kappaS);

        %% APD environmental selection on the penalized LCB
        [Next,Adiag] = APDSelection(Fl,N,W,apdPen);
        if MON.on()
            % [MON] per-generation trace: offspring survivors, active
            % directions among survivors, mean angle to the assigned
            % direction, fraction beyond the Mahalanobis threshold, and
            % the counterfactual effect of the penalty on the survivors.
            FlNo = penalizedLCB(P.objs,P.objmse,P.decs,Dist,kappaS,false);
            A    = APDSelection(Fl  ,N,W,apdPen);
            B    = APDSelection(FlNo,N,W,apdPen);
            % [FIX F0] A/B 是 logical 掩码，intersect 会返回取值 {0,1}
            % 而不是被选下标，导致 imp 恒为 0。改用逻辑与/或计数。
            imp  = 1 - nnz(A & B)./max(nnz(A | B),1);
            P.diag.gen(end+1,:) = [w,sum(Next(N+1:end)),Adiag.nActive, ...
                                   mean(Adiag.theta),mean(dm>Dist.thr),apdPen,imp];
        end
        P.decs   = P.decs(Next,:);
        P.objs   = P.objs(Next,:);
        P.objmse = P.objmse(Next,:);
    end

    %% Final candidate field for the infill criterion: normalized predicted
    %% mean + penalty (default infKappa = 0; ablation showed that a mild
    %% LCB optimism helps smooth problems but harms multimodal ones, so it
    %% remains an optional tuning knob rather than the default)
    infKappa = 0;
    if isfield(GKREATUNE,'infKappa')   % optional override for ablations
        infKappa = GKREATUNE.infKappa;
    end
    [P.lcb,P.dmahal] = penalizedLCB(P.objs,P.objmse,P.decs,Dist,infKappa);

    %% [MON] Effect of the generative-distribution channel (counterfactual).
    %  Re-run the final APD selection on the same candidates with and
    %  without the Mahalanobis penalty and measure how much the surviving
    %  set changes. This is a read-only diagnostic: it consumes no random
    %  numbers and does not alter P.
    if MON.on()
        FlNo = penalizedLCB(P.objs,P.objmse,P.decs,Dist,infKappa,false);
        A    = APDSelection(P.lcb,N,W,apdPen);
        B    = APDSelection(FlNo ,N,W,apdPen);
        % [FIX F0] 同上：logical 掩码必须用 nnz(A&B) 计数
        P.diag.distImpact = 1 - nnz(A & B)./max(nnz(A | B),1);
        if ~isempty(P.diag.gen)
            % 逐代平均：惩罚在搜索过程中对存活集合的真实影响
            P.diag.dmOutFrac  = mean(P.diag.gen(:,5));
            P.diag.distImpact = mean(P.diag.gen(:,7));
        else
            P.diag.dmOutFrac  = mean(P.dmahal > Dist.thr);
        end
        P.diag.gsPerIter  = nGS.*wmax;
        P.diag.genCount   = size(P.diag.gen,1);
    end
    S = P.diag;
end

function [Fl,dm] = penalizedLCB(Obj,MSE,Dec,Dist,kappa,usePen)
% LCB normalized to [0,1] plus a Mahalanobis outlier penalty.
    if nargin < 6
        usePen = true;
    end
    LCB  = Obj - kappa.*sqrt(MSE);
    zmin = min(LCB,[],1);
    zmax = max(LCB,[],1);
    Fl   = (LCB - zmin)./max(zmax - zmin,1e-12);
    dc   = Dec - Dist.mu;
    dm   = sqrt(max(sum((dc*Dist.invK).*dc,2),0));
    if isfield(Dist,'thrRobust')
        % 发布「用稳健阈值度量出的真实离群比例」：这是自适应规则判断
        % 马氏惩罚是否该启用的唯一依据（原 1.5·max 阈值下该比例恒为 0）。
        gkrea_state('update',struct('dmOutRobust',mean(dm > Dist.thrRobust)));
    end
    if usePen
        if gkrea_opt('distGate')
            % [FIX F1] 归一化惩罚 + 截断：越界多少倍罚多少，量纲与 D 无关。
            % 截断在 2 倍阈值处，避免个别极端候选（dm~1e3）主导整个排序。
            pen = max(0,dm./max(Dist.thr,eps) - 1);
            pen = min(pen,2);
        else
            pen = max(0,dm - Dist.thr);
        end
        Fl   = Fl + 0.1.*repmat(pen,1,size(Fl,2));
    end
end
