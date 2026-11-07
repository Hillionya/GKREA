function [C,info] = InfillSelect(P,A2,alpha,Problem,W,State,Model)
%InfillSelect - ASF exploitation + uncertainty/sparsity exploration infill.
%
%   [C,info] = InfillSelect(P,A2,alpha,Problem,W,State) selects up to
%   alpha candidates from the surrogate-optimal population P for expensive
%   evaluation, and evaluates them.
%
%   The batch is composed of two functional parts:
%     (i) EXPLOITATION picks (the majority): candidates minimizing an
%         augmented-Tchebycheff scalarization of the predicted objective
%         vector (penalized by the Mahalanobis outlier penalty) along
%         their associated reference vector, chosen greedily with an
%         angle/distance diversification against the current front;
%     (ii) EXPLORATION picks (1 by default, 2 under stagnation or high
%         model error): candidates maximizing a mixture of prediction
%         uncertainty and decision-space sparsity.
%
%   This split guarantees that most of the evaluation budget drives
%   convergence while a controlled share keeps the model honest.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    C    = [];
    info = struct('predObj',[],'predMSE',[],'degenerate',false, ...
                  'latePhase',false,'gateImpact',0);
    if isempty(P) || isempty(P.decs)
        return;
    end

    %% Duplicate removal against the archive (normalized decision space)
    span  = Problem.upper - Problem.lower;
    Pn    = (P.decs - Problem.lower)./span;
    An    = (A2.decs - Problem.lower)./span;
    dmin  = min(pdist2(Pn,An),[],2);
    keep  = dmin > 1e-6;
    idxAll = find(keep);
    if isempty(idxAll)
        return;
    end
    decs = P.decs(idxAll,:);
    objs = P.objs(idxAll,:);
    mse  = P.objmse(idxAll,:);
    lcb  = P.lcb(idxAll,:);      % normalized predicted mean + penalty
    Pn   = Pn(idxAll,:);

    %% Exploitation score: ASF of predicted objectives along reference vectors
    M    = size(objs,2);
    Wn   = W./max(sqrt(sum(W.^2,2)),1e-12);
    Ln   = lcb./max(sqrt(sum(lcb.^2,2)),1e-12);
    cosA = min(max(Ln*Wn.',-1),1);
    [~,assign] = max(cosA,[],2);
    wsel  = max(W(assign,:),1e-6);
    sE    = max(lcb./wsel,[],2) + 0.05.*sum(lcb,2);

    %% Exploration score: uncertainty + decision-space sparsity
    oRange = max(objs,[],1) - min(objs,[],1);
    unc    = mean(sqrt(mse)./max(oRange,1e-12),2);
    unc    = unc./max(max(unc),1e-12);
    spar   = min(pdist2(Pn,An),[],2)./sqrt(size(decs,2));
    spar   = spar./max(max(spar),1e-12);
    sX     = 0.5.*unc + 0.5.*spar;
    % NOTE: an optimism term (rewarding predicted-good candidates when the
    % model error is high) was added for multimodal landscapes and measured
    % as neutral over 10 runs (see the basin-hopping ablation record in
    % 02_GKREA算法总结文档 §7); the plain uncertainty/sparsity form is kept.

    %% Functional split of the batch (exploration is state-adaptive)
    nExplore = double(State.err > 0.05 || State.stag >= 1) ...
             + double(State.stag >= 2 || State.err > 0.20);
    if gkrea_opt('exploreFloor')
        % [FIX F7] 公式在「不滞且模型较准」时给出 0，与「默认 1 个探索点」
        % 的文档不一致；补一个下限，保证批次始终留有探索通道。
        nExplore = nExplore + 1;
    end
    nExplore = min(max(nExplore,0),alpha-1);
    nExploit = alpha - nExplore;

    %% Reference sets for diversification
    [F1,~]   = NDSort(A2.objs,1);
    Front    = A2(F1==1).objs;
    zmin     = min([Front;objs],[],1);
    zmax     = max([Front;objs],[],1);
    FrontN   = (Front - zmin)./max(zmax-zmin,1e-12);
    objsN    = (objs - zmin)./max(zmax-zmin,1e-12);
    FrontDecN = (A2(F1==1).decs - Problem.lower)./span;

    %% --- Exploitation picks: directional champions + greedy coverage ---
    % One champion (best ASF) per reference vector keeps the candidates
    % spread along the whole estimated front; greedy max-angle selection
    % among the champions then maximizes the coverage gain per batch.
    %
    % Active-direction gating: on degenerate or partially covered fronts
    % (the archive front activates less than half of the reference
    % vectors), champions attached to inactive directions are penalized
    % so that most of the evaluation budget is concentrated along the
    % region where the true front actually lies, while a small residual
    % pressure keeps expanding into unexplored directions. The gating of
    % exploitation picks only applies in the later budget phase: early
    % on, the archive front is too small to identify the active region
    % reliably, and premature concentration causes coverage holes.
    FnF      = FrontN./max(sqrt(sum(FrontN.^2,2)),1e-12);
    cosF     = min(max(FnF*Wn.',-1),1);
    [~,fAsg] = max(cosF,[],2);
    active   = unique(fAsg);
    % 发布演进状态：方向占用比例与相邻两轮的占用稳定性（Jaccard）。
    % 「占用已经稳定」是自适应相位判据的核心信号：占用仍在剧烈变化时
    % 说明搜索还没找到真正的前沿区域，此时集中评估会把范围过早锁死。
    Sst = gkrea_state('get');
    jac = NaN;
    if isfield(Sst,'active') && ~isempty(Sst.active)
        jac = numel(intersect(active,Sst.active))./ ...
              max(numel(union(active,Sst.active)),1);
    end
    actR  = numel(active)./max(size(W,1),1);
    actMx = actR;
    if isfield(Sst,'activeMax') && ~isnan(Sst.activeMax)
        actMx = max(actMx,Sst.activeMax);      % 历史最高覆盖率（自适应相位判据）
    end
    gkrea_state('update',struct('activeRatio',actR,'activeMax',actMx, ...
        'jaccard',jac,'active',active,'nDirs',size(W,1)));

    % ---- 门控触发：两个判据各司其职 ----
    % 占用率判「该不该集中」：档案还没铺开 = 预算应集中到已证实有解的方向。
    % PCA 判「要不要豁免」：真正的一维退化前沿（如 DTLZ5）沿曲线铺开才更优，
    % 集中评估反而会把搜索锁在曲线的一小段上。
    degCov = numel(active) < 0.5.*size(W,1);
    oneDim = false;
    p95    = NaN;
    if (gkrea_opt('degenHybrid') || gkrea_opt('degenPCA')) && ...
            size(FrontN,1) >= 3*Problem.M
        % 样本不足 3M 时 PCA 维数估计不可靠，不做豁免判定。
        Fn = FrontN - mean(FrontN,1);
        [~,Sv,~] = svd(Fn,'econ');
        sv  = diag(Sv);
        p95 = find(cumsum(sv.^2)/max(sum(sv.^2),eps) >= 0.95,1);
        if isempty(p95), p95 = size(Fn,2); end
        oneDim = p95 <= 1;
    end
    if gkrea_opt('degenHybrid')
        % [FIX F3'] 占用率为主、PCA 仅作豁免，且豁免只在 M<=6 生效：
        % 高 M 时一维曲线的长度随 M 增长，沿曲线铺开的预算不够，仍应集中。
        % 依据 2026-10-23 全矩阵：DTLZ3/M=6 因纯 PCA 判据恒假导致 +63%；
        % DTLZ5/M=4、M=8 因豁免而取得 −53%、−29%。
        degenerate = degCov && ~(oneDim && Problem.M <= 6);
    elseif gkrea_opt('degenPCA')
        % 原 F3 语义（纯 PCA 维数判据），保留用于消融复现。
        if isnan(p95)
            degenerate = degCov;
        else
            degenerate = p95 <= max(2,floor(Problem.M/2));
        end
    else
        degenerate = degCov;
    end
    if strcmpi(gkrea_mode(),'auto')
        % [ADAPT] 自适应模式下「规则判定为假」= 当前不该门控（而不是回落到
        % 原式）。原式 FE>0.3·maxFE 在 N=100、maxFE=300 时恒真，若在此回落，
        % 关闭自适应反而等于更强的门控。
        latePhase = gkrea_opt('latePhaseRel');
    elseif gkrea_opt('latePhaseRel')
        % [FIX F2] 相位以「初始采样之后的预算进度」判定。原式
        % FE > 0.3*maxFE 在 maxFE=300、N=100 时第 1 轮即恒真。
        p0        = Problem.N;
        latePhase = (Problem.FE - p0) > 0.3.*(Problem.maxFE - p0);
    else
        latePhase = Problem.FE > 0.3.*Problem.maxFE;
    end
    gateExploit = degenerate && latePhase;
    % 发布门控状态：供下一轮的迟滞判据使用（避免阈值附近抖动）
    gkrea_state('update',struct('gateOn',gateExploit));

    champ  = [];
    champV = [];
    for v = 1 : size(W,1)
        cand = find(assign==v);
        if ~isempty(cand)
            [~,k]  = min(sE(cand));
            champ  = [champ,cand(k)];
            champV = [champV,v];
        end
    end
    picked = greedyExploit(sE,objsN,Pn,FrontN,FrontDecN,assign,active, ...
                           nExploit,gateExploit,champ,champV);
    % [MON] Counterfactual: what would the batch be without active-direction
    % gating? Read-only diagnostic; no random numbers are consumed.
    if gateExploit && MON.on()
        pickedNoGate = greedyExploit(sE,objsN,Pn,FrontN,FrontDecN,assign,active, ...
                                     nExploit,false,champ,champV);
        info.gateImpact = 1 - numel(intersect(picked,pickedNoGate))./ ...
                              max(numel(union(picked,pickedNoGate)),1);
    end
    % Fallback: if the champion pool ran out before the exploitation
    % quota was met, fill the remaining slots by the best ASF scores.
    if length(picked) < nExploit
        rest2 = setdiff(1:numel(sE),picked);
        if ~isempty(rest2)
            [~,ord2] = sort(sE(rest2));
            picked   = [picked,rest2(ord2(1:min(nExploit-length(picked),numel(ord2))))];
        end
    end
    nExploitPicked = numel(picked);

    %% --- Exploration picks: best exploration score among novel candidates ---
    rest = setdiff(1:numel(sX),picked);
    % On degenerate fronts the highest-uncertainty/sparsity candidates
    % usually lie far off the true front, where evaluations are wasted;
    % restrict the exploration pool to the active directions.
    if degenerate
        restA = rest(ismember(assign(rest),active));
        if ~isempty(restA)
            rest = restA;
        end
    end
    if nExplore > 0 && ~isempty(rest)
        % avoid duplicating the picked ones in decision space
        if ~isempty(picked)
            dP = min(pdist2(Pn(rest,:),Pn(picked,:)),[],2);
            rest = rest(dP > 1e-6);
        end
        if ~isempty(rest)
            [~,ordX] = sort(sX(rest),'descend');
            picked   = [picked,rest(ordX(1:min(nExplore,numel(ordX))))];
        end
    end

    nExplorePicked = numel(picked) - nExploitPicked;

    %% [MON] Diagnostics for the monitor (read-only)
    [~,ordE]             = sort(sE);
    rankE                = zeros(1,numel(sE));
    rankE(ordE)          = 1:numel(sE);
    info.nCand           = numel(idxAll);
    info.nDup            = sum(~keep);
    info.nExploit        = nExploitPicked;
    info.nExplore        = nExplorePicked;
    info.nPicked         = numel(picked);
    info.picked          = picked(:)';
    info.pickedDec       = decs(picked,:);
    info.sourceCode      = [ones(1,nExploitPicked),2.*ones(1,nExplorePicked)];
    info.sourceName      = [repmat({'exploit'},1,nExploitPicked), ...
                            repmat({'explore'},1,nExplorePicked)];
    info.pickedDir       = assign(picked)';
    info.pickedASF       = sE(picked)';
    info.pickedSX        = sX(picked)';
    info.pickedLcb       = lcb(picked,:);
    info.pickedASFRank   = rankE(picked)';
    info.objsAll         = objs;
    info.lcbAll          = lcb;
    info.sE              = sE;
    info.sX              = sX;
    info.assign          = assign;
    info.frontDecN       = FrontDecN;
    info.active          = active;
    info.nDirs          = size(W,1);
    info.activeDirs      = numel(active);
    info.gateExploit     = gateExploit;

    %% Expensive evaluation
    C            = Problem.Evaluation(decs(picked,:));
    info.predObj = objs(picked,:);
    info.predMSE = mse(picked,:);
    info.degenerate = degenerate;
    info.latePhase  = latePhase;
end

function picked = greedyExploit(sE,objsN,Pn,FrontN,FrontDecN,assign,active, ...
                                nExploit,gateExploit,champ,champV)
%greedyExploit - Directional champions + greedy angle/distance coverage.
%   Extracted verbatim from the original selection loop so that the
%   counterfactual (gate on/off) diagnostic reuses exactly the same logic.

    picked = [];
    while length(picked) < nExploit && ~isempty(champ)
        if isempty(picked)
            score = sE(champ);
            if gateExploit
                score(~ismember(champV,active)) = score(~ismember(champV,active)) + ...
                    0.5.*max(score,[],'omitnan');
            end
            [~,k] = min(score);
        else
            Ref  = [FrontN;objsN(picked,:)];
            A    = acos(min(max(1-pdist2(objsN(champ,:),Ref,'cosine'),-1),1));
            dAng = min(A,[],2);
            RefD = [FrontDecN;Pn(picked,:)];
            dDec = min(pdist2(Pn(champ,:),RefD),[],2)./sqrt(size(Pn,2));
            div  = 0.7.*dAng./max(max(dAng),1e-12) + 0.3.*dDec./max(max(dDec),1e-12);
            if gateExploit
                div(~ismember(champV,active)) = 0.2.*div(~ismember(champV,active));
            end
            [~,k] = max(div - 0.05.*sE(champ)./max(max(sE(champ)),1e-12));
        end
        picked  = [picked,champ(k)];
        champ(k)  = [];
        champV(k) = [];
    end
end
