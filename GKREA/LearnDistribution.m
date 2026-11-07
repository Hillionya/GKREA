function Dist = LearnDistribution(A2,Problem)
%LearnDistribution - Robust generative distribution model of the elite set.
%
%   Dist = LearnDistribution(A2,Problem) estimates the mean mu and the
%   covariance K of the decision vectors of the current first (and, when
%   the first front is too small, second) non-dominated front. The
%   covariance is regularized by shrinkage towards its diagonal plus a
%   small jitter, which removes the singularity risk of the original DISK
%   implementation and stabilizes sampling in high dimensions.
%
%   The returned struct provides everything needed for the generative use
%   of the distribution:
%     mu      - mean vector (1 x D)
%     V       - eigenvectors of K (D x D)
%     s       - square roots of the (clipped) eigenvalues (1 x D)
%     invK    - regularized inverse of K (for Mahalanobis distance)
%     thr     - Mahalanobis distance threshold (1.5 x max over the front)
%     range   - per-variable range of the front (for scaling)
%
%   Sample new points by: X = repmat(mu,n,1) + randn(n,D).*repmat(s,n,1) * V.'

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [F,~]  = NDSort(A2.objs,inf);
    idxF1  = find(F==1);
    % [FIX F5] 高 M 下 NDSort 第一层≈整个档案（M=10 时占 83%~95%），
    % 此时 mu/K 拟合的是搜索云而非前沿。按参考方向做一次分层截断，
    % 每个方向只保留 APD 最优的代表点。
    if gkrea_opt('eliteTrunc') && numel(idxF1) > 0.5*Problem.N
        Wl     = UniformPoint(min(Problem.N,numel(idxF1)),Problem.M);
        sel    = APDSelection(A2(idxF1).objs,size(Wl,1),Wl,Problem.M);
        PopDec = A2(idxF1(sel)).decs;
    else
        PopDec = A2(F==1).decs;
    end
    if size(PopDec,1) <= 2
        PopDec = [PopDec; A2(F==2).decs];
    end
    n  = size(PopDec,1);
    D  = Problem.D;
    mu = mean(PopDec,1);
    if n > 1
        K = cov(PopDec);
    else
        K = zeros(D);
    end

    %% Shrinkage regularization (towards the diagonal) + jitter
    if gkrea_opt('distGate')
        % [FIX F1] 收缩强度设下限：精英集增大时 (D+2)/(n-1) 会趋近 0，
        % 协方差接近奇异，invK 被零空间方向主导。
        shrink = min(0.5,max(0.10,(D+2)/max(n-1,1)));
    else
        shrink = min(0.5,(D+2)/max(n-1,1));
    end
    diagK    = max(diag(K),1e-12.*(Problem.upper-Problem.lower)'.^2);
    Ks       = (1-shrink).*K + shrink.*diag(diagK);
    jitter   = 1e-10*max(mean(diagK),eps);
    Ks       = (Ks + Ks')/2 + jitter.*eye(D);

    %% Spectral decomposition for sampling and Mahalanobis distance
    [V,Lam]  = eig(Ks);
    if gkrea_opt('distGate')
        % [FIX F1] 特征值下限改为相对量（1e-6·tr(K)/D），把条件数钳在 1e6
        % 以内；绝对值下限 1e-16 会让 invK 达到 1e16，使马氏距离病态。
        Lam = max(diag(Lam),1e-6*trace(Ks)/D)';
    else
        Lam = max(diag(Lam),1e-16)';
    end
    Dist.mu  = mu;
    Dist.V   = V;
    Dist.s   = sqrt(Lam);
    Dist.invK = V*diag(1./Lam)*V';
    Dist.range = max(max(PopDec,[],1)-min(PopDec,[],1),1e-12);

    %% Mahalanobis threshold from the front itself
    dc   = PopDec - mu;
    dm   = sqrt(max(sum((dc*Dist.invK).*dc,2),0));
    % [FIX F1] 稳健阈值（median + 2·MAD）始终计算：既供自适应规则判断
    % 「是否真的存在离群候选」，也作为启用时的阈值。
    % 原式 1.5·max(dm) 由参考样本自身最大值决定，候选与前沿同分布时永不越界。
    med = median(dm);
    mad = 1.4826*median(abs(dm-med));
    Dist.thrRobust = max(med + 2*mad,sqrt(D));
    if gkrea_opt('distGate')
        Dist.thr = Dist.thrRobust;
    else
        Dist.thr = max(1.5*max(dm),sqrt(D));
    end
    % [MON] 记录阈值是否来自稳健尺度，便于报告区分
    Dist.thrMode = double(gkrea_opt('distGate'));

    %% [MON] Diagnostic fields (read-only, no effect on the model)
    %  These fields are consumed only by the MON recorder; they do not
    %  influence sampling, selection or the random stream.
    Dist.n       = n;                       % number of front points used
    Dist.nFront  = size(A2(F==1).decs,1);   % size of the first front
    Dist.shrink  = shrink;                  % shrinkage intensity
    Dist.eig     = Lam;                     % eigenvalues of K (D x 1)
    Dist.meanDm  = mean(dm);                % mean Mahalanobis distance
    Dist.maxDm   = max(dm);                 % max Mahalanobis distance
    Dist.effRank = sum(Lam)^2/max(sum(Lam.^2),eps);  % participation ratio

    % 发布演进状态：谱秩亏程度与第一层规模是自适应规则的主要输入
    gkrea_state('update',struct('effRankRel',Dist.effRank/D,'nFront',Dist.nFront));
end
