classdef GKREA < ALGORITHM
% <2026> <multi/many> <real/integer> <expensive>
% Generative-distribution Kriging-assisted Reference-vector Evolutionary Algorithm
% wmax     ---  60 --- Generations of surrogate search per iteration
% alpha    ---   5 --- Number of infill candidates per iteration
% maxTrain --- 250 --- Maximum size of the Kriging training set
% kappa0   ---   2 --- Base exploration weight of the LCB criterion

%------------------------------- Reference --------------------------------
% GKREA is a ground-up reconstruction of the distribution-information
% Kriging framework for expensive multi-/many-objective optimization:
%   Z. Zhang, Y. Wang, G. Sun, and T. Pang. A distribution information based
%   Kriging-assisted evolutionary algorithm for expensive many-objective
%   optimization problems. IEEE Transactions on Evolutionary Computation,
%   2025, 29(6): 2656-2670.
%
% Five core mechanisms (all implemented as independent modules):
%   (1) Generative use of the distribution information: the elite
%       distribution (mu,K) drives Gaussian sampling and a Mahalanobis
%       trust penalty, replacing the O(N^2) pairwise dominance comparison
%       (DIPD) and its covariance-singularity risk      -- LearnDistribution.
%   (2) Reference-vector guided surrogate search with APD environmental
%       selection whose angle-penalty coefficient adapts to the model error
%       (pen = 1 + (M-1)*1[err>=0.1]), replacing the GA-only search with
%       constant diversity pressure                          -- SurrogateSearch,
%       APDSelection.
%   (3) A functional-split infill criterion: exploitation picks are the
%       per-direction champions of an augmented-Tchebycheff scalarization
%       of the predicted means, greedily diversified; exploration picks
%       maximize a mixture of prediction uncertainty and decision-space
%       sparsity with a state-adaptive quota; on degenerate/partially
%       covered fronts an active-direction gate concentrates the late-phase
%       evaluation budget on the region where the front actually lies
%                                                            -- InfillSelect.
%   (4) A state monitor that estimates the online prediction error and
%       stagnation, adapts the exploration weight and triggers a budgeted
%       local exploration with an eager precision-refinement mode when the
%       surrogate is highly accurate        -- StateMonitor, LocalExplore.
%   (5) Windowed Kriging training (elite / recent / farthest-point tiers)
%       that caps the training cost                        -- ModelHub.
% In addition, all Kriging predictions are computed in one batched call per
% objective, which is about one order of magnitude faster than the pointwise
% prediction loop of the original implementation.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [wmax,alpha,maxTrain,kappa0] = Algorithm.ParameterSet(60,5,250,2);

            %% Initialization
            global GKREALOG
            GKREALOG = struct('iter',{},'FE',{},'time',{},'kappa',{},'stag',{},'local',{},'err',{});
            NI    = Problem.N;
            OP    = UniformPoint(NI,Problem.D,'Latin');
            A2    = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*OP+repmat(Problem.lower,NI,1));
            A1    = A2;
            % [ADAPT] 清空上一 run 的演进状态，避免跨 run 泄漏
            gkrea_state('reset');
            W = UniformPoint(NI,Problem.M);
            % [FIX F4] 先按 NBI 生成并发布 |W|；|W| < 0.9N 时改用 MUD
            gkrea_state('update',struct('nDirs',size(W,1),'N',NI, ...
                'M',Problem.M,'D',Problem.D));
            if gkrea_opt('alignW')
                W = UniformPoint(NI,Problem.M,'MUD');
                gkrea_state('update',struct('nDirs',size(W,1)));
            end
            THETA = 5.*ones(Problem.M,Problem.D);
            Model = cell(1,Problem.M);
            State = StateMonitor('init',kappa0);
            % Local-exploration budget: safety cap on the evaluations that
            % triggered local refinements may consume (generous by default
            % so that stagnation-driven refinement is never throttled).
            State.localCap  = 0.40.*Problem.maxFE;
            State.localUsed = 0;
            State.localYield = 0;          % [ADAPT] 局部探索收益的 EMA
            iter  = 0;

            while Algorithm.NotTerminated(A2)
                iter = iter + 1;
                tIt  = tic;

                % [ADAPT] 发布演进状态（供 gkrea_rule 使用）
                gkrea_state('update',struct('iter',iter, ...
                    'prog',(Problem.FE-Problem.N)./max(1,Problem.maxFE-Problem.N), ...
                    'M',Problem.M,'D',Problem.D,'N',NI, ...
                    'err',State.err,'stag',State.stag, ...
                    'localYield',State.localYield));

                %% Surrogate construction (windowed training)
                [Model,THETA] = ModelHub(A2,Model,THETA,maxTrain);

                %% Generative distribution learning
                Dist = LearnDistribution(A2,Problem);

                %% Surrogate search (GA offspring + APD environmental selection)
                P = SurrogateSearch(A1,wmax,Model,Dist,Problem,W,State);

                %% Infill criterion (ASF exploitation + uncertainty/sparsity exploration)
                [C,info] = InfillSelect(P,A2,alpha,Problem,W,State,Model);

                %% State update (stagnation / prediction error / adaptive weights)
                localFlag = false;
                if ~isempty(C)
                    State = StateMonitor(State,C,A2,info);
                    A2    = [A2,C];
                else
                    State = StateMonitor(State,[],A2,info);
                end

                %% Budgeted local exploration upon stagnation / precision refinement
                if State.trigger && State.localUsed < State.localCap
                    localFlag    = true;
                    [Model,THETA] = ModelHub(A2,Model,THETA,maxTrain);
                    % More accurate surrogate -> larger polish batch
                    nOut = gkrea_batch(State,alpha,State.localCap,State.localUsed);
                    [C2,Linfo] = LocalExplore(P,Model,Dist,A2,Problem,State,wmax,nOut);
                    if isfield(Linfo,'gain')
                        % [ADAPT] 记录本次局部搜索的相对收益，供下次批量决策
                        State.localYield = 0.5*State.localYield + 0.5*Linfo.gain;
                    end
                    if ~isempty(C2)
                        A2 = [A2,C2];
                        State.localUsed = State.localUsed + numel(C2);
                    end
                    State.trigger = false;
                    State.stag    = 0;
                end

                %% Population update (APD environmental selection)
                global GKREATUNE
                apdPen = 1 + (Problem.M-1).*double(State.err >= 0.1);
                if ~isempty(GKREATUNE) && isfield(GKREATUNE,'apdPen')
                    apdPen = GKREATUNE.apdPen;
                end
                Next = APDSelection(A2.objs,NI,W,apdPen);
                A1   = A2(Next);

                %% Runtime logging
                log.iter  = iter;
                log.FE    = Problem.FE;
                log.time  = toc(tIt);
                log.kappa = State.kappa;
                log.stag  = State.stag;
                log.local = localFlag;
                log.err   = State.err;
                GKREALOG(end+1) = log;
            end
        end
    end
end
