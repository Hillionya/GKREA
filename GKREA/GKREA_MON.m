classdef GKREA_MON < GKREA
% <2026> <multi/many> <real/integer> <expensive>
% GKREA-MON - GKREA 的监测孪生版本（数值行为与 GKREA 完全一致）。
%
%   本类只做一件事：在与 GKREA 完全相同的主循环里插入 MON 记录调用。
%   所有机制模块（LearnDistribution / SurrogateSearch / InfillSelect /
%   APDSelection / StateMonitor / LocalExplore / ModelHub）共用，随机数
%   调用顺序不变，因此同种子下 GKREA_MON 与 GKREA 的结果逐位相同
%   （见 GKREA_workspace/monitor/README_监测使用说明.md 的 A/B 校验）。
%
%   为什么用子类而不是直接改 GKREA：
%     生产算法保持零改动，论文实验可复现；监测代码出问题也不会污染
%     基线。调试时用 GKREA_MON，出结果时用 GKREA。
%
%   用法（一般通过 gkrea_mon_run 调用）：
%       MON.start(struct('outdir',outdir,'seed',1,'live',true));
%       Alg = GKREA_MON('save',0);
%       Alg.Solve(DTLZ2('M',3,'D',7,'maxFE',300));
%       MON.finish(Alg,Problem);
%
%   参见 MON, gkrea_mon_run, gkrea_mon_report.

    methods
        function obj = GKREA_MON(varargin)
            obj@GKREA(varargin{:});
        end

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
            % [FIX F4] 先按 NBI 生成并发布 |W|；自适应规则在 |W| < 0.9N
            % 时改用 MUD（恰好 N 个方向），静态开关则直接覆盖该判定。
            gkrea_state('update',struct('nDirs',size(W,1),'N',NI, ...
                'M',Problem.M,'D',Problem.D));
            if gkrea_opt('alignW')
                W = UniformPoint(NI,Problem.M,'MUD');
                gkrea_state('update',struct('nDirs',size(W,1)));
            end
            THETA = 5.*ones(Problem.M,Problem.D);
            Model = cell(1,Problem.M);
            State = StateMonitor('init',kappa0);
            State.localCap  = 0.40.*Problem.maxFE;
            State.localUsed = 0;
            State.localYield = 0;          % [ADAPT] 局部探索收益的 EMA
            iter  = 0;

            while Algorithm.NotTerminated(A2)
                iter = iter + 1;
                tIt  = tic;
                MON.beginIter(iter,Problem,A2);

                % [ADAPT] 发布演进状态：预算进度、模型误差、停滞轮数
                gkrea_state('update',struct('iter',iter, ...
                    'prog',(Problem.FE-Problem.N)./max(1,Problem.maxFE-Problem.N), ...
                    'M',Problem.M,'D',Problem.D,'N',NI, ...
                    'err',State.err,'stag',State.stag, ...
                    'localYield',State.localYield));

                %% Surrogate construction (windowed training)
                [Model,THETA,Minfo] = ModelHub(A2,Model,THETA,maxTrain);
                MON.model(Minfo);

                %% Generative distribution learning
                Dist = LearnDistribution(A2,Problem);
                MON.dist(Dist,Problem);

                %% Surrogate search (GA offspring + APD environmental selection)
                [P,S] = SurrogateSearch(A1,wmax,Model,Dist,Problem,W,State);
                MON.search(S,Problem);

                %% Infill criterion (ASF exploitation + uncertainty/sparsity exploration)
                [C,info] = InfillSelect(P,A2,alpha,Problem,W,State,Model);
                MON.infill(C,info,P,Problem,W);

                %% State update (stagnation / prediction error / adaptive weights)
                localFlag = false;
                if ~isempty(C)
                    State = StateMonitor(State,C,A2,info);
                    A2    = [A2,C];
                else
                    State = StateMonitor(State,[],A2,info);
                end
                MON.state(State);

                %% Budgeted local exploration upon stagnation / precision refinement
                if State.trigger
                    if State.localUsed < State.localCap
                        localFlag    = true;
                        [Model,THETA] = ModelHub(A2,Model,THETA,maxTrain);
                        nOut = gkrea_batch(State,alpha,State.localCap,State.localUsed);
                        [C2,Linfo] = LocalExplore(P,Model,Dist,A2,Problem,State,wmax,nOut);
                        MON.local(C2,Linfo,State);
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
                    else
                        MON.localSkipped(State);
                    end
                end

                %% Population update (APD environmental selection)
                global GKREATUNE
                apdPen = 1 + (Problem.M-1).*double(State.err >= 0.1);
                if ~isempty(GKREATUNE) && isfield(GKREATUNE,'apdPen')
                    apdPen = GKREATUNE.apdPen;
                end
                [Next,Adiag] = APDSelection(A2.objs,NI,W,apdPen);
                A1   = A2(Next);
                % [MON] Counterfactual APD impact: pen=1 (convergence) vs
                % pen=M (diversity) on the identical archive.
                if MON.on()
                    Sa = APDSelection(A2.objs,NI,W,1);
                    Sb = APDSelection(A2.objs,NI,W,Problem.M);
                    apdImpact = 1 - nnz(Sa & Sb)./max(nnz(Sa | Sb),1);
                else
                    apdImpact = NaN;
                end
                MON.apdUpdate(Adiag,apdPen,apdImpact);

                %% Runtime logging
                log.iter  = iter;
                log.FE    = Problem.FE;
                log.time  = toc(tIt);
                log.kappa = State.kappa;
                log.stag  = State.stag;
                log.local = localFlag;
                log.err   = State.err;
                GKREALOG(end+1) = log;

                MON.endIter(A2,A1,W,Problem);
            end
        end
    end
end
