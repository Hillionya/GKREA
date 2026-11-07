function v = gkrea_rule(name,S)
%gkrea_rule - 状态自适应开关的判定规则。
%
%   v = gkrea_rule(name,S) 根据当前演进状态 S（见 gkrea_state）判定机制是否启用。
%
%   与「按 M 分档」的静态配置相比，这里只用与失败模式直接相关的**演进信号**：
%   谱秩亏程度、离群比例、方向占用是否稳定、档案规模、局部搜索收益。
%   M 只出现在「估计是否可靠」这类样本量条件里，不作为开关依据。
%
%   首轮或状态未知时一律返回 false（= 原行为），保证自适应不会在冷启动阶段
%   改变算法。

    if nargin < 2 || isempty(S) || ~isstruct(S) || ~isfield(S,'ready') || ~S.ready
        v = false; return;
    end

    switch name
        case 'distGate'
            % 马氏惩罚只在「协方差明显秩亏（度量才有区分度）」且
            % 「确实存在离群候选」时启用；M 大时前沿接近满秩，自动关闭。
            v = ~isnan(S.effRankRel) && ~isnan(S.dmOutRobust) && ...
                S.effRankRel < 0.85 && S.dmOutRobust > 0.005;

        case 'latePhaseRel'
            % 相位只回答「是不是已经过了初始阶段」，不再掺入覆盖率条件。
            % 覆盖率已经由触发判据（degCov）负责；两者都挂在覆盖率上会互相
            % 打架（覆盖率下降时触发说该门控、相位说不该），导致门控抖动。
            v = S.prog > 0.10;

        case 'degenPCA'
            % PCA 估维数需要足够样本；不足 3M 时退回占用判据。
            v = S.nFront >= 3*S.M;

        case 'degenHybrid'
            % 门控触发改用混合判据（占用率为主、PCA 仅作一维豁免）。
            % 这是 2026-10-23 归因后的修复：纯 PCA 判据在 DTLZ3 这类多峰
            % 问题上恒为假，门控从不触发，IGD +63%。
            v = true;

        case 'alignW'
            % 结构性：NBI 在 M>=4 给出少于 N 个方向时才换成 MUD。
            v = S.nDirs > 0 && S.N > 0 && S.nDirs < 0.9*S.N;

        case 'exploreFloor'
            % 只在「预算早期」且「确实停滞」时保留一个探索位。
            % 原条件 stag>=2 单独成立就等于几乎每轮 +1（实测 exploreShare
            % 0.144 → 0.219），在多峰的 DTLZ3/DTLZ4 上代价明显。
            v = S.prog < 0.25 && S.stag >= 1;

        case 'eliteTrunc'
            % 第一层接近全档案时才需要分层截断。
            v = S.nFront > 0.6*S.N;

        case 'localBatch'
            % 批量由 localYield 连续调节（见 GKREA.m），这里恒为真。
            v = true;

        case 'localAbs'
            % [F9 撤回] 去 abs() 曾被视为正确性修正，实测是错的：
            % LocalExplore 的 zmin 是档案真实最小值，优于档案的候选 lcbN<0，
            % abs() 是刻意的信赖域。去掉后 DTLZ3/M=6 由 +5% 恶化到 +50%，
            % 5 个种子的逐种子变化全为正。默认关闭，仅保留做消融。
            v = false;

        case 'localIso'
            % 各向同性混合：给局部搜索注入法向噪声。从未单独验证，且会削弱
            % 局部抛光，自适应模式默认关闭（需要时显式打开做消融）。
            v = false;

        case 'localStop'
            % 相对停机阈值：改变局部探索的预算占比（实测 localFE 12→32.6），
            % 属于预算策略而非纠错，默认关闭，保持原始「跑满 gens 轮」。
            v = false;

        case 'localFix'
            % 历史别名：自适应模式下由上面三个子开关各自决定。
            v = false;

        otherwise
            v = false;
    end
end
