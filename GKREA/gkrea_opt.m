function v = gkrea_opt(name)
%gkrea_opt - 读取 GKREA 机制开关（静态标志 或 状态自适应规则）。
%
%   v = gkrea_opt(name)
%
%   三种工作模式：
%     1) 未设置 GKREAOPT            → 全部 false（原始行为，基线可复现）
%     2) GKREAOPT.<name> 显式给出   → 用该值（静态消融用；
%                                      gkrea_fix_all / gkrea_fix_recommended 走这条）
%     3) GKREAOPT.mode = 'auto'     → 由 gkrea_rule(name, GKREASTATE) 按演进状态决定
%
%   因此「自适应」与「静态消融」共用同一套开关，实验可逐项对照。
%
%   可用开关
%     distGate      F1  马氏阈值 median+2MAD、惩罚归一化并截断、λ 相对下限、shrink>=0.10
%     latePhaseRel  F2  门控相位改为「初始采样后的预算进度」+ 占用稳定
%     degenPCA      F3  退化判据改为目标空间 PCA 固有维数
%     alignW        F4  参考方向改用 MUD（恰好 N 个）
%     eliteTrunc    F5  精英集在高 M 下按方向分层截断
%     localAbs      F9  加权 Tchebycheff 去 abs()（正确性修正）
%     localIso      F6  各向同性混合（搜索策略，默认关）
%     localStop     F6' 相对停机阈值（预算策略，默认关）
%     localFix      历史别名：为真时等价于同时打开上面三项
%     exploreFloor  F7  探索配额至少 1 个
%     localBatch    F8  局部探索批量按观测收益自适应
%     degenHybrid   F3' 门控触发＝占用率为主 + PCA 一维豁免（M<=6）
%
%   另见 gkrea_auto, gkrea_state, gkrea_rule, gkrea_fix_all。

    global GKREAOPT
    if isempty(GKREAOPT) || ~isstruct(GKREAOPT)
        v = false;
        return;
    end
    if isfield(GKREAOPT,name)
        v = logical(GKREAOPT.(name));                  % 显式配置优先
        return;
    end
    % [FIX] localFix 是历史别名：显式为真时等价于同时打开三个子开关。
    % 2026-10-23 归因发现这三件事性质不同——abs() 是正确性修正，各向同性混
    % 合与相对停机是搜索/预算策略——捆在一起无法单独验证，DTLZ3/M=6 上
    % 因此吃了 +60%。现在拆开，自适应模式只开 localAbs。
    if isfield(GKREAOPT,'localFix') && logical(GKREAOPT.localFix) && ...
            any(strcmp(name,{'localAbs','localIso','localStop'}))
        v = true;
        return;
    end
    if isfield(GKREAOPT,'mode') && strcmpi(GKREAOPT.mode,'auto')
        v = gkrea_rule(name, gkrea_state('get'));       % 状态自适应
    else
        v = false;
    end
end
