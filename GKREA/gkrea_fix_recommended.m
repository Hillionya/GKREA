function S = gkrea_fix_recommended(M)
%gkrea_fix_recommended - 按目标数返回经验证的推荐开关配置。
%
%   S = gkrea_fix_recommended(M) 依据 2026-10-22 的三臂 A/B 结果给出配置：
%     M >= 6 ：启用纠错集（distGate/latePhaseRel/degenPCA/eliteTrunc）。
%              实测 DTLZ5/M=10 上 IGD −7.6%（5 种子），DTLZ1/M=10 −6.5%、
%              DTLZ2/M=10 −2.3%（完整包）。
%     M <= 4 ：全部关闭。实测 M=3 上任何修复集都退步（phase +8.5%、
%              struct +19.7%、full +19.1%），原版在该档最优——低 M 时
%              「第 1 轮就开门控」虽然与注释的意图不符，但事实上有利收敛。
%
%   用法：
%       global GKREAOPT
%       GKREAOPT = gkrea_fix_recommended(Problem.M);
%   注意：GKREA_MON / gkrea_mon_run 不会清除 GKREAOPT，可在批量脚本里按 M 设置。
%
%   参见 gkrea_opt, gkrea_fix_all, GKREA_workspace/krea_ab_run2.m。

    if nargin < 1 || isempty(M), M = 10; end
    if M >= 6
        S = struct('distGate',true,'latePhaseRel',true,'degenPCA',true, ...
                   'eliteTrunc',true,'alignW',false,'localFix',false, ...
                   'exploreFloor',false,'localBatch',false);
    else
        S = struct('distGate',false,'latePhaseRel',false,'degenPCA',false, ...
                   'eliteTrunc',false,'alignW',false,'localFix',false, ...
                   'exploreFloor',false,'localBatch',false);
    end
end
