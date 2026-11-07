function S = gkrea_auto(on)
%gkrea_auto - 打开/关闭「状态自适应」模式。
%
%   gkrea_auto()      启用：每个机制由 gkrea_rule 按当前演进状态决定
%   gkrea_auto(false) 关闭：恢复原始行为（所有开关 false）
%   S = gkrea_auto(...) 返回设置后的配置
%
%   与 gkrea_fix_all（全静态开）和 gkrea_fix_recommended（按 M 分档）并列，
%   三者共用同一套开关，可直接做 A/B。
%
%   注意：gkrea_mon_run / GKREA_MON 不会清除 GKREAOPT，因此在批量脚本里
%   设置一次即可对后续所有运行生效。

    global GKREAOPT
    if nargin < 1, on = true; end
    if on
        S = struct('mode','auto');
    else
        S = struct();
    end
    GKREAOPT = S;
    gkrea_state('reset');
end
