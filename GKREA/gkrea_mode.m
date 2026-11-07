function m = gkrea_mode()
%gkrea_mode - 返回当前机制配置模式：''（原始）/ 'auto'（状态自适应）/ 'static'。
%
%   供需要区分「未启用」与「自适应判定为不启用」的模块使用
%   （例如 InfillSelect 里的 latePhase：两者语义不同）。

    global GKREAOPT
    if isempty(GKREAOPT) || ~isstruct(GKREAOPT) || ~isfield(GKREAOPT,'mode')
        m = '';
    else
        m = GKREAOPT.mode;
    end
end
