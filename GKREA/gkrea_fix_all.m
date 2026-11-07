function S = gkrea_fix_all(on)
%gkrea_fix_all - 一次性打开/关闭全部机制修复开关。
%
%   gkrea_fix_all()     打开全部（等价 gkrea_fix_all(true)）
%   gkrea_fix_all(false) 全部关闭（回到原始 GKREA 行为）
%   S = gkrea_fix_all(...) 返回设置后的开关结构体
%
%   开关语义见 gkrea_opt。

    if nargin < 1, on = true; end
    global GKREAOPT
    names = {'distGate','latePhaseRel','degenPCA','alignW', ...
             'eliteTrunc','localFix','exploreFloor','localBatch'};
    S = struct();
    for k = 1:numel(names)
        S.(names{k}) = logical(on);
    end
    GKREAOPT = S;
end
