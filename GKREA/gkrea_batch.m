function nOut = gkrea_batch(State,alpha,localCap,localUsed)
%gkrea_batch - 局部探索的批量（自适应或原始）。
%
%   自适应模式（GKREAOPT.mode='auto' 或 localBatch 显式开启）：
%     按上一次局部探索的**观测收益** localYield 决定 2~4 个点
%     （收益 = 归一化 Tchebycheff 最优值的相对下降）。收益接近 0 时退回 2 点，
%     避免把预算丢给毫无进展的局部搜索。
%   原始模式：2 + double(State.errLow)。

    if gkrea_opt('localBatch')
        y = 0;
        if isfield(State,'localYield'), y = State.localYield; end
        nOut = 2 + double(y > 1e-3) + double(y > 0.05);      % 2 / 3 / 4
        nOut = min(nOut, max(2,ceil(alpha/2)));              % 不超过半个批次
    else
        nOut = 2 + double(State.errLow);
    end
    nOut = min(nOut, localCap - localUsed);
    nOut = max(nOut, 0);
end
