function S = gkrea_state(action,payload)
%gkrea_state - 演进状态的黑板（blackboard），供自适应开关读取。
%
%   gkrea_state('reset')          清空
%   gkrea_state('update',S)       以结构体 S 的字段增量更新（未给出的字段保留）
%   S = gkrea_state('get')        读取当前状态
%
%   字段（由各模块在每轮内发布，缺省为 NaN = 未知）：
%     iter        当前轮次
%     prog        (FE-NI)/(maxFE-NI)，初始采样之后的预算进度 ∈ [0,1]
%     err         代理模型的 EMA 相对误差
%     stag        连续未改善轮数
%     M,D,N       问题规模
%     nDirs       参考方向数 |W|
%     nFront      NDSort 第一层的规模
%     effRankRel  协方差谱参与率 / D ∈ (0,1]
%     dmOutRobust 候选点中马氏距离超过「median+2MAD」的比例
%     activeRatio 档案占用的参考方向比例
%     jaccard     相邻两轮占用方向集合的 Jaccard 相似度
%     localYield  局部探索相对收益的 EMA
%
%   设计要点：所有字段都可能在某轮尚未更新，规则函数必须对 NaN 保守处理
%   （返回原行为），避免第一轮因为「状态未知」而改变算法。

    global GKREASTATE
    if nargin < 2, payload = struct(); end

    if isempty(GKREASTATE) || ~isstruct(GKREASTATE) || ~isfield(GKREASTATE,'ready')
        GKREASTATE = defaultState();
    end

    switch lower(action)
        case 'reset'
            GKREASTATE = defaultState();
        case 'update'
            if ~isstruct(payload), return; end
            f = fieldnames(payload);
            for i = 1:numel(f)
                GKREASTATE.(f{i}) = payload.(f{i});
            end
            GKREASTATE.ready = true;
        case 'get'
            % 直接返回
        otherwise
            error('gkrea_state: 未知动作 %s',action);
    end
    S = GKREASTATE;
end

function S = defaultState()
    S = struct('ready',false,'iter',0,'prog',0,'err',NaN,'stag',0, ...
        'M',0,'D',0,'N',0,'nDirs',0,'nFront',0,'effRankRel',NaN, ...
        'dmOutRobust',NaN,'activeRatio',NaN,'activeMax',NaN,'jaccard',NaN,'localYield',0,'active',[],'gateOn',false);
end
