function State = StateMonitor(arg1,arg2,arg3,arg4)
%StateMonitor - Optimization-state monitor and adaptive weight controller.
%
%   State = StateMonitor('init',kappa0) initializes the state struct.
%
%   State = StateMonitor(State,C,A2old,info) updates the state after the
%   infill batch C has been expensively evaluated:
%     err     - EMA of the normalized surrogate prediction error,
%     stag    - consecutive iterations without domination progress,
%     kappa   - exploration weight of the LCB (grows with stagnation),
%     wX      - weight of the exploration score in the infill criterion
%               (grows with prediction error and stagnation),
%     trigger - true when a budgeted local exploration should be launched.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if ischar(arg1)
        %% Initialization mode: StateMonitor('init',kappa0)
        State.kappa0  = arg2;
        State.kappa   = arg2;
        State.wX      = 0.25;
        State.stag    = 0;
        State.err     = 0.1;
        State.trigger = false;
        State.triggerReason = 'none';
        State.progress = false;
        return;
    end

    %% Update mode: StateMonitor(State,C,A2old,info)
    State = arg1;
    C     = arg2;
    A2old = arg3;
    info  = arg4;

    %% Prediction error (normalized per objective, EMA-smoothed)
    if ~isempty(C) && ~isempty(info.predObj)
        range = max(max(A2old.objs,[],1) - min(A2old.objs,[],1),1e-12);
        e     = abs(C.objs - info.predObj)./range;
        newErr = median(mean(e,2));
        State.err = 0.5.*State.err + 0.5.*min(newErr,2);
    end

    %% Domination progress of the batch against the old ND front
    progress = false;
    if ~isempty(C)
        [F,~] = NDSort(A2old.objs,1);
        Front = A2old(F==1).objs;
        Cobjs = C.objs;
        for i = 1 : size(Cobjs,1)
            dom = all(Cobjs(i,:) <= Front,2) & any(Cobjs(i,:) < Front,2);
            if any(dom)
                progress = true;
                break;
            end
        end
    end

    %% Stagnation counter and adaptive weights
    if progress
        State.stag = 0;
    else
        State.stag = State.stag + 1;
    end
    State.kappa = min(max(State.kappa0.*(1+0.5.*State.stag),0.5),6);
    State.wX    = min(0.8,0.25 + 0.4.*double(State.err > 0.10) + 0.1.*State.stag);
    % Trigger rule. Base: stagnation triggers a local exploration. Extra:
    % on DEGENERATE fronts (the archive front activates less than half of
    % the reference directions) whose surrogate is highly accurate, a first
    % stall already justifies a precision refinement, because there the IGD
    % is limited by the coverage density ALONG the low-dimensional front --
    % exactly what a local refinement sharpens. On regular (full-front)
    % problems the champion infill already spreads the budget well, so the
    % eager mode stays disabled and their behaviour is unchanged.
    if isfield(info,'degenerate')
        deg = info.degenerate;
    else
        deg = false;
    end
    if isfield(info,'latePhase')
        late = info.latePhase;   % front shape is only trustworthy late on
    else
        late = false;
    end
    State.degenerate = deg;
    State.errLow  = State.err < 0.01;
    State.trigger = (State.stag >= 2) || (State.errLow && State.stag >= 1 && deg && late);
    % [MON] Read-only diagnostics: why the trigger fired (or not).
    State.progress = progress;
    if State.trigger
        if State.stag >= 2
            State.triggerReason = 'stagnation';
        else
            State.triggerReason = 'eager-degenerate';
        end
    else
        State.triggerReason = 'none';
    end
end
