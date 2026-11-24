function payload = make_payload(task, cfg)
%MAKE_PAYLOAD  Convert epoched trials into model-ready sliding-window arrays.
%
%   PAYLOAD = MAKE_PAYLOAD(TASK, CFG) applies the asymmetric sliding-window
%   segmentation to one subject/task.
%
%   TASK must contain EEG, HBO, HBR (each [time, channels, trials]) and
%   LABELS ([trials, 1]). The returned structure contains:
%
%       EEG_DATA, HBO_DATA, HBR_DATA   [samples, channels, time]
%       LABELS                         [samples, 1], 1-based
%       TRIAL_IDS                      [samples, 1], 1-based
%       N_WINDOWS                      windows per trial
%       STARTS_S                       window onset times
%
%   Windows are stored in trial-major order, so TRIAL_IDS is each trial
%   index repeated N_WINDOWS times.
%
%   See also SGTPNET.MAKE_WINDOWS, SGTPNET.ZSCORE_WINDOWS.

n_trials = numel(task.labels);
if n_trials == 0
    error('sgtnet:make_payload:emptyTask', 'Task has no trials.');
end
if size(task.eeg, 3) ~= n_trials || size(task.hbo, 3) ~= n_trials || ...
        size(task.hbr, 3) ~= n_trials
    error('sgtnet:make_payload:trialMismatch', ...
        'EEG/HbO/HbR/label trial counts differ.');
end

[eeg_w, starts_eeg] = sgtnet.make_windows(task.eeg, cfg.fs.eeg, ...
    cfg.epoch_s, cfg.window.eeg_range_s, cfg.window.eeg_size_s, ...
    cfg.window.eeg_step_s, 'EEG windows');
[hbo_w, starts_nirs] = sgtnet.make_windows(task.hbo, cfg.fs.nirs, ...
    cfg.epoch_s, cfg.window.nirs_range_s, cfg.window.nirs_size_s, ...
    cfg.window.nirs_step_s, 'HbO windows');
[hbr_w, ~] = sgtnet.make_windows(task.hbr, cfg.fs.nirs, ...
    cfg.epoch_s, cfg.window.nirs_range_s, cfg.window.nirs_size_s, ...
    cfg.window.nirs_step_s, 'HbR windows');

if ~isequal(starts_eeg, starts_nirs)
    error('sgtnet:make_payload:windowMismatch', ...
        'EEG and fNIRS window onset grids differ.');
end
n_win = numel(starts_eeg);
if size(eeg_w, 1) ~= n_trials * n_win || size(hbo_w, 1) ~= n_trials * n_win
    error('sgtnet:make_payload:windowCountMismatch', ...
        'Unexpected number of windows.');
end

payload = struct();
payload.n_windows = n_win;
payload.starts_s = starts_eeg;
payload.labels = repelem(task.labels(:), n_win);
payload.trial_ids = repelem((1:n_trials)', n_win);

if strcmpi(cfg.normalization, 'zscore')
    payload.eeg_data = sgtnet.zscore_windows(eeg_w);
    payload.hbo_data = sgtnet.zscore_windows(hbo_w);
    payload.hbr_data = sgtnet.zscore_windows(hbr_w);
else
    payload.eeg_data = eeg_w;
    payload.hbo_data = hbo_w;
    payload.hbr_data = hbr_w;
end

end
