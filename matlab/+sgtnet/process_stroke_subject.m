function data = process_stroke_subject(data_root, subject_id, cfg, group)
%PROCESS_STROKE_SUBJECT  Preprocess all sessions of one HEFMI-ICH subject.
%
%   DATA = PROCESS_STROKE_SUBJECT(DATA_ROOT, SUBJECT_ID, CFG) loads the
%   pre-epoched sessions of one patient, resamples them to 200 Hz / 10 Hz,
%   filters both modalities, crops the common [-5, 20] s analysis window
%   around the cue and applies baseline correction.
%
%   The clinical release already provides HbO/HbR concentration changes, so
%   the Beer-Lambert step is skipped. The epochs start 10 s before the cue
%   (see MakeDatasetFromRaw.m in the dataset release); the cue is therefore
%   at 10 s and the analysis window is cut around it. The rest of the
%   protocol is identical to the benchmark branch:
%
%       EEG   : 4-35 Hz, 6th-order zero-phase Butterworth, CAR
%       fNIRS : 0.01-0.1 Hz, 6th-order zero-phase Butterworth
%       Epoch : [-5, 20] s; baseline [-3, 0] s
%
%   DATA.TASKS.MI contains EEG, HBO, HBR, LABELS (1/2), CLASS_NAMES,
%   N_TRIALS, EEG_CLAB and NIRS_CLAB.
%
%   See also SGTPNET.BUILD_STROKE_DATASET, SGTPNET.FILTER_BANDPASS.

if nargin < 4 || isempty(group)
    group = 'Patients';
end

raw = sgtnet.load_stroke_raw(data_root, subject_id, group);

session = struct('eeg', {}, 'hbo', {}, 'hbr', {}, 'labels', {});
eeg_clab = {};

for s = 1:raw.n_sessions
    ctx = sprintf('stroke sub %02d session %d', subject_id, raw.session_num(s));
    epo_eeg = raw.eeg{s};
    epo_nirs = raw.nirs{s};

    if s == 1
        if isfield(epo_eeg, 'channelsName') && ~isempty(epo_eeg.channelsName)
            eeg_clab = cellstr(string(epo_eeg.channelsName(:)));
        else
            eeg_clab = arrayfun(@(i) sprintf('EEG%02d', i), ...
                1:size(epo_eeg.X, 2), 'UniformOutput', false);
        end
    end

    % ---- EEG -----------------------------------------------------------
    x = double(epo_eeg.X);
    fs_eeg = double(epo_eeg.fs);
    x = sgtnet.resample_3d(x, fs_eeg, cfg.fs.eeg, [ctx ' EEG']);
    x = sgtnet.filter_bandpass(x, cfg.fs.eeg, cfg.eeg.band_hz, ...
        cfg.eeg.filter_order, [ctx ' EEG']);
    x = x - mean(x, 2);                       % common average reference

    % ---- fNIRS ---------------------------------------------------------
    hbo = double(epo_nirs.oxy.X);
    hbr = double(epo_nirs.doxy.X);
    fs_nirs = double(epo_nirs.oxy.fs);
    hbo = sgtnet.resample_3d(hbo, fs_nirs, cfg.fs.nirs, [ctx ' HbO']);
    hbr = sgtnet.resample_3d(hbr, fs_nirs, cfg.fs.nirs, [ctx ' HbR']);
    hbo = sgtnet.filter_bandpass(hbo, cfg.fs.nirs, cfg.nirs.band_hz, ...
        cfg.nirs.filter_order, [ctx ' HbO']);
    hbr = sgtnet.filter_bandpass(hbr, cfg.fs.nirs, cfg.nirs.band_hz, ...
        cfg.nirs.filter_order, [ctx ' HbR']);
    hbo = hbo * cfg.nirs.stroke_unit_scale;   % mol/L -> mmol/L
    hbr = hbr * cfg.nirs.stroke_unit_scale;

    % ---- Common analysis window around the cue -------------------------
    cue_s = 10;                               % epoch starts 10 s before cue
    idx_eeg = window_indices(size(x, 1), cfg.fs.eeg, cue_s, cfg.epoch_s);
    idx_nirs = window_indices(size(hbo, 1), cfg.fs.nirs, cue_s, cfg.epoch_s);

    x = x(idx_eeg, :, :);
    hbo = hbo(idx_nirs, :, :);
    hbr = hbr(idx_nirs, :, :);

    x = sgtnet.baseline_correct(x, cfg.fs.eeg, cfg.epoch_s, cfg.baseline_s, ctx);
    hbo = sgtnet.baseline_correct(hbo, cfg.fs.nirs, cfg.epoch_s, cfg.baseline_s, ctx);
    hbr = sgtnet.baseline_correct(hbr, cfg.fs.nirs, cfg.epoch_s, cfg.baseline_s, ctx);

    % ---- Labels and trial consistency ----------------------------------
    labels = double(epo_eeg.y(:)) + 1;
    labels_nirs = double(epo_nirs.oxy.y(:)) + 1;
    if numel(labels) ~= size(x, 3) || numel(labels) ~= size(hbo, 3)
        error('sgtnet:process_stroke_subject:trialMismatch', ...
            '%s: %d labels, %d EEG trials, %d fNIRS trials.', ...
            ctx, numel(labels), size(x, 3), size(hbo, 3));
    end
    if ~isequal(labels, labels_nirs)
        error('sgtnet:process_stroke_subject:labelMismatch', ...
            '%s: EEG and fNIRS labels differ.', ctx);
    end

    session(s).eeg = x;
    session(s).hbo = hbo;
    session(s).hbr = hbr;
    session(s).labels = labels;
end

data = struct();
data.subject_id = subject_id;
data.group = group;
data.eeg_clab = eeg_clab;
data.nirs_clab = arrayfun(@(i) sprintf('NIRS%02d', i), ...
    1:size(session(1).hbo, 2), 'UniformOutput', false);

task = struct();
task.eeg = cat(3, session.eeg);
task.hbo = cat(3, session.hbo);
task.hbr = cat(3, session.hbr);
task.labels = vertcat(session.labels);
task.class_names = {'Left', 'Right'};
task.n_trials = numel(task.labels);

if size(task.eeg, 3) ~= task.n_trials || ...
        size(task.hbo, 3) ~= task.n_trials || ...
        size(task.hbr, 3) ~= task.n_trials
    error('sgtnet:process_stroke_subject:assemblyMismatch', ...
        'Subject %d: inconsistent trial counts after assembly.', subject_id);
end

data.tasks = struct();
data.tasks.MI = task;

end

function idx = window_indices(n_time, fs, cue_s, epoch_s)
%WINDOW_INDICES  Sample indices of [cue + epoch_s(1), cue + epoch_s(2)].
cue_idx = round(cue_s * fs) + 1;
idx = (cue_idx + round(epoch_s(1) * fs)):(cue_idx + round(epoch_s(2) * fs));
if idx(1) < 1 || idx(end) > n_time
    error('sgtnet:process_stroke_subject:badWindow', ...
        'Analysis window [%d, %d] exceeds the epoch length %d.', ...
        idx(1), idx(end), n_time);
end
end
