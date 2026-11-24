function summary = build_stroke_dataset(cfg, opts)
%BUILD_STROKE_DATASET  Generate the clinical HDF5 file used by SGT-Net.
%
%   SUMMARY = BUILD_STROKE_DATASET(CFG, OPTS) preprocesses the HEFMI-ICH
%   patients (Shi et al., 2025) and writes
%
%       <OPTS.OUTPUT_DIR>/stroke_patients_Tn8.h5
%
%   Only the 20 patients are written; the 17 healthy controls distributed
%   with HEFMI-ICH are not used by the default clinical pipeline.
%
%   Each subject contributes 30 trials per available session (1-3 sessions,
%   i.e. up to 90 trials) and 8 sliding windows per trial. As for the
%   benchmark dataset, only the sliding-window tensors are stored and
%   TRIAL_IDS identifies the parent trial of every window.
%
%   OPTS fields: STROKE_ROOT, OUTPUT_DIR, SUBJECTS, OVERWRITE, VERBOSE.
%
%   See also RUN_PREPROCESSING, SGTPNET.PROCESS_STROKE_SUBJECT.

out_file = fullfile(opts.output_dir, 'stroke_patients_Tn8.h5');
if exist(out_file, 'file')
    if opts.overwrite
        delete(out_file);
    else
        error('sgtnet:build_stroke_dataset:fileExists', ...
            'Output file already exists: %s (set ''Overwrite'', true).', out_file);
    end
end

subjects = opts.subjects;
if isempty(subjects)
    subjects = 1:20;
end

layout = reference_layout(opts.stroke_root, subjects);
n_win = windows_per_trial(cfg);

info = struct();
info.dataset_name = 'HEFMI-ICH: Hybrid EEG-fNIRS Motor Imagery Dataset for ICH';
info.dataset_key = 'stroke';
info.dataset_citation = ['Shi, J., Chen, D., Zhao, X., Zhao, Z., Li, S., Xu, Y., ' ...
    'Ding, T., Zhu, Z., Zhang, P., Ye, Q., et al. (2025). HEFMI-ICH: A hybrid ' ...
    'EEG-fNIRS motor imagery dataset for brain-computer interface in ' ...
    'intracerebral hemorrhage. Scientific Data, 12, 1816.'];
info.dataset_url = 'https://doi.org/10.6084/m9.figshare.28955456.v4';
info.group = 'Patients';
info.tasks = {'MI'};
info.class_names = {{'Left', 'Right'}};
info.eeg_clab = layout.eeg_clab;
info.nirs_clab = layout.nirs_clab;
info.eeg_mnt_original = [];
info.nirs_mnt_original = [];
info.eeg_units = 'uV';
info.nirs_units = 'mmol/L';
info.n_subjects = numel(subjects);
info.n_windows = n_win;

log_msg(cfg, 'Creating %s', out_file);
sgtnet.h5_init(out_file, cfg, info);

summary = struct();
summary.output_file = out_file;
summary.n_subjects_requested = numel(subjects);
summary.n_subjects_written = 0;
summary.failed_subjects = [];
summary.n_trials = 0;
summary.n_samples = 0;
summary.n_windows = n_win;

for i = 1:numel(subjects)
    sid = subjects(i);
    log_msg(cfg, '[stroke %d/%d] subject %d', i, numel(subjects), sid);
    try
        data = sgtnet.process_stroke_subject(opts.stroke_root, sid, cfg, 'Patients');
        task = data.tasks.MI;
        if task.n_trials == 0
            warning('sgtnet:build_stroke_dataset:emptyTask', ...
                'Subject %d has no trials; skipped.', sid);
            continue;
        end

        payload = sgtnet.make_payload(task, cfg);
        sgtnet.h5_write_subject(out_file, 'MI', sid, payload, cfg);

        summary.n_subjects_written = summary.n_subjects_written + 1;
        summary.n_trials = summary.n_trials + task.n_trials;
        summary.n_samples = summary.n_samples + numel(payload.labels);

        log_msg(cfg, '    MI: %d trials, %d samples (%d windows/trial)', ...
            task.n_trials, numel(payload.labels), payload.n_windows);
    catch ME
        warning('sgtnet:build_stroke_dataset:subjectFailed', ...
            'Subject %d failed: %s', sid, ME.message);
        summary.failed_subjects(end+1) = sid;
    end
end

log_msg(cfg, 'Done: %s', out_file);
log_msg(cfg, '  subjects written: %d/%d, failed: %d', ...
    summary.n_subjects_written, summary.n_subjects_requested, ...
    numel(summary.failed_subjects));
log_msg(cfg, '  MI: %d trials, %d samples', ...
    summary.n_trials, summary.n_samples);

try
    h5writeatt(out_file, '/', 'n_subjects_written', summary.n_subjects_written);
    h5writeatt(out_file, '/MI', 'n_subjects_written', summary.n_subjects_written);
catch ME
    warning('sgtnet:build_stroke_dataset:attrFailed', ...
        'Could not update n_subjects_written: %s', ME.message);
end

end

% =========================================================================
function layout = reference_layout(data_root, subjects)
%REFERENCE_LAYOUT  Channel labels shared by all clinical subjects.
layout = struct('eeg_clab', {{}}, 'nirs_clab', {{}});

for i = 1:numel(subjects)
    sid = subjects(i);
    eeg_dir = fullfile(data_root, 'Patients', 'EEG', sprintf('subject%d', sid));
    nirs_dir = fullfile(data_root, 'Patients', 'NIRS', sprintf('subject%d', sid));
    eeg_files = dir(fullfile(eeg_dir, '*_epo.mat'));
    nirs_files = dir(fullfile(nirs_dir, '*_epo.mat'));
    if isempty(eeg_files) || isempty(nirs_files)
        continue;
    end

    S = load(fullfile(eeg_dir, eeg_files(1).name), 'epo');
    if isfield(S.epo, 'channelsName') && ~isempty(S.epo.channelsName)
        layout.eeg_clab = cellstr(string(S.epo.channelsName(:)));
    else
        layout.eeg_clab = arrayfun(@(k) sprintf('EEG%02d', k), ...
            1:size(S.epo.X, 2), 'UniformOutput', false);
    end

    N = load(fullfile(nirs_dir, nirs_files(1).name), 'epo');
    n_chan = size(N.epo.oxy.X, 2);
    layout.nirs_clab = arrayfun(@(k) sprintf('NIRS%02d', k), ...
        1:n_chan, 'UniformOutput', false);
    return;
end

error('sgtnet:build_stroke_dataset:noReferenceSubject', ...
    'Could not find any subject with EEG and fNIRS files under %s.', data_root);

end

% =========================================================================
function n_win = windows_per_trial(cfg)
n_eeg = numel(cfg.window.eeg_range_s(1):cfg.window.eeg_step_s: ...
    (cfg.window.eeg_range_s(2) - cfg.window.eeg_size_s));
n_nirs = numel(cfg.window.nirs_range_s(1):cfg.window.nirs_step_s: ...
    (cfg.window.nirs_range_s(2) - cfg.window.nirs_size_s));
if n_eeg ~= n_nirs
    error('sgtnet:build_stroke_dataset:windowMismatch', ...
        'EEG and fNIRS configurations produce %d vs %d windows per trial.', ...
        n_eeg, n_nirs);
end
n_win = n_eeg;
end

% =========================================================================
function log_msg(cfg, fmt, varargin)
if cfg.verbose
    fprintf([fmt '\n'], varargin{:});
end
end
