function summary = build_public_dataset(cfg, opts)
%BUILD_PUBLIC_DATASET  Generate the benchmark HDF5 file used by SGT-Net.
%
%   SUMMARY = BUILD_PUBLIC_DATASET(CFG, OPTS) preprocesses the Open Access
%   EEG+NIRS dataset (Shin et al., 2017) and writes
%
%       <OPTS.OUTPUT_DIR>/hybridBCI_Tn8.h5
%
%   The file contains both benchmark tasks:
%       /MI  motor imagery, left vs. right hand
%       /MA  mental arithmetic, rest vs. arithmetic
%
%   Each subject contributes 60 trials per task (3 sessions x 20 trials) and
%   8 sliding windows per trial, i.e. 480 samples per task. TRIAL_IDS
%   identifies the parent trial of every window.
%
%   OPTS fields: PUBLIC_ROOT, OUTPUT_DIR, SUBJECTS, OVERWRITE, VERBOSE.
%
%   See also RUN_PREPROCESSING, SGTPNET.PROCESS_PUBLIC_SUBJECT.

out_file = fullfile(opts.output_dir, 'hybridBCI_Tn8.h5');
if exist(out_file, 'file')
    if opts.overwrite
        delete(out_file);
    else
        error('sgtnet:build_public_dataset:fileExists', ...
            'Output file already exists: %s (set ''Overwrite'', true).', out_file);
    end
end

subjects = opts.subjects;
if isempty(subjects)
    subjects = 1:29;
end

layout = reference_layout(opts.public_root, subjects, cfg);
n_win = windows_per_trial(cfg);

info = struct();
info.dataset_name = 'Open Access EEG+NIRS Dataset (Shin et al., 2017)';
info.dataset_key = 'public';
info.dataset_citation = ['Shin, J., von Luehmann, A., Blankertz, B., Kim, D.-W., ' ...
    'Jeong, J., Hwang, H.-J., & Mueller, K.-R. (2017). Open access dataset ' ...
    'for EEG+NIRS single-trial classification. IEEE Transactions on Neural ' ...
    'Systems and Rehabilitation Engineering, 25(10), 1735-1745.'];
info.dataset_url = 'http://doc.ml.tu-berlin.de/hBCI';
info.tasks = {'MI', 'MA'};
info.class_names = {{'Left', 'Right'}, {'Rest', 'Arithmetic'}};
info.eeg_clab = layout.eeg_clab;
info.nirs_clab = layout.nirs_clab;
info.eeg_mnt_original = layout.eeg_mnt;
info.nirs_mnt_original = layout.nirs_mnt;
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
summary.n_trials = struct('MI', 0, 'MA', 0);
summary.n_samples = struct('MI', 0, 'MA', 0);
summary.n_windows = n_win;

for i = 1:numel(subjects)
    sid = subjects(i);
    log_msg(cfg, '[public %d/%d] subject %02d', i, numel(subjects), sid);
    try
        data = sgtnet.process_public_subject(opts.public_root, sid, cfg);
        wrote_any = false;

        for t = 1:numel(info.tasks)
            task_name = info.tasks{t};
            task = data.tasks.(task_name);
            if task.n_trials == 0
                warning('sgtnet:build_public_dataset:emptyTask', ...
                    'Subject %02d task %s has no trials; skipped.', sid, task_name);
                continue;
            end

            payload = sgtnet.make_payload(task, cfg);
            sgtnet.h5_write_subject(out_file, task_name, sid, payload, cfg);

            summary.n_trials.(task_name) = summary.n_trials.(task_name) + task.n_trials;
            summary.n_samples.(task_name) = summary.n_samples.(task_name) + numel(payload.labels);
            wrote_any = true;

            log_msg(cfg, '    %s: %d trials, %d samples (%d windows/trial)', ...
                task_name, task.n_trials, numel(payload.labels), payload.n_windows);
        end

        if wrote_any
            summary.n_subjects_written = summary.n_subjects_written + 1;
        end
    catch ME
        warning('sgtnet:build_public_dataset:subjectFailed', ...
            'Subject %02d failed: %s', sid, ME.message);
        summary.failed_subjects(end+1) = sid;
    end
end

log_msg(cfg, 'Done: %s', out_file);
log_msg(cfg, '  subjects written: %d/%d, failed: %d', ...
    summary.n_subjects_written, summary.n_subjects_requested, ...
    numel(summary.failed_subjects));
log_msg(cfg, '  MI: %d trials, %d samples', ...
    summary.n_trials.MI, summary.n_samples.MI);
log_msg(cfg, '  MA: %d trials, %d samples', ...
    summary.n_trials.MA, summary.n_samples.MA);

try
    h5writeatt(out_file, '/', 'n_subjects_written', summary.n_subjects_written);
    for t = 1:numel(info.tasks)
        h5writeatt(out_file, ['/' info.tasks{t}], ...
            'n_subjects_written', summary.n_subjects_written);
    end
catch ME
    warning('sgtnet:build_public_dataset:attrFailed', ...
        'Could not update n_subjects_written: %s', ME.message);
end

end

% =========================================================================
function layout = reference_layout(data_root, subjects, cfg)
%REFERENCE_LAYOUT  Channel labels and montages shared by all subjects.
layout = struct('eeg_clab', {{}}, 'nirs_clab', {{}}, ...
    'eeg_mnt', [], 'nirs_mnt', []);

for i = 1:numel(subjects)
    sid = subjects(i);
    eeg_dir = fullfile(data_root, 'EEG', sprintf('subject %02d', sid), ...
        'with occular artifact');
    nirs_dir = fullfile(data_root, 'NIRS', sprintf('subject %02d', sid));
    if ~exist(fullfile(eeg_dir, 'cnt.mat'), 'file') || ...
            ~exist(fullfile(nirs_dir, 'mnt.mat'), 'file')
        continue;
    end

    S = load(fullfile(eeg_dir, 'cnt.mat'), 'cnt');
    clab = cellstr(S.cnt{1}.clab);
    clab = clab(~contains(clab, cfg.eeg.eog_pattern));
    S = load(fullfile(eeg_dir, 'mnt.mat'));
    f = fieldnames(S);
    eeg_mnt = sgtnet.subset_montage(S.(f{1}), clab, 'reference EEG montage');

    S = load(fullfile(nirs_dir, 'mnt.mat'));
    f = fieldnames(S);
    nirs_mnt = S.(f{1});
    if isfield(nirs_mnt, 'clab')
        nirs_clab = cellstr(nirs_mnt.clab);
    else
        S = load(fullfile(nirs_dir, 'cnt.mat'), 'cnt');
        nirs_clab = cellstr(S.cnt{1}.clab(1:numel(S.cnt{1}.clab) / 2));
    end

    layout.eeg_clab = clab;
    layout.nirs_clab = nirs_clab;
    layout.eeg_mnt = eeg_mnt;
    layout.nirs_mnt = sgtnet.subset_montage(nirs_mnt, nirs_clab, ...
        'reference fNIRS montage');
    return;
end

error('sgtnet:build_public_dataset:noReferenceSubject', ...
    'Could not find any subject with EEG and fNIRS files under %s.', data_root);

end

% =========================================================================
function n_win = windows_per_trial(cfg)
n_eeg = numel(cfg.window.eeg_range_s(1):cfg.window.eeg_step_s: ...
    (cfg.window.eeg_range_s(2) - cfg.window.eeg_size_s));
n_nirs = numel(cfg.window.nirs_range_s(1):cfg.window.nirs_step_s: ...
    (cfg.window.nirs_range_s(2) - cfg.window.nirs_size_s));
if n_eeg ~= n_nirs
    error('sgtnet:build_public_dataset:windowMismatch', ...
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

