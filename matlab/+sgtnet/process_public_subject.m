function data = process_public_subject(data_root, subject_id, cfg)
%PROCESS_PUBLIC_SUBJECT  Preprocess all sessions of one benchmark subject.
%
%   DATA = PROCESS_PUBLIC_SUBJECT(DATA_ROOT, SUBJECT_ID, CFG) loads the six
%   sessions of the Open Access EEG+NIRS dataset, preprocesses each session
%   independently and assembles the MI (sessions 1, 3, 5) and MA (sessions
%   2, 4, 6) tasks.
%
%   DATA.EEG_CLAB and DATA.NIRS_CLAB contain the channel labels.
%   DATA.TASKS.MI and DATA.TASKS.MA contain the fields EEG, HBO, HBR and
%   LABELS.
%
%   Each session is filtered before segmentation, which matches the
%   "continuous data segmented into trials" description of Section IV and
%   avoids edge transients at artificial session boundaries.
%
%   See also SGTPNET.BUILD_PUBLIC_DATASET, SGTPNET.PREPROCESS_EEG,
%   SGTPNET.PREPROCESS_NIRS.

raw = sgtnet.load_public_raw(data_root, subject_id);
if raw.n_sessions < 6
    warning('sgtnet:process_public_subject:fewerSessions', ...
        'Subject %02d has %d sessions; expected 6.', subject_id, raw.n_sessions);
end

session = struct('eeg', {}, 'hbo', {}, 'hbr', {}, 'labels', {});

for s = 1:raw.n_sessions
    ctx_eeg = sprintf('public sub %02d session %d EEG', subject_id, s);
    ctx_nirs = sprintf('public sub %02d session %d fNIRS', subject_id, s);

    [eeg_epo, eeg_lab, eeg_clab, valid_e] = ...
        sgtnet.preprocess_eeg(raw.eeg_cnt{s}, raw.eeg_mrk{s}, cfg, ctx_eeg);
    [hbo, hbr, nirs_lab, nirs_clab, ~, valid_n] = ...
        sgtnet.preprocess_nirs(raw.nirs_cnt{s}, raw.nirs_mrk{s}, cfg, ctx_nirs);

    if numel(valid_e) ~= numel(valid_n) || any(valid_e ~= valid_n)
        error('sgtnet:process_public_subject:trialMismatch', ...
            ['Subject %02d session %d: EEG and fNIRS trials do not match ' ...
             '(%d vs %d valid markers).'], ...
            subject_id, s, sum(valid_e), sum(valid_n));
    end
    if ~isempty(eeg_lab) && ~isempty(nirs_lab) && ~isequal(eeg_lab, nirs_lab)
        error('sgtnet:process_public_subject:labelMismatch', ...
            'Subject %02d session %d: EEG and fNIRS labels differ.', ...
            subject_id, s);
    end

    session(s).eeg = eeg_epo;
    session(s).hbo = hbo;
    session(s).hbr = hbr;
    session(s).labels = eeg_lab;
end

data = struct();
data.subject_id = subject_id;
data.eeg_clab = eeg_clab;
if isfield(raw.nirs_mnt, 'clab') && numel(raw.nirs_mnt.clab) == size(session(1).hbo, 2)
    % The montage holds the clean physiological channel names (without the
    % "oxy"/"deoxy" suffix added by the Beer-Lambert conversion).
    data.nirs_clab = cellstr(raw.nirs_mnt.clab);
else
    data.nirs_clab = nirs_clab;
    warning('sgtnet:process_public_subject:noNirsMontage', ...
        'Subject %02d: fNIRS montage is missing or has the wrong size.', ...
        subject_id);
end
data.tasks = struct();

task_defs = struct( ...
    'name', {'MI', 'MA'}, ...
    'sessions', {[1 3 5], [2 4 6]});
task_defs(1).class_names = {'Left', 'Right'};
task_defs(2).class_names = {'Rest', 'Arithmetic'};

for t = 1:numel(task_defs)
    def = task_defs(t);
    idx = def.sessions(def.sessions <= numel(session));
    if isempty(idx)
        data.tasks.(def.name) = empty_task();
        warning('sgtnet:process_public_subject:noTaskSessions', ...
            'Subject %02d: no sessions available for task %s.', ...
            subject_id, def.name);
        continue;
    end

    task = struct();
    task.eeg = cat(3, session(idx).eeg);
    task.hbo = cat(3, session(idx).hbo);
    task.hbr = cat(3, session(idx).hbr);
    task.labels = vertcat(session(idx).labels);
    task.class_names = def.class_names;
    task.n_trials = numel(task.labels);

    if size(task.eeg, 3) ~= task.n_trials || ...
            size(task.hbo, 3) ~= task.n_trials || ...
            size(task.hbr, 3) ~= task.n_trials
        error('sgtnet:process_public_subject:assemblyMismatch', ...
            'Subject %02d task %s: inconsistent trial counts.', ...
            subject_id, def.name);
    end

    data.tasks.(def.name) = task;
end

end

function task = empty_task()
task = struct('eeg', [], 'hbo', [], 'hbr', [], 'labels', [], ...
    'class_names', {{}}, 'n_trials', 0);
end
