function report = verify_dataset(h5_path, varargin)
%VERIFY_DATASET  Sanity-check an SGT-Net HDF5 file.
%
%   REPORT = VERIFY_DATASET(H5_PATH) checks the structure produced by
%   RUN_PREPROCESSING:
%
%     * required root datasets and task groups,
%     * required subject datasets and their shapes,
%     * labels are 1-based and binary,
%     * trial_ids form consecutive blocks of N_WINDOWS samples,
%     * EEG and fNIRS sample counts agree.
%
%   Name-value options:
%       'CheckValues'  also scan the data arrays for NaN/Inf (default false;
%                      this reads the full file and can take a while).
%
%   Example:
%       report = verify_dataset('datasets/hybridBCI_Tn8.h5');
%
%   See also RUN_PREPROCESSING.

p = inputParser();
p.addParameter('CheckValues', false, @(x) islogical(x) || isnumeric(x));
p.parse(varargin{:});
check_values = logical(p.Results.CheckValues);

if ~exist(h5_path, 'file')
    error('verify_dataset:missingFile', 'File not found: %s', h5_path);
end

info = h5info(h5_path);
root_names = {info.Datasets.Name};
required_root = {'eeg_fs', 'nirs_fs', 'eeg_units', 'nirs_units', ...
    'eeg_channels', 'nirs_channels'};
for k = 1:numel(required_root)
    if ~ismember(required_root{k}, root_names)
        error('verify_dataset:missingRoot', ...
            'Missing root dataset /%s.', required_root{k});
    end
end

all_groups = {info.Groups.Name};
task_names = {};
for k = 1:numel(all_groups)
    child = h5info(h5_path, all_groups{k});
    if ~isempty(child.Groups) && ...
            any(startsWith({child.Groups.Name}, [all_groups{k} '/subject_']))
        task_names{end+1} = strrep(all_groups{k}, '/', ''); %#ok<AGROW>
    end
end
if isempty(task_names)
    error('verify_dataset:noTasks', 'No task groups found.');
end

report = struct();
report.file = h5_path;
report.tasks = struct();

fprintf('File: %s\n', h5_path);
fprintf('Tasks: %s\n', strjoin(task_names, ', '));

for t = 1:numel(task_names)
    task = task_names{t};
    task_info = h5info(h5_path, ['/' task]);
    subject_groups = {task_info.Groups.Name};
    subject_groups = subject_groups(startsWith(subject_groups, ['/' task '/subject_']));
    n_win = double(h5readatt(h5_path, ['/' task], 'n_windows_per_trial'));

    fprintf('\n[%s] %d subject(s), %d windows/trial\n', ...
        task, numel(subject_groups), n_win);

    n_total = 0;
    for s = 1:numel(subject_groups)
        g = subject_groups{s};
        g_info = h5info(h5_path, g);
        names = {g_info.Datasets.Name};

        required = {'eeg_data', 'hbo_data', 'hbr_data', ...
            'labels_decimal', 'trial_ids'};
        for k = 1:numel(required)
            if ~ismember(required{k}, names)
                error('verify_dataset:missingDataset', ...
                    'Missing %s/%s.', g, required{k});
            end
        end

        eeg_sz = h5info(h5_path, [g '/eeg_data']).Dataspace.Size;
        hbo_sz = h5info(h5_path, [g '/hbo_data']).Dataspace.Size;
        hbr_sz = h5info(h5_path, [g '/hbr_data']).Dataspace.Size;
        labels = double(h5read(h5_path, [g '/labels_decimal']));
        trial_ids = double(h5read(h5_path, [g '/trial_ids']));
        labels = labels(:);
        trial_ids = trial_ids(:);

        n_samples = numel(labels);
        if numel(trial_ids) ~= n_samples
            error('verify_dataset:lengthMismatch', ...
                '%s: labels (%d) and trial_ids (%d) differ.', ...
                g, n_samples, numel(trial_ids));
        end
        % h5info reports dimensions in MATLAB order: [samples, channels, time].
        if eeg_sz(1) ~= n_samples || hbo_sz(1) ~= n_samples || ...
                hbr_sz(1) ~= n_samples
            error('verify_dataset:sampleMismatch', ...
                '%s: data arrays do not have %d samples.', g, n_samples);
        end
        if any(~ismember(unique(labels), [1 2]))
            error('verify_dataset:badLabels', ...
                '%s: labels must be 1 or 2.', g);
        end

        n_trials = numel(unique(trial_ids));
        if mod(n_samples, n_trials) ~= 0 || n_samples / n_trials ~= n_win
            error('verify_dataset:badTrialIds', ...
                '%s: %d samples / %d trials is not %d windows.', ...
                g, n_samples, n_trials, n_win);
        end
        expected = repelem((1:n_trials)', n_win);
        if ~isequal(trial_ids, expected)
            error('verify_dataset:nonContiguousTrials', ...
                '%s: trial_ids are not consecutive blocks of %d.', ...
                g, n_win);
        end

        if check_values
            arrays = {'eeg_data', 'hbo_data', 'hbr_data'};
            for k = 1:numel(arrays)
                a = h5read(h5_path, [g '/' arrays{k}]);
                if any(~isfinite(a(:)))
                    error('verify_dataset:nonfinite', ...
                        '%s/%s contains %d non-finite values.', ...
                        g, arrays{k}, sum(~isfinite(a(:))));
                end
            end
        end

        n_total = n_total + n_samples;
        fprintf('  %s: %d trials, %d samples, eeg %s, hbo %s\n', ...
            strrep(g, ['/' task '/'], ''), n_trials, n_samples, ...
            mat2str(eeg_sz), mat2str(hbo_sz));
    end

    report.tasks.(task).n_subjects = numel(subject_groups);
    report.tasks.(task).n_samples = n_total;
    report.tasks.(task).n_windows = n_win;
end

fprintf('\nVerification passed.\n');

end
