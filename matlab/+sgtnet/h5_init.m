function h5_init(h5_path, cfg, info)
%H5_INIT  Create the SGT-Net HDF5 file and its metadata scaffold.
%
%   H5_INIT(H5_PATH, CFG, INFO) creates a new HDF5 file with the global
%   metadata, the task groups and the channel/montage information shared by
%   all subjects. Subject arrays are appended by H5_WRITE_SUBJECT.
%
%   INFO fields:
%       dataset_name      human-readable dataset name
%       dataset_key       short key ('public' or 'stroke')
%       dataset_citation  citation string
%       dataset_url       download URL
%       group             optional group name (clinical data)
%       tasks             cell array of task names ('MI', 'MA')
%       class_names       cell array of cell arrays, one per task
%       eeg_clab          EEG channel labels
%       nirs_clab         fNIRS channel labels
%       eeg_mnt_original  EEG montage or []
%       nirs_mnt_original fNIRS montage or []
%       eeg_units         e.g. 'uV'
%       nirs_units        e.g. 'mmol/L'
%       n_subjects        number of requested subjects
%       n_windows         windows per trial (EEG and fNIRS must agree)
%
%   The layout follows the conventions expected by the Python loaders:
%   data arrays are stored as [time, channels, samples] and labels/trial IDs
%   as [1, samples].
%
%   See also SGTPNET.H5_WRITE_SUBJECT, SGTPNET.BUILD_PUBLIC_DATASET.

if exist(h5_path, 'file')
    delete(h5_path);
end

% ---- root metadata -----------------------------------------------------
write_numeric(h5_path, '/eeg_fs', cfg.fs.eeg, cfg);
write_numeric(h5_path, '/nirs_fs', cfg.fs.nirs, cfg);
write_strings(h5_path, '/eeg_units', string(info.eeg_units));
write_strings(h5_path, '/nirs_units', string(info.nirs_units));
write_strings(h5_path, '/eeg_channels', string(info.eeg_clab(:)'));
write_strings(h5_path, '/nirs_channels', string(info.nirs_clab(:)'));

write_attrs(h5_path, '/', struct( ...
    'dataset_name', info.dataset_name, ...
    'dataset_key', info.dataset_key, ...
    'dataset_citation', info.dataset_citation, ...
    'dataset_url', info.dataset_url, ...
    'created_utc', char(datetime('now', 'TimeZone', 'UTC', ...
        'Format', 'yyyy-MM-dd HH:mm:ss')), ...
    'matlab_version', version, ...
    'n_subjects', info.n_subjects));

if ~isempty(info.eeg_mnt_original)
    write_montage(h5_path, '/eeg_mnt_original', info.eeg_mnt_original, cfg);
end
if ~isempty(info.nirs_mnt_original)
    write_montage(h5_path, '/nirs_mnt_original', info.nirs_mnt_original, cfg);
end

% ---- task scaffold -----------------------------------------------------
for t = 1:numel(info.tasks)
    task = info.tasks{t};
    group = ['/' task];

    write_strings(h5_path, [group '/class_names'], string(info.class_names{t}(:)'));
    write_strings(h5_path, [group '/eeg_channels'], string(info.eeg_clab(:)'));
    write_strings(h5_path, [group '/nirs_channels'], string(info.nirs_clab(:)'));

    write_attrs(h5_path, group, struct( ...
        'task', task, ...
        'dataset_key', info.dataset_key, ...
        'dataset_name', info.dataset_name, ...
        'eeg_fs', cfg.fs.eeg, ...
        'nirs_fs', cfg.fs.nirs, ...
        'eeg_band_hz', cfg.eeg.band_hz, ...
        'eeg_filter_order', cfg.eeg.filter_order, ...
        'eeg_reference', cfg.eeg.reference, ...
        'nirs_band_hz', cfg.nirs.band_hz, ...
        'nirs_filter_order', cfg.nirs.filter_order, ...
        'epoch_s', cfg.epoch_s, ...
        'baseline_s', cfg.baseline_s, ...
        'window_eeg_size_s', cfg.window.eeg_size_s, ...
        'window_eeg_step_s', cfg.window.eeg_step_s, ...
        'window_eeg_range_s', cfg.window.eeg_range_s, ...
        'window_nirs_size_s', cfg.window.nirs_size_s, ...
        'window_nirs_step_s', cfg.window.nirs_step_s, ...
        'window_nirs_range_s', cfg.window.nirs_range_s, ...
        'n_windows_per_trial', info.n_windows, ...
        'eeg_units', info.eeg_units, ...
        'nirs_units', info.nirs_units, ...
        'n_subjects', info.n_subjects));
end

if isfield(info, 'group') && ~isempty(info.group)
    write_attrs(h5_path, '/', struct('subject_group', info.group));
end

end

% =========================================================================
function write_numeric(h5_path, name, data, cfg)
if isempty(data)
    return;
end
sz = size(data);
if cfg.h5.deflate > 0 && numel(sz) >= 2
    chunk = sz;
    chunk(1) = min(sz(1), cfg.h5.chunk_samples);
    h5create(h5_path, name, sz, 'Datatype', class(data), ...
        'ChunkSize', chunk, 'Deflate', cfg.h5.deflate);
else
    h5create(h5_path, name, sz, 'Datatype', class(data));
end
h5write(h5_path, name, data);
end

% =========================================================================
function write_strings(h5_path, name, data)
if isempty(data)
    return;
end
h5create(h5_path, name, size(data), 'Datatype', 'string');
h5write(h5_path, name, data);
end

% =========================================================================
function write_montage(h5_path, group, mnt, cfg)
write_strings(h5_path, [group '/clab'], string(mnt.clab(:)'));
if isfield(mnt, 'x')
    write_numeric(h5_path, [group '/x'], mnt.x, cfg);
end
if isfield(mnt, 'y')
    write_numeric(h5_path, [group '/y'], mnt.y, cfg);
end
if isfield(mnt, 'pos_3d')
    write_numeric(h5_path, [group '/pos_3d'], mnt.pos_3d, cfg);
end
end

% =========================================================================
function write_attrs(h5_path, group, attrs)
names = fieldnames(attrs);
for k = 1:numel(names)
    value = attrs.(names{k});
    if isempty(value)
        continue;
    end
    if iscell(value)
        value = strjoin(cellstr(value), '; ');
    end
    if isstring(value)
        value = char(value);
    end
    if islogical(value)
        value = double(value);
    end
    h5writeatt(h5_path, group, names{k}, value);
end
end
