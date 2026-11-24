function h5_write_subject(h5_path, task, subject_id, payload, cfg)
%H5_WRITE_SUBJECT  Append one subject/task payload to an SGT-Net HDF5 file.
%
%   H5_WRITE_SUBJECT(H5_PATH, TASK, SUBJECT_ID, PAYLOAD, CFG) writes:
%
%       /TASK/subject_XX/eeg_data       [time, channels, samples]
%       /TASK/subject_XX/hbo_data       [time, channels, samples]
%       /TASK/subject_XX/hbr_data       [time, channels, samples]
%       /TASK/subject_XX/labels_decimal [samples, 1]  (1-based)
%       /TASK/subject_XX/trial_ids      [samples, 1]  (1-based)
%
%   The array orientation matches the Python loaders, which transpose the
%   first and last dimensions after reading.
%
%   See also SGTPNET.H5_INIT, SGTPNET.MAKE_PAYLOAD.

group = sprintf('/%s/subject_%02d', task, subject_id);

% MATLAB's HDF5 layer reverses dimension order: a MATLAB array [samples,
% channels, time] is stored as [time, channels, samples], which is exactly
% the layout expected by the Python loaders. Do not permute here.
write_array(h5_path, [group '/eeg_data'], payload.eeg_data, cfg);
write_array(h5_path, [group '/hbo_data'], payload.hbo_data, cfg);
write_array(h5_path, [group '/hbr_data'], payload.hbr_data, cfg);

labels = double(payload.labels(:));
trial_ids = double(payload.trial_ids(:));
write_array(h5_path, [group '/labels_decimal'], labels, cfg);
write_array(h5_path, [group '/trial_ids'], trial_ids, cfg);

n_trials = numel(unique(trial_ids));
h5writeatt(h5_path, group, 'subject_id', subject_id);
h5writeatt(h5_path, group, 'n_trials', n_trials);
h5writeatt(h5_path, group, 'n_samples', numel(labels));
h5writeatt(h5_path, group, 'n_windows_per_trial', payload.n_windows);

end

% =========================================================================
function write_array(h5_path, name, data, cfg)
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
