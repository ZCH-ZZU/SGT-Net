function [epo, labels, clab, valid] = preprocess_eeg(cnt, mrk, cfg, context)
%PREPROCESS_EEG  EEG preprocessing for one continuous recording.
%
%   [EPO, LABELS, CLAB, VALID] = PREPROCESS_EEG(CNT, MRK, CFG) performs the
%   EEG branch of Section IV:
%
%     1. resample to CFG.FS.EEG when needed,
%     2. 4-35 Hz 6th-order zero-phase Butterworth band-pass,
%     3. optionally regress out ocular artifacts using the EOG channels
%        (disabled by default),
%     4. discard the EOG channels,
%     5. common average reference over the remaining EEG channels,
%     6. segmentation into [-5, 20] s trials around each marker,
%     7. baseline correction with the [-3, 0] s interval.
%
%   CNT is a BBCI-style continuous structure (x, fs, clab); MRK provides
%   marker times in milliseconds and one-hot labels in y. EPO is
%   [time, channels, trials], LABELS is [trials, 1] with 1-based class
%   indices, CLAB is a cell array of channel labels and VALID flags the
%   markers whose epoch was inside the recording.
%
%   See also SGTPNET.PREPROCESS_NIRS, SGTPNET.PROCESS_PUBLIC_SUBJECT.

if nargin < 4 || isempty(context)
    context = 'EEG';
end

required = {'x', 'fs', 'clab'};
for k = 1:numel(required)
    if ~isfield(cnt, required{k})
        error('sgtnet:preprocess_eeg:missingField', ...
            '%s: continuous structure is missing field "%s".', context, required{k});
    end
end

x = double(cnt.x);
fs = double(cnt.fs);
clab = cellstr(cnt.clab);

if fs ~= cfg.fs.eeg
    x = sgtnet.resample_3d(x, fs, cfg.fs.eeg, context);
    fs = cfg.fs.eeg;
end

x = sgtnet.filter_bandpass(x, fs, cfg.eeg.band_hz, cfg.eeg.filter_order, context);

if cfg.eeg.eog_correction
    x = sgtnet.regress_out_eog(x, clab, cfg.eeg.eog_pattern, context);
end

% EOG channels are never part of the model input: they are either used for
% the optional regression above and then discarded, or discarded directly.
keep = ~contains(clab, cfg.eeg.eog_pattern);
if ~any(keep)
    error('sgtnet:preprocess_eeg:noChannels', ...
        '%s: discarding "%s" removed every channel.', ...
        context, cfg.eeg.eog_pattern);
end
x = x(:, keep);
clab = clab(keep);

if strcmpi(cfg.eeg.reference, 'car')
    x = x - mean(x, 2);
else
    error('sgtnet:preprocess_eeg:badReference', ...
        '%s: unsupported reference "%s".', context, cfg.eeg.reference);
end

if ~isfield(mrk, 'time')
    error('sgtnet:preprocess_eeg:missingMarkerTime', ...
        '%s: marker structure has no time field.', context);
end
marker_samples = round(double(mrk.time(:)) / 1000 * fs) + 1;

[epo, valid] = sgtnet.cut_epochs(x, fs, marker_samples, cfg.epoch_s);

if isfield(mrk, 'y')
    [~, labels] = max(double(mrk.y), [], 1);
    labels = labels(:);
    if numel(labels) ~= numel(valid)
        error('sgtnet:preprocess_eeg:labelMismatch', ...
            '%s: %d markers but %d labels.', context, numel(valid), numel(labels));
    end
    labels = labels(valid);
else
    labels = [];
end

epo = sgtnet.baseline_correct(epo, fs, cfg.epoch_s, cfg.baseline_s, context);

end
