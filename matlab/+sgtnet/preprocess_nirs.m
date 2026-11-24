function [hbo, hbr, labels, clab_oxy, clab_deoxy, valid] = preprocess_nirs(cnt, mrk, cfg, context)
%PREPROCESS_NIRS  fNIRS preprocessing for one continuous recording.
%
%   [HBO, HBR, LABELS, CLAB_OXY, CLAB_DEOXY, VALID] =
%   PREPROCESS_NIRS(CNT, MRK, CFG) performs the fNIRS branch of Section IV
%   for a continuous recording:
%
%     1. resample to CFG.FS.NIRS when needed,
%     2. convert raw optical intensities to HbO/HbR with the modified
%        Beer-Lambert law when CNT.WAVELENGTHS is present,
%     3. 0.01-0.1 Hz 6th-order zero-phase band-pass,
%     4. segmentation into [-5, 20] s trials around each marker,
%     5. baseline correction with the [-3, 0] s interval.
%
%   HBO and HBR are [time, channels, trials] in mmol/L. CNT is a BBCI-style
%   NIRS structure (x, fs, clab, wavelengths); MRK provides marker times in
%   milliseconds and one-hot labels in y.
%
%   The clinical HEFMI-ICH release is already epoched and already contains
%   HbO/HbR; it is handled by SGTPNET.PROCESS_STROKE_SUBJECT instead of this
%   function.
%
%   See also SGTPNET.PREPROCESS_EEG, SGTPNET.BEER_LAMBERT.

if nargin < 4 || isempty(context)
    context = 'fNIRS';
end

required = {'x', 'fs', 'clab'};
for k = 1:numel(required)
    if ~isfield(cnt, required{k})
        error('sgtnet:preprocess_nirs:missingField', ...
            '%s: continuous structure is missing field "%s".', context, required{k});
    end
end

fs = double(cnt.fs);
clab = cellstr(cnt.clab);

if isfield(cnt, 'wavelengths') && ~isempty(cnt.wavelengths)
    [hb, clab_oxy, clab_deoxy] = sgtnet.beer_lambert(cnt, cfg, context);
else
    % Already-converted concentration data: first half HbO, second half HbR.
    hb = double(cnt.x);
    if mod(size(hb, 2), 2) ~= 0
        error('sgtnet:preprocess_nirs:badChannels', ...
            '%s: expected two equal channel blocks for HbO/HbR.', context);
    end
    n_chan = size(hb, 2) / 2;
    clab_oxy = clab(1:n_chan);
    clab_deoxy = clab(n_chan+1:end);
end

if fs ~= cfg.fs.nirs
    hb = sgtnet.resample_3d(hb, fs, cfg.fs.nirs, context);
    fs = cfg.fs.nirs;
end

hb = sgtnet.filter_bandpass(hb, fs, cfg.nirs.band_hz, cfg.nirs.filter_order, context);

if ~isfield(mrk, 'time')
    error('sgtnet:preprocess_nirs:missingMarkerTime', ...
        '%s: marker structure has no time field.', context);
end
marker_samples = round(double(mrk.time(:)) / 1000 * fs) + 1;

[epo, valid] = sgtnet.cut_epochs(hb, fs, marker_samples, cfg.epoch_s);

if isfield(mrk, 'y')
    [~, labels] = max(double(mrk.y), [], 1);
    labels = labels(:);
    if numel(labels) ~= numel(valid)
        error('sgtnet:preprocess_nirs:labelMismatch', ...
            '%s: %d markers but %d labels.', context, numel(valid), numel(labels));
    end
    labels = labels(valid);
else
    labels = [];
end

epo = sgtnet.baseline_correct(epo, fs, cfg.epoch_s, cfg.baseline_s, context);

n_chan = size(epo, 2) / 2;
hbo = epo(:, 1:n_chan, :);
hbr = epo(:, n_chan+1:end, :);

end
