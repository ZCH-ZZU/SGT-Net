function [epo, valid] = cut_epochs(x, fs, marker_samples, epoch_s)
%CUT_EPOCHS  Segment a continuous signal into fixed-length trials.
%
%   [EPO, VALID] = CUT_EPOCHS(X, FS, MARKER_SAMPLES, EPOCH_S) cuts trials
%   from X ([time, channels]) around the 1-based sample indices in
%   MARKER_SAMPLES. EPOCH_S is [start end] in seconds relative to each
%   marker; the returned EPO is [n_samples, channels, n_valid_trials].
%
%   Trials whose epoch would fall outside X are skipped and flagged as
%   false in VALID. The caller is responsible for applying the same mask to
%   the trial labels.
%
%   The number of samples is round((EPOCH_S(2) - EPOCH_S(1)) * FS) + 1, so
%   both endpoints are included (e.g. [-5, 20] s at 200 Hz gives 5001
%   samples).
%
%   See also SGTPNET.BASELINE_CORRECT, SGTPNET.PROCESS_PUBLIC_SUBJECT.

validateattributes(x, {'double', 'single'}, {'2d', 'nonsparse'}, mfilename, 'x', 1);
validateattributes(fs, {'double'}, {'scalar', 'positive'}, mfilename, 'fs', 2);
validateattributes(marker_samples, {'double'}, {'vector', 'finite'}, ...
    mfilename, 'marker_samples', 3);
validateattributes(epoch_s, {'double'}, {'numel', 2, 'increasing'}, ...
    mfilename, 'epoch_s', 4);

n_time = size(x, 1);
n_chan = size(x, 2);
n_epoch = round(diff(epoch_s) * fs) + 1;
n_marker = numel(marker_samples);

valid = false(n_marker, 1);
starts = zeros(n_marker, 1);
for i = 1:n_marker
    s0 = round(marker_samples(i) + epoch_s(1) * fs);
    s1 = s0 + n_epoch - 1;
    if s0 >= 1 && s1 <= n_time
        valid(i) = true;
        starts(i) = s0;
    else
        warning('sgtnet:cut_epochs:outOfRange', ...
            'Trial %d skipped: samples [%d, %d] outside [1, %d].', ...
            i, s0, s1, n_time);
    end
end

n_valid = sum(valid);
epo = zeros(n_epoch, n_chan, n_valid);
idx = find(valid);
for j = 1:n_valid
    s0 = starts(idx(j));
    epo(:, :, j) = x(s0:s0 + n_epoch - 1, :);
end

end
