function [windows, starts_s] = make_windows(epo, fs, epoch_s, range_s, size_s, step_s, context)
%MAKE_WINDOWS  Extract sliding windows from epoched data.
%
%   WINDOWS = MAKE_WINDOWS(EPO, FS, EPOCH_S, RANGE_S, SIZE_S, STEP_S)
%   extracts windows of SIZE_S seconds from EPO ([time, channels, trials]).
%   RANGE_S is the allowed start range in seconds relative to stimulus
%   onset; STEP_S is the sliding step.
%
%   The output has layout [n_trials * n_windows, channels, samples] in
%   trial-major order, i.e. all windows of trial 1, then all windows of
%   trial 2, and so on. STARTS_S lists the window onset times.
%
%   For the SGT-Net defaults (EEG: size 2 s over [1, 10] s; fNIRS: size 8 s
%   over [1, 16] s; step 1 s) both modalities produce 8 windows per trial.
%   The fNIRS window extends 6 s beyond the EEG window and therefore covers
%   the delayed haemodynamic response.
%
%   See also SGTPNET.MAKE_PAYLOAD, SGTPNET.ZSCORE_WINDOWS.

if nargin < 7 || isempty(context)
    context = 'epoch';
end

validateattributes(epo, {'double', 'single'}, {'nonsparse'}, mfilename, 'epo', 1);
if ndims(epo) ~= 3
    error('sgtnet:make_windows:badInput', ...
        '%s: EPO must be [time, channels, trials].', context);
end
validateattributes(fs, {'double'}, {'scalar', 'positive'}, mfilename, 'fs', 2);
validateattributes(epoch_s, {'double'}, {'numel', 2, 'increasing'}, ...
    mfilename, 'epoch_s', 3);
validateattributes(range_s, {'double'}, {'numel', 2, 'increasing'}, ...
    mfilename, 'range_s', 4);
validateattributes(size_s, {'double'}, {'scalar', 'positive'}, mfilename, 'size_s', 5);
validateattributes(step_s, {'double'}, {'scalar', 'positive'}, mfilename, 'step_s', 6);

if range_s(1) < epoch_s(1) || range_s(2) > epoch_s(2)
    error('sgtnet:make_windows:outsideEpoch', ...
        '%s: window range [%g, %g] s is outside the epoch [%g, %g] s.', ...
        context, range_s, epoch_s);
end

starts_s = range_s(1):step_s:(range_s(2) - size_s);
if isempty(starts_s)
    error('sgtnet:make_windows:noWindows', ...
        '%s: no windows fit in [%g, %g] s with size %g s and step %g s.', ...
        context, range_s, size_s, step_s);
end

n_win = numel(starts_s);
n_samples = round(size_s * fs);
[~, n_chan, n_trial] = size(epo);

windows = zeros(n_trial * n_win, n_chan, n_samples);
n = 0;
for i_trial = 1:n_trial
    for i_win = 1:n_win
        s0 = round((starts_s(i_win) - epoch_s(1)) * fs) + 1;
        s1 = s0 + n_samples - 1;
        if s0 < 1 || s1 > size(epo, 1)
            error('sgtnet:make_windows:badIndex', ...
                '%s: window %d of trial %d exceeds the epoch.', ...
                context, i_win, i_trial);
        end
        n = n + 1;
        win = epo(s0:s1, :, i_trial)';        % [channels, samples]
        windows(n, :, :) = reshape(win, [1, n_chan, n_samples]);
    end
end

end
