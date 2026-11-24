function epo = baseline_correct(epo, fs, epoch_s, baseline_s, context)
%BASELINE_CORRECT  Subtract the pre-stimulus baseline mean from each trial.
%
%   EPO = BASELINE_CORRECT(EPO, FS, EPOCH_S, BASELINE_S) subtracts, for
%   every channel and trial, the mean of the samples inside BASELINE_S. EPO
%   is [time, channels, trials]; EPOCH_S and BASELINE_S are [start end] in
%   seconds relative to stimulus onset.
%
%   The default interval is [-3, 0] s.
%
%   See also SGTPNET.CUT_EPOCHS.

if nargin < 5 || isempty(context)
    context = 'epoch';
end

validateattributes(epo, {'double', 'single'}, {'nonsparse'}, mfilename, 'epo', 1);
if ndims(epo) ~= 3
    error('sgtnet:baseline_correct:badInput', ...
        '%s: EPO must be [time, channels, trials].', context);
end
validateattributes(fs, {'double'}, {'scalar', 'positive'}, mfilename, 'fs', 2);
validateattributes(epoch_s, {'double'}, {'numel', 2, 'increasing'}, ...
    mfilename, 'epoch_s', 3);
validateattributes(baseline_s, {'double'}, {'numel', 2, 'increasing'}, ...
    mfilename, 'baseline_s', 4);

if baseline_s(1) < epoch_s(1) || baseline_s(2) > epoch_s(2)
    error('sgtnet:baseline_correct:outsideEpoch', ...
        '%s: baseline [%g, %g] s is outside the epoch [%g, %g] s.', ...
        context, baseline_s, epoch_s);
end

b0 = round((baseline_s(1) - epoch_s(1)) * fs) + 1;
b1 = round((baseline_s(2) - epoch_s(1)) * fs) + 1;
if b1 > size(epo, 1)
    error('sgtnet:baseline_correct:badIndex', ...
        '%s: baseline indices [%d, %d] exceed the epoch length %d.', ...
        context, b0, b1, size(epo, 1));
end

baseline = mean(epo(b0:b1, :, :), 1);
epo = epo - baseline;

end
