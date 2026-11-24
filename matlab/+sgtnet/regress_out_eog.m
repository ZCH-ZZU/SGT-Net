function x = regress_out_eog(x, clab, eog_pattern, context)
%REGRESS_OUT_EOG  Remove ocular artifacts by regression on the EOG channels.
%
%   X = REGRESS_OUT_EOG(X, CLAB, EOG_PATTERN) regresses each EEG channel on
%   the EOG channels (with an intercept) and replaces it by the residual.
%   X is [time, channels]; CLAB labels the columns. Channels whose label
%   contains EOG_PATTERN are used as regressors and are left unchanged (the
%   caller discards them afterwards).
%
%   This is the optional EOG-correction branch of PREPROCESS_EEG. It is
%   disabled by default because the default protocol does not include an EOG
%   correction step; when disabled, the EOG channels are simply discarded.
%
%   See also SGTPNET.PREPROCESS_EEG.

if nargin < 4 || isempty(context)
    context = 'EEG';
end

validateattributes(x, {'double', 'single'}, {'2d', 'nonsparse'}, mfilename, 'x', 1);
clab = cellstr(clab);
if numel(clab) ~= size(x, 2)
    error('sgtnet:regress_out_eog:labelMismatch', ...
        '%s: %d labels for %d channels.', context, numel(clab), size(x, 2));
end

is_eog = contains(clab, eog_pattern);
if ~any(is_eog)
    warning('sgtnet:regress_out_eog:noEOG', ...
        '%s: no channel matches "%s"; EOG correction skipped.', ...
        context, eog_pattern);
    return;
end
if all(is_eog)
    error('sgtnet:regress_out_eog:onlyEOG', ...
        '%s: all channels match "%s".', context, eog_pattern);
end

n_time = size(x, 1);
design = [ones(n_time, 1), double(x(:, is_eog))];
eeg = double(x(:, ~is_eog));

% pinv handles rank-deficient EOG blocks (e.g. highly correlated VEOG/HEOG).
beta = pinv(design) * eeg;
x(:, ~is_eog) = eeg - design * beta;

end
