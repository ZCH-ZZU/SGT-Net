function w = zscore_windows(w, eps_std)
%ZSCORE_WINDOWS  Window-wise z-score normalization.
%
%   W = ZSCORE_WINDOWS(W) normalizes every channel of every window
%   independently: the temporal mean is removed and the temporal standard
%   deviation (N-1 normalization, as in MATLAB's STD/NORMALIZE) is divided
%   out. W is [samples, channels, time].
%
%   W = ZSCORE_WINDOWS(W, EPS_STD) uses EPS_STD as the lower bound of the
%   standard deviation to avoid division by zero for flat channels. The
%   default is 1e-8.
%
%   This implements the "window-wise z-score normalization applied prior to
%   model input" used by the pipeline.
%
%   See also SGTPNET.MAKE_PAYLOAD.

if nargin < 2 || isempty(eps_std)
    eps_std = 1e-8;
end

validateattributes(w, {'double', 'single'}, {'nonsparse'}, mfilename, 'w', 1);
if ndims(w) ~= 3
    error('sgtnet:zscore_windows:badInput', ...
        'W must be [samples, channels, time].');
end

mu = mean(w, 3);
sd = std(w, 0, 3);
w = (w - mu) ./ max(sd, eps_std);

end
