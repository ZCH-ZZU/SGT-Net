function y = filter_bandpass(x, fs, band_hz, order, context)
%FILTER_BANDPASS  Zero-phase Butterworth band-pass filtering.
%
%   Y = FILTER_BANDPASS(X, FS, BAND_HZ, ORDER) filters X along its first
%   dimension with a zero-phase Butterworth band-pass. X may be a matrix
%   [time, channels] or a 3-D array [time, channels, trials]; channels and
%   trials are filtered independently.
%
%   Y = FILTER_BANDPASS(..., CONTEXT) appends CONTEXT to warning and error
%   messages (for example, a subject/task identifier).
%
%   The implementation uses second-order sections (zp2sos) because
%   high-order band-pass designs are poorly conditioned in transfer-function
%   form, especially for the low fNIRS cut-offs.
%
%   See also SGTPNET.PREPROCESS_EEG, SGTPNET.PROCESS_STROKE_SUBJECT.

if nargin < 5 || isempty(context)
    context = 'signal';
end

validateattributes(x, {'double', 'single'}, {'nonsparse'}, mfilename, 'x', 1);
if ndims(x) > 3
    error('sgtnet:filter_bandpass:badInput', ...
        '%s: X must be [time, channels] or [time, channels, trials].', context);
end
validateattributes(fs, {'double'}, {'scalar', 'positive'}, mfilename, 'fs', 2);
validateattributes(band_hz, {'double'}, {'numel', 2, 'increasing', 'positive'}, ...
    mfilename, 'band_hz', 3);
validateattributes(order, {'double'}, {'scalar', 'integer', 'positive'}, ...
    mfilename, 'order', 4);

if band_hz(2) >= fs / 2
    error('sgtnet:filter_bandpass:badBand', ...
        '%s: upper cut-off %.3f Hz must be below Nyquist (%.3f Hz).', ...
        context, band_hz(2), fs / 2);
end

was3d = ndims(x) == 3;
if was3d
    [n_time, n_chan, n_trial] = size(x);
    x = reshape(x, n_time, n_chan * n_trial);
end

[z, p, k] = butter(order, band_hz / (fs / 2), 'bandpass');
[sos, g] = zp2sos(z, p, k);

if any(~isfinite(sos(:))) || any(~isfinite(g))
    error('sgtnet:filter_bandpass:unstable', ...
        '%s: Butterworth design produced non-finite coefficients.', context);
end

% Fold the overall gain into the first section. Keeping G separate leaves
% the state-space initial-condition problem badly scaled for low cut-offs
% (G can be ~1e-10), which makes filtfilt emit near-singular warnings.
sos(1, 1:3) = sos(1, 1:3) * g;
g = 1;

y = filtfilt(sos, g, double(x));

if was3d
    y = reshape(y, n_time, n_chan, n_trial);
end

if any(~isfinite(y(:)))
    warning('sgtnet:filter_bandpass:nonfinite', ...
        '%s: filtered signal contains %d non-finite samples.', ...
        context, sum(~isfinite(y(:))));
end

end
