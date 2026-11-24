function y = resample_3d(x, fs_in, fs_out, context)
%RESAMPLE_3D  Resample a 2-D or 3-D signal along its first dimension.
%
%   Y = RESAMPLE_3D(X, FS_IN, FS_OUT) resamples X from FS_IN to FS_OUT.
%   X may be [time, channels] or [time, channels, trials]. Channels and
%   trials are resampled independently with the anti-aliasing FIR filter of
%   RESAMPLE; the rational factor is reduced by its greatest common divisor
%   before the call.
%
%   The clinical HEFMI-ICH release provides 256 Hz EEG and 11 Hz fNIRS; both
%   are converted to the 200 Hz / 10 Hz rates used by SGT-Net.
%
%   See also SGTPNET.PROCESS_STROKE_SUBJECT.

if nargin < 4 || isempty(context)
    context = 'signal';
end

validateattributes(x, {'double', 'single'}, {'nonsparse'}, mfilename, 'x', 1);
validateattributes(fs_in, {'double'}, {'scalar', 'positive'}, mfilename, 'fs_in', 2);
validateattributes(fs_out, {'double'}, {'scalar', 'positive'}, mfilename, 'fs_out', 3);

if ndims(x) > 3
    error('sgtnet:resample_3d:badInput', ...
        '%s: X must be [time, channels] or [time, channels, trials].', context);
end

if fs_in == fs_out
    y = x;
    return;
end

% resample expects integer up/down factors. Both sampling rates are
% integers here, so the ratio is exact.
p = round(fs_out);
q = round(fs_in);
if abs(p - fs_out) > 1e-9 || abs(q - fs_in) > 1e-9
    error('sgtnet:resample_3d:nonIntegerRate', ...
        '%s: sampling rates must be integers (got %g and %g).', ...
        context, fs_in, fs_out);
end
g = gcd(p, q);
p = p / g;
q = q / g;

was3d = ndims(x) == 3;
if was3d
    [n_time, n_chan, n_trial] = size(x);
    x = reshape(x, n_time, n_chan * n_trial);
end

y = resample(double(x), p, q);

if was3d
    y = reshape(y, size(y, 1), n_chan, n_trial);
end

end
