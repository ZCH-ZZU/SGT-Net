function [hb, clab_oxy, clab_deoxy] = beer_lambert(cnt, cfg, context)
%BEER_LAMBERT  Modified Beer-Lambert conversion of raw optical intensities.
%
%   [HB, CLAB_OXY, CLAB_DEOXY] = BEER_LAMBERT(CNT, CFG) converts the raw
%   two-wavelength NIRS intensities stored in CNT.X ([time, channels]) into
%   concentration changes of oxygenated and deoxygenated haemoglobin.
%
%   HB has the layout [time, 2 * n_channels]: the first block is HbO and the
%   second block is HbR, in mmol/L. CLAB_OXY and CLAB_DEOXY contain the
%   corresponding channel labels.
%
%   The implementation follows the modified Beer-Lambert law with the
%   extinction coefficients of Gratzer/Kollias (the same table used by the
%   BBCI proc_BeerLambert function referenced by the benchmark dataset):
%
%       A = -log10(I / mean(I))
%       [HbO; HbR] = inv(E * DPF * d) * A
%
%   where the mean is taken over the complete recording, DPF is the
%   differential pathlength factor and d is the source-detector distance.
%
%   CNT must provide the fields X, FS, CLAB and WAVELENGTHS. The function is
%   only used for datasets distributed as optical intensities; HEFMI-ICH
%   already releases HbO/HbR concentration changes.
%
%   See also SGTPNET.PREPROCESS_NIRS.

if nargin < 3 || isempty(context)
    context = 'NIRS';
end

required = {'x', 'fs', 'clab', 'wavelengths'};
for k = 1:numel(required)
    if ~isfield(cnt, required{k})
        error('sgtnet:beer_lambert:missingField', ...
            '%s: NIRS structure is missing field "%s".', context, required{k});
    end
end

x = double(cnt.x);
n_time = size(x, 1);
n_total = size(x, 2);
if mod(n_total, 2) ~= 0
    error('sgtnet:beer_lambert:badChannels', ...
        '%s: expected an even number of wavelength channels, got %d.', ...
        context, n_total);
end
n_chan = n_total / 2;

wl = double(cnt.wavelengths(:)');
if numel(wl) ~= 2
    error('sgtnet:beer_lambert:badWavelengths', ...
        '%s: expected two wavelengths, got %d.', context, numel(wl));
end
if ~isequal(sort(wl), sort(cfg.nirs.mbll.wavelengths))
    error('sgtnet:beer_lambert:unsupportedWavelengths', ...
        ['%s: wavelengths [%s] nm are not tabulated. Add their extinction ' ...
         'coefficients to CFG.NIRS.MBLL.EXTINCTION before processing.'], ...
        context, num2str(wl));
end

% Reorder the extinction table so that its rows follow the order of WL.
[~, order] = ismember(wl, cfg.nirs.mbll.wavelengths);
ext = cfg.nirs.mbll.extinction(order, :);   % [HbO HbR] per wavelength

% The channel blocks follow the dataset convention: low wavelength first,
% high wavelength second.
if wl(1) > wl(2)
    error('sgtnet:beer_lambert:channelOrder', ...
        '%s: expected the low wavelength in the first channel block.', context);
end

low = x(:, 1:n_chan) + eps;
high = x(:, n_chan+1:end) + eps;

% Optical density relative to the recording mean (Ival = all).
att_high = real(-log10(high ./ repmat(mean(high, 1), [n_time, 1])));
att_low = real(-log10(low ./ repmat(mean(low, 1), [n_time, 1])));

% Extinction coefficients are in cm^-1/(mol/L); /1000 converts to
% mmol^-1 cm^-1, matching the mmol/L output unit. The factor 10 mirrors the
% BBCI implementation and keeps the numerical scale of the original
% benchmark pipeline.
epsilon = ext / 1000;
[~, idx] = max(wl);
if idx == 2
    epsilon = flipud(epsilon);
end
e = epsilon / 10;
e2 = (e .* repmat(cfg.nirs.mbll.dpf(:), 1, 2)) * cfg.nirs.mbll.opdist_cm;

a_mat = [reshape(att_high, [], 1), reshape(att_low, [], 1)];
conc = (e2 \ a_mat')';   % [time * channels, 2], columns [HbO HbR]

hb = [reshape(conc(:, 1), n_time, n_chan), reshape(conc(:, 2), n_time, n_chan)];

% Channel labels: strip the wavelength suffix and append the chromophore.
clab = cnt.clab;
clab_oxy = strrep(clab(1:n_chan), 'lowWL', 'oxy');
clab_oxy = strrep(clab_oxy, 'highWL', 'oxy');
clab_deoxy = strrep(clab(n_chan+1:end), 'lowWL', 'deoxy');
clab_deoxy = strrep(clab_deoxy, 'highWL', 'deoxy');

if any(~isfinite(hb(:)))
    warning('sgtnet:beer_lambert:nonfinite', ...
        '%s: Beer-Lambert output contains %d non-finite samples.', ...
        context, sum(~isfinite(hb(:))));
end

end
