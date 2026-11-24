function cfg = default_config()
%DEFAULT_CONFIG  Scientific and I/O parameters for SGT-Net preprocessing.
%
%   CFG = DEFAULT_CONFIG() returns a structure with the preprocessing
%   parameters used by the pipeline:
%
%     EEG    : 4-35 Hz, 6th-order zero-phase Butterworth, common average
%              reference.
%     fNIRS  : 0.01-0.1 Hz band-pass; raw optical intensities are converted
%              to concentration changes with the modified Beer-Lambert law
%              when the input dataset stores intensities.
%     Epoch  : [-5, 20] s relative to stimulus onset.
%     Baseline: [-3, 0] s.
%     Windows: EEG 2 s over [1, 10] s; fNIRS 8 s over [1, 16] s; 1 s step.
%              The 6 s difference between the two window lengths is the
%              temporal buffer that encloses the delayed haemodynamic peak.
%
%   See also RUN_PREPROCESSING, SGTPNET.BUILD_PUBLIC_DATASET,
%   SGTPNET.BUILD_STROKE_DATASET.

cfg = struct();

% -------------------------------------------------------------------------
% Target sampling rates (both datasets are resampled to these rates)
% -------------------------------------------------------------------------
cfg.fs.eeg = 200;      % Hz, as used by the benchmark and clinical data
cfg.fs.nirs = 10;      % Hz

% -------------------------------------------------------------------------
% EEG preprocessing
% -------------------------------------------------------------------------
cfg.eeg.band_hz = [4 35];          % sensorimotor and cognitive rhythms
cfg.eeg.filter_order = 6;          % zero-phase Butterworth (filtfilt)
cfg.eeg.eog_correction = false;    % optional EOG regression; EOG channels are always discarded
cfg.eeg.eog_pattern = 'EOG';       % substring matched against channel labels
cfg.eeg.reference = 'car';         % common average reference

% -------------------------------------------------------------------------
% fNIRS preprocessing
% -------------------------------------------------------------------------
cfg.nirs.band_hz = [0.01 0.1];     % remove drift, respiration and heartbeat
cfg.nirs.filter_order = 6;         % zero-phase Butterworth order

% HEFMI-ICH stores concentration changes in mol/L (values on the order of
% 1e-6). Scaling by 1000 converts them to mmol/L, the unit produced by the
% modified Beer-Lambert law for the benchmark dataset.
cfg.nirs.stroke_unit_scale = 1000;

% Modified Beer-Lambert law (benchmark dataset only; the clinical dataset
% already provides HbO/HbR concentration changes).
% Extinction coefficients are the Gratzer/Kollias values used by the BBCI
% proc_BeerLambert implementation, in cm^-1/(mol/L), rows ordered as
% [HbO HbR] and columns following cfg.nirs.mbll.wavelengths.
cfg.nirs.mbll.wavelengths = [760 850];           % nm, low then high
cfg.nirs.mbll.extinction = [1349.56 3566.24; ... % 760 nm: HbO, HbR
                            2436.62 1592.09];    % 850 nm: HbO, HbR
cfg.nirs.mbll.dpf = [5.98 7.15];                 % differential pathlength factor
cfg.nirs.mbll.opdist_cm = 3;                     % source-detector distance

% -------------------------------------------------------------------------
% Trial segmentation and baseline correction
% -------------------------------------------------------------------------
cfg.epoch_s = [-5 20];             % relative to stimulus onset
cfg.baseline_s = [-3 0];           % relative to stimulus onset

% -------------------------------------------------------------------------
% Asymmetric sliding-window segmentation
% -------------------------------------------------------------------------
cfg.window.eeg_size_s = 2;         % Te
cfg.window.eeg_range_s = [1 10];   % EEG execution window
cfg.window.eeg_step_s = 1;
cfg.window.nirs_size_s = 8;        % Tn
cfg.window.nirs_range_s = [1 16];  % fNIRS execution window
cfg.window.nirs_step_s = 1;
% With the values above both modalities yield 8 windows per trial and the
% fNIRS window extends 6 s beyond the EEG window (temporal buffer).

cfg.normalization = 'none';

% -------------------------------------------------------------------------
% HDF5 output options
% -------------------------------------------------------------------------
cfg.h5.deflate = 0;                % 0 = no compression (fastest writing)
cfg.h5.chunk_samples = 64;         % used only when deflate > 0

% -------------------------------------------------------------------------
% Logging
% -------------------------------------------------------------------------
cfg.verbose = true;

end
