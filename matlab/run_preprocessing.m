function run_preprocessing(dataset, varargin)
%RUN_PREPROCESSING  Build the SGT-Net HDF5 datasets from raw EEG-fNIRS data.
%
%   RUN_PREPROCESSING('public') preprocesses the Open Access EEG+NIRS
%   benchmark and writes datasets/hybridBCI_Tn8.h5.
%
%   RUN_PREPROCESSING('stroke') preprocesses the HEFMI-ICH patients and
%   writes datasets/stroke_patients_Tn8.h5.
%
%   RUN_PREPROCESSING('all') runs both branches.
%
%   Name-value options:
%       'PublicRoot'   directory containing EEG/ and NIRS/ of the benchmark
%                      (default: <project>/data/public)
%       'StrokeRoot'   directory containing Patients/ and Normals/
%                      (default: <project>/data/stroke)
%       'OutputDir'    output directory (default: <project>/datasets)
%       'Subjects'     subject IDs to process (default: all)
%       'Overwrite'    delete an existing output file (default: false)
%       'Verbose'      print progress (default: true)
%
%   Examples:
%       run_preprocessing('public', 'PublicRoot', 'D:/data/hybridBCI', ...
%           'Overwrite', true)
%       run_preprocessing('stroke', 'StrokeRoot', 'D:/data/HEFMI-ICH/epo_data', ...
%           'Subjects', 1:3, 'Overwrite', true)
%
%   The scientific parameters (bands, orders, epoch, baseline, windows) are
%   defined in SGTPNET.DEFAULT_CONFIG. Only the Signal Processing Toolbox is
%   required.
%
%   See also SGTPNET.DEFAULT_CONFIG, SGTPNET.BUILD_PUBLIC_DATASET,
%   SGTPNET.BUILD_STROKE_DATASET.

if nargin < 1 || isempty(dataset)
    dataset = 'all';
end
dataset = lower(char(dataset));
if ~ismember(dataset, {'public', 'stroke', 'all'})
    error('run_preprocessing:badDataset', ...
        'DATASET must be ''public'', ''stroke'' or ''all'' (got ''%s'').', dataset);
end

matlab_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(matlab_dir);
addpath(matlab_dir);

check_requirements();

cfg = sgtnet.default_config();
opts = parse_options(project_root, varargin{:});
opts = resolve_roots(opts);
cfg.verbose = opts.verbose;

fprintf('\n============================================================\n');
fprintf(' SGT-Net MATLAB preprocessing\n');
fprintf(' Dataset   : %s\n', dataset);
fprintf(' Output    : %s\n', opts.output_dir);
fprintf('============================================================\n');

t_start = tic;

if ismember(dataset, {'public', 'all'})
    if ~exist(opts.public_root, 'dir')
        error('run_preprocessing:missingPublicRoot', ...
            ['Benchmark data directory not found: %s\n' ...
             'Place the hybrid-BCI data there or pass ''PublicRoot''.'], ...
            opts.public_root);
    end
    fprintf('\n--- Benchmark dataset (Shin et al., 2017) ---\n');
    sgtnet.build_public_dataset(cfg, opts);
end

if ismember(dataset, {'stroke', 'all'})
    if ~exist(opts.stroke_root, 'dir')
        error('run_preprocessing:missingStrokeRoot', ...
            ['Clinical data directory not found: %s\n' ...
             'Point ''StrokeRoot'' to the epo_data directory that contains ' ...
             'Patients/ and Normals/.'], opts.stroke_root);
    end
    fprintf('\n--- Clinical dataset (Shi et al., 2025) ---\n');
    sgtnet.build_stroke_dataset(cfg, opts);
end

fprintf('\nAll done in %.1f s.\n', toc(t_start));

end

% =========================================================================
function opts = parse_options(project_root, varargin)
opts = struct();
opts.public_root = fullfile(project_root, 'data', 'public');
opts.stroke_root = fullfile(project_root, 'data', 'stroke');
opts.output_dir = fullfile(project_root, 'datasets');
opts.subjects = [];
opts.overwrite = false;
opts.verbose = true;

if mod(numel(varargin), 2) ~= 0
    error('run_preprocessing:badOptions', ...
        'Name-value options must come in pairs.');
end

for k = 1:2:numel(varargin)
    name = char(varargin{k});
    value = varargin{k+1};
    switch lower(name)
        case 'publicroot'
            opts.public_root = value;
        case 'strokeroot'
            opts.stroke_root = value;
        case 'outputdir'
            opts.output_dir = value;
        case 'subjects'
            opts.subjects = value;
        case 'overwrite'
            opts.overwrite = logical(value);
        case 'verbose'
            opts.verbose = logical(value);
        otherwise
            error('run_preprocessing:unknownOption', ...
                'Unknown option "%s".', name);
    end
end

if ~exist(opts.output_dir, 'dir')
    mkdir(opts.output_dir);
end
end

% =========================================================================
function opts = resolve_roots(opts)
%RESOLVE_ROOTS  Accept the repository root as well as the actual data root.
if ~exist(fullfile(opts.public_root, 'EEG'), 'dir') && ...
        exist(fullfile(opts.public_root, 'data', 'EEG'), 'dir')
    opts.public_root = fullfile(opts.public_root, 'data');
    fprintf('Note: PublicRoot resolved to %s\n', opts.public_root);
end

if ~exist(fullfile(opts.stroke_root, 'Patients'), 'dir')
    candidates = { ...
        fullfile(opts.stroke_root, 'epo_data'), ...
        fullfile(opts.stroke_root, '28955456', 'epo_data')};
    for k = 1:numel(candidates)
        if exist(fullfile(candidates{k}, 'Patients'), 'dir')
            opts.stroke_root = candidates{k};
            fprintf('Note: StrokeRoot resolved to %s\n', opts.stroke_root);
            break;
        end
    end
end
end

% =========================================================================
function check_requirements()
if ~license('test', 'Signal_Toolbox')
    error('run_preprocessing:missingToolbox', ...
        ['The Signal Processing Toolbox is required (butter, filtfilt, ' ...
         'zp2sos, resample).']);
end
if exist('h5create', 'file') ~= 2
    error('run_preprocessing:missingHDF5', ...
        'HDF5 support (h5create/h5write) is not available in this MATLAB.');
end
end
