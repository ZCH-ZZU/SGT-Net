function raw = load_stroke_raw(data_root, subject_id, group)
%LOAD_STROKE_RAW  Load the epoched EEG and fNIRS sessions of one HEFMI-ICH subject.
%
%   RAW = LOAD_STROKE_RAW(DATA_ROOT, SUBJECT_ID, GROUP) loads the session
%   files of the clinical HEFMI-ICH dataset (Shi et al., 2025).
%
%   Expected layout:
%       DATA_ROOT/Patients/EEG/subjectN/<k>_epo.mat
%       DATA_ROOT/Patients/NIRS/subjectN/<k>_epo.mat
%
%   GROUP is normally 'Patients'; 'Normals' is supported so that the same
%   loader can be reused for the healthy control group, although the default
%   pipeline only uses the patients.
%
%   Each session file stores EPO.X ([time, channels, trials]), EPO.Y, EPO.FS
%   and EPO.CHANNELSNAME. The NIRS file stores EPO.OXY.X, EPO.DOXY.X and
%   EPO.OXY.FS. The epoch starts 10 s before the cue and ends 15 s after the
%   rest marker, giving ~37 s in total.
%
%   See also SGTPNET.PROCESS_STROKE_SUBJECT.

if nargin < 3 || isempty(group)
    group = 'Patients';
end

subject_str = sprintf('subject%d', subject_id);
eeg_dir = fullfile(data_root, group, 'EEG', subject_str);
nirs_dir = fullfile(data_root, group, 'NIRS', subject_str);

if ~exist(eeg_dir, 'dir')
    error('sgtnet:load_stroke_raw:missingEEG', ...
        'EEG directory not found: %s', eeg_dir);
end
if ~exist(nirs_dir, 'dir')
    error('sgtnet:load_stroke_raw:missingNIRS', ...
        'fNIRS directory not found: %s', nirs_dir);
end

files = dir(fullfile(eeg_dir, '*_epo.mat'));
if isempty(files)
    error('sgtnet:load_stroke_raw:noSessions', ...
        'No *_epo.mat files found in %s.', eeg_dir);
end

session_nums = zeros(numel(files), 1);
for i = 1:numel(files)
    session_nums(i) = sscanf(files(i).name, '%d_epo.mat');
end
[~, order] = sort(session_nums);
files = files(order);
session_nums = session_nums(order);

raw = struct();
raw.subject_id = subject_id;
raw.group = group;
raw.session_num = session_nums(:)';
raw.eeg = {};
raw.nirs = {};

for i = 1:numel(files)
    eeg_file = fullfile(eeg_dir, files(i).name);
    nirs_file = fullfile(nirs_dir, files(i).name);
    if ~exist(nirs_file, 'file')
        warning('sgtnet:load_stroke_raw:missingNIRSSession', ...
            'Subject %d: no matching fNIRS file for %s; session skipped.', ...
            subject_id, files(i).name);
        continue;
    end

    S = load(eeg_file, 'epo');
    N = load(nirs_file, 'epo');
    epo_eeg = S.epo;
    epo_nirs = N.epo;

    if ~isfield(epo_eeg, 'X') || ~isfield(epo_eeg, 'y') || ~isfield(epo_eeg, 'fs')
        error('sgtnet:load_stroke_raw:badEEGFile', ...
            'EEG file %s does not contain epo.X, epo.y and epo.fs.', eeg_file);
    end
    if ~isfield(epo_nirs, 'oxy') || ~isfield(epo_nirs, 'doxy')
        error('sgtnet:load_stroke_raw:badNIRSFile', ...
            'fNIRS file %s does not contain epo.oxy and epo.doxy.', nirs_file);
    end

    raw.eeg{end+1} = epo_eeg;
    raw.nirs{end+1} = epo_nirs;
end

if isempty(raw.eeg)
    error('sgtnet:load_stroke_raw:noValidSessions', ...
        'Subject %d: no session has both EEG and fNIRS data.', subject_id);
end

raw.n_sessions = numel(raw.eeg);

end
