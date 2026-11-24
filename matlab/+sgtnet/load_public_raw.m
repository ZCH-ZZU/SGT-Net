function raw = load_public_raw(data_root, subject_id)
%LOAD_PUBLIC_RAW  Load the continuous EEG and fNIRS recordings of one subject.
%
%   RAW = LOAD_PUBLIC_RAW(DATA_ROOT, SUBJECT_ID) loads the six sessions of
%   the Open Access EEG+NIRS dataset (Shin et al., 2017).
%
%   Expected layout:
%       DATA_ROOT/EEG/subject XX/with occular artifact/{cnt,mrk,mnt}.mat
%       DATA_ROOT/NIRS/subject XX/{cnt,mrk,mnt}.mat
%
%   The returned structure contains cell arrays EEG_CNT, EEG_MRK, NIRS_CNT,
%   NIRS_MRK (one entry per session) and the montages EEG_MNT, NIRS_MNT.
%   The files are read with LOAD(..., variable) so that only the required
%   variables are pulled into memory.
%
%   See also SGTPNET.PROCESS_PUBLIC_SUBJECT.

subject_str = sprintf('subject %02d', subject_id);
eeg_dir = fullfile(data_root, 'EEG', subject_str, 'with occular artifact');
nirs_dir = fullfile(data_root, 'NIRS', subject_str);

if ~exist(eeg_dir, 'dir')
    error('sgtnet:load_public_raw:missingEEG', ...
        'EEG directory not found: %s', eeg_dir);
end
if ~exist(nirs_dir, 'dir')
    error('sgtnet:load_public_raw:missingNIRS', ...
        'fNIRS directory not found: %s', nirs_dir);
end

raw = struct();
raw.subject_id = subject_id;
raw.eeg_cnt = {};
raw.eeg_mrk = {};
raw.nirs_cnt = {};
raw.nirs_mrk = {};

% ---- EEG ---------------------------------------------------------------
S = load(fullfile(eeg_dir, 'cnt.mat'), 'cnt');
raw.eeg_cnt = S.cnt;
S = load(fullfile(eeg_dir, 'mrk.mat'));
f = fieldnames(S);
raw.eeg_mrk = S.(f{1});
S = load(fullfile(eeg_dir, 'mnt.mat'));
f = fieldnames(S);
raw.eeg_mnt = S.(f{1});

% ---- fNIRS -------------------------------------------------------------
S = load(fullfile(nirs_dir, 'cnt.mat'), 'cnt');
raw.nirs_cnt = S.cnt;
S = load(fullfile(nirs_dir, 'mrk.mat'));
f = fieldnames(S);
raw.nirs_mrk = S.(f{1});
S = load(fullfile(nirs_dir, 'mnt.mat'));
f = fieldnames(S);
raw.nirs_mnt = S.(f{1});

n_eeg = numel(raw.eeg_cnt);
n_nirs = numel(raw.nirs_cnt);
if n_eeg ~= numel(raw.eeg_mrk)
    error('sgtnet:load_public_raw:sessionMismatch', ...
        'Subject %02d: %d EEG recordings but %d EEG marker sets.', ...
        subject_id, n_eeg, numel(raw.eeg_mrk));
end
if n_nirs ~= numel(raw.nirs_mrk)
    error('sgtnet:load_public_raw:sessionMismatch', ...
        'Subject %02d: %d fNIRS recordings but %d fNIRS marker sets.', ...
        subject_id, n_nirs, numel(raw.nirs_mrk));
end
if n_eeg ~= n_nirs
    error('sgtnet:load_public_raw:modalityMismatch', ...
        'Subject %02d: %d EEG sessions but %d fNIRS sessions.', ...
        subject_id, n_eeg, n_nirs);
end

raw.n_sessions = n_eeg;

end
