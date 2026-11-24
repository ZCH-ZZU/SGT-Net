function mnt_out = subset_montage(mnt, clab, context)
%SUBSET_MONTAGE  Restrict a montage to the channels present in the data.
%
%   MNT_OUT = SUBSET_MONTAGE(MNT, CLAB) copies the fields clab, x, y and
%   pos_3d of MNT and keeps only the channels listed in CLAB, in that order.
%   Missing channels are reported and filled with zeros/empty labels so that
%   the output always aligns with the data array.
%
%   This helper only makes the montage consistent with the data columns; it
%   does not perform channel selection for the model.
%
%   See also SGTPNET.BUILD_PUBLIC_DATASET, SGTPNET.H5_INIT.

if nargin < 3 || isempty(context)
    context = 'montage';
end

if isempty(mnt) || ~isstruct(mnt) || ~isfield(mnt, 'clab')
    error('sgtnet:subset_montage:badMontage', ...
        '%s: montage must be a structure with a clab field.', context);
end

clab = cellstr(clab);
[tf, idx] = ismember(clab, mnt.clab);
if ~all(tf)
    warning('sgtnet:subset_montage:missingChannels', ...
        '%s: %d channel(s) not found in the montage: %s', ...
        context, sum(~tf), strjoin(clab(~tf), ', '));
end

mnt_out = struct();
mnt_out.clab = clab;

fields_1d = {'x', 'y'};
for k = 1:numel(fields_1d)
    f = fields_1d{k};
    if isfield(mnt, f)
        v = zeros(1, numel(clab));
        tmp = mnt.(f);
        v(tf) = tmp(idx(tf));
        mnt_out.(f) = v;
    end
end

if isfield(mnt, 'pos_3d')
    v = zeros(3, numel(clab));
    v(:, tf) = mnt.pos_3d(:, idx(tf));
    mnt_out.pos_3d = v;
end

end
