function [settings,allOk] = checkSettings(varargin)
%checkSettings Validates the hardware and algorithm settings variable for
%use with a RTDist wrapper like LDMDist, LCADist, IDMDist.
%   [settings,allOk] = checkSettings(settings) checks and (where possible)
%   completes hardware and algorithm settings with common defaults.
%
%   pars (in): struct, containing settings (see manual)
%   verbose (in, optional): 0 (off), 1 (errors and warnings), 2 (errors, 
%       warnings and info)
%   pars (out): struct, containing settings (see manual)
%   allOk (out): logical, if true, settings is ready for use
%
%   This file is part of the RTDist project
%   Copyright (c) 2014 Stijn Verdonck
%   Copyright (c) 2014 Kristof Meers
%
%   Verdonck, S., Meers, K., & Tuerlinckx, F. (in press). Efficient simulation
%       of diffusion-based choice RT models on CPU and GPU. Behavior Research
%       Methods. doi:10.3758/s13428-015-0569-0
% 
%   RTDist comes without any warranty of any kind. You are not allowed to
%   redistribute a copy of RTDist to others. If you want others to use RTDist,
%   refer them to http://ppw.kuleuven.be/okp/software/RTDist/. See the root
%   folder of this project for full license information in the LICENSE.txt file.
%
%   $Id: checkSettings.m 69 2015-03-30 16:35:24Z u0066818@kuleuven.be $

settings=varargin{1};
if size(varargin,2)>1
    verbose=varargin{2};
else
    verbose=2;
end

settings=struct(settings);
allOk=true;

[settings,ok]=checkParField(settings,'nWalkers','int32',int32(100000),[1,1],1,[],verbose);allOk=ok&&allOk;
[settings,ok]=checkParField(settings,'seed','int32',int32(0),[1,1],[],[],verbose);allOk=ok&&allOk;
[settings,ok]=checkParField(settings,'dt','single',single(0.001),[1,1],0,[],verbose);allOk=ok&&allOk;
[settings,ok]=checkParField(settings,'nBins','int32',int32(3000),[1,1],0,[],verbose);allOk=ok&&allOk;
[settings,ok]=checkParField(settings,'binWidth','single',single(0.001),[1,1],0,[],verbose);allOk=ok&&allOk;
[settings,ok]=checkParField(settings,'nGPUs','int32',int32(1),[1,1],0,[],verbose);allOk=ok&&allOk;
if ok&&(settings.nGPUs>0)
    [settings,ok]=checkParField(settings,'gpuIds','int32',int32(0:(settings.nGPUs-1)),[1,settings.nGPUs],0,[],verbose);allOk=ok&&allOk;
    [settings,ok]=checkParField(settings,'loadPerGPU','single',single(ones(1,settings.nGPUs)),[1,settings.nGPUs],0,[],verbose);allOk=ok&&allOk;
else
    settings.gpuIds=[];
    settings.loadPerGPU=[];
end

end
