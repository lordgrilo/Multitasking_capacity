function [pars,allOk] = checkIDMPars(varargin)
%checkIDMPars Validates IDM parameters for use with IDMDist.
%   [pars,allOk] = checkIDMPars(pars) checks and (where possible) completes 
%   IDM parameters with common defaults to comply with IDMDist 
%   requirements.
%
%   pars (in): struct, containing IDM parameters (see manual)
%   verbose (in, optional): 0 (off), 1 (errors and warnings), 2 (errors, 
%       warnings and info)
%   pars (out): struct, containing IDM parameters (see manual)
%   allOk (out): logical, if true, pars is ready for use with IDMDist
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
%   $Id: checkIDMPars.m 69 2015-03-30 16:35:24Z u0066818@kuleuven.be $

pars=varargin{1};
if size(varargin,2)>1
    verbose=varargin{2};
else
    verbose=2;
end

pars=struct(pars);
allOk=true;

[pars,ok]=checkParField(pars,'nDim','int32',[],[1,1],1,[],verbose);allOk=ok&&allOk;
[pars,ok]=checkParField(pars,'nStimuli','int32',[],[1,1],1,[],verbose);allOk=ok&&allOk;
if allOk
    nDim=pars.nDim;
    nStimuli=pars.nStimuli;
    [pars,ok]=checkParField(pars,'startPos','single',single(0.3*ones(nDim,1)),[nDim,1],0.001,0.999,verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'beta','single',single(1/24),[1,1],0,[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'N','single',[],[nDim,1],0,[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'W','single',[],[nDim,nDim],0,[],verbose);allOk=ok&&allOk;
    if ok
        if ~min(pars.W==transpose(pars.W))
            if verbose>=1
                display('ERROR: ','W not symmetric');
            end
            allOk=false;
        end
    end
    [pars,ok]=checkParField(pars,'Theta','single',[],[nDim,1],0,[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'boxShape','int32',int32(0),[1,1],0,2,verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'h','single',[],[nDim,nDim],0,[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'B','single',[],[nDim,nStimuli],[],[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'Ter','single',single(0),[1,1],0,[],verbose);allOk=ok&&allOk;
    if allOk
        [pars,ok]=checkParField(pars,'sTer','single',single(0),[1,1],0,2*pars.Ter,verbose);allOk=ok&&allOk;
    end
    [pars,ok]=checkParField(pars,'spontaneousTime','single',[],[1,1],0,[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'dynamics','int32',[],[1,1],0,1,verbose);allOk=ok&&allOk;
    if ok
        switch pars.dynamics
            case 0
                [pars,ok]=checkParField(pars,'D','single',[],[nDim,1],0,[],verbose);allOk=ok&&allOk;
            case 1
                [pars,ok]=checkParField(pars,'sigma','single',[],[nDim,1],0,[],verbose);allOk=ok&&allOk;
                [pars,ok]=checkParField(pars,'deltat','single',[],[1,1],0,[],verbose);allOk=ok&&allOk;
        end
    end
    %has to be checked in relation to settings
    [pars,ok]=checkParField(pars,'profile','single',single(1),[],[],[],verbose);allOk=ok&&allOk;
end

end
