function [pars,allOk] = checkLDMPars(varargin)
%checkLDMPars Validates LDM parameters for use with LDMDist.
%   [pars,allOk] = checkLDMPars(pars) checks and (where possible) completes 
%   LDM parameters with common defaults to comply with LDMDist 
%   requirements.
%
%   pars (in): struct, containing LDM parameters (see manual)
%   verbose (in, optional): 0 (off), 1 (errors and warnings), 2 (errors, 
%       warnings and info)
%   pars (out): struct, containing LDM parameters (see manual)
%   allOk (out): logical, if true, pars is ready for use with LDMDist
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
%   $Id: checkLDMPars.m 69 2015-03-30 16:35:24Z u0066818@kuleuven.be $

pars=varargin{1};
if size(varargin,2)>1
    verbose=varargin{2};
else
    verbose=2;
end

pars=struct(pars);
allOk=true;

[pars,ok]=checkParField(pars,'nStimuli','int32',[],[1,1],1,[],verbose);allOk=ok&&allOk;
if allOk
    nStimuli=pars.nStimuli;
    [pars,ok]=checkParField(pars,'c','single',single(0.1),[1,1],0,[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'a','single',[],[1,1],0,0.99,verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'zr','single',single(0.5),[1,1],0,1,verbose);allOk=ok&&allOk;
    if allOk 
        [pars,ok]=checkParField(pars,'sz','single',single(0),[1,1],0,2*pars.a*max(pars.zr,1-pars.zr),verbose);allOk=ok&&allOk;
    end
    [pars,ok]=checkParField(pars,'Ter','single',single(0),[1,1],0,[],verbose);allOk=ok&&allOk;
    if allOk
        [pars,ok]=checkParField(pars,'sTer','single',single(0),[1,1],0,2*pars.Ter,verbose);allOk=ok&&allOk;
    end
    [pars,ok]=checkParField(pars,'gamma','single',single(0),[1,1],[],[],verbose);allOk=ok&&allOk;%for OU    
    [pars,ok]=checkParField(pars,'v','single',[],[1,nStimuli],[],[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'eta','single',single(zeros(1,nStimuli)),[1,nStimuli],0,[],verbose);allOk=ok&&allOk;
    %has to be checked in relation to settings
    [pars,ok]=checkParField(pars,'profile','single',single(1),[],[],[],verbose);allOk=ok&&allOk;
end

end
