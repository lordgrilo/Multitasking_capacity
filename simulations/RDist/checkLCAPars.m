function [pars,allOk] = checkLCAPars(varargin)
%checkLCAPars Validates LCA parameters for use with LCADist.
%   [pars,allOk] = checkLCAPars(pars) checks and (where possible) completes 
%   LCA parameters with common defaults to comply with LCADist 
%   requirements.
%
%   pars (in): struct, containing LCA parameters (see manual)
%   verbose (in, optional): 0 (off), 1 (errors and warnings), 2 (errors, 
%       warnings and info)
%   pars (out): struct, containing LCA parameters (see manual)
%   allOk (out): logical, if true, pars is ready for use with LCADist
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
%   $Id: checkLCAPars.m 69 2015-03-30 16:35:24Z u0066818@kuleuven.be $

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
    [pars,ok]=checkParField(pars,'c','single',single(0.1*ones(nDim,1)),[nDim,1],0,[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'startPos','single',single(0.0001*ones(nDim,1)),[nDim,1],single(0.0001),single(0.9999),verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'Gamma','single',[],[nDim,nDim],[],[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'a','single',[],[nDim,1],single(0.0001),single(0.9999),verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'v','single',[],[nDim,nStimuli],[],[],verbose);allOk=ok&&allOk;
    [pars,ok]=checkParField(pars,'Ter','single',single(0),[1,1],0,[],verbose);allOk=ok&&allOk;
    if allOk
        [pars,ok]=checkParField(pars,'sTer','single',single(0),[1,1],0,2*pars.Ter,verbose);allOk=ok&&allOk;
    end
    %has to be checked in relation to settings
    [pars,ok]=checkParField(pars,'profile','single',single(1),[],[],[],verbose);allOk=ok&&allOk;
end

end
