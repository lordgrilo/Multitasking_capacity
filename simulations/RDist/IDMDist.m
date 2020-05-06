function [distributions,ok,extra,used] = IDMDist(varargin)
%IDMDist Returns simulated RT distributions for IDM.
%   [distributions,ok,extra,used] = IDMDist(varargin) returns simulated
%   RT distributions for IDM. The function transforms input IDM parameters 
%   to more general parameters and forwards them to the RTDist plugin.
%
%   IDMPars (in): struct, containing IDM parameters (see manual)
%   settings (in, optional): struct, specifying hardware and algoritm
%       settings (see manual)
%   verbose (in, optional): 0 (off), 1 (errors and warnings), 2 (errors, 
%       warnings and info)
%   distributions (out): struct, containing the simulated RT distributions
%   ok (out): logical, true if the simulations have run sucessfully
%   extra (out): struct, contains exra information about the simulations
%   used (out): struct, contains all used parameters and settings
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
%   $Id: IDMDist.m 69 2015-03-30 16:35:24Z u0066818@kuleuven.be $

%default output
distributions=[];
ok=false;
extra=struct;
used=struct;

IDMPars=varargin{1};
if size(varargin,2)>1
    settings=varargin{2};
else
    settings=struct;
end

if size(varargin,2)>2
    verbose=varargin{3};
else
    verbose=2;
end

%settings
[settings,ok]=checkSettings(settings,verbose);
if ~ok
	return
end

%IDMPars
[IDMPars,ok]=checkIDMPars(IDMPars,verbose);
if ~ok
	return
end
    
%transform to general parameters
pars=struct();
%if settings.seed=0, then set time-dependant seed
if settings.seed==0
    temp=clock();
    settings.seed=1+int32(2147483647*temp(6)/60);
end
rng(settings.seed-1);
pars.nDim=IDMPars.nDim;
pars.nStimuli=IDMPars.nStimuli;
pars.N=IDMPars.N;
pars.B=IDMPars.B;
pars.eta=single(zeros(size(pars.B)));
pars.beta=IDMPars.beta;
pars.W=IDMPars.W;
pars.Theta=IDMPars.Theta;
pars.spontaneousTime=IDMPars.spontaneousTime;
pars.dynamics=IDMPars.dynamics;
if pars.dynamics==0
    pars.D=IDMPars.D;
elseif pars.dynamics==1
    pars.deltat=IDMPars.deltat;
    pars.sigma=IDMPars.sigma;
end
pars.startPos=single(repmat(IDMPars.startPos,1,settings.nWalkers));
pars.boxShape=IDMPars.boxShape;
pars.h=IDMPars.h;
if IDMPars.sTer==0
    pars.Ter=single(IDMPars.Ter*ones(1,settings.nWalkers));
else
    pars.Ter=single(IDMPars.Ter+IDMPars.sTer*(rand(1,settings.nWalkers)-0.5));
end
%extend profile if a constant
if numel(IDMPars.profile)==1
	pars.profile=single(IDMPars.profile*ones(settings.nBins,1));
else
    pars.profile=IDMPars.profile;
end
%check if profile is compatible with settings
[pars,ok]=checkParField(pars,'profile','single',single(1),[settings.nBins,1],[],[],verbose);
if ~ok
	return    
end
     
if pars.dynamics==0
    [distributions,extra]=RTDist(pars.nDim,pars.dynamics,pars.startPos,pars.eta,pars.nStimuli,pars.beta,pars.N,pars.W,pars.Theta,pars.B,pars.profile,pars.D,pars.h,pars.boxShape,pars.spontaneousTime,pars.Ter,settings.dt,settings.nBins,settings.binWidth,settings.seed,settings.nWalkers,settings.nGPUs,settings.gpuIds,settings.loadPerGPU);
elseif pars.dynamics==1    
    [distributions,extra]=RTDist(pars.nDim,pars.dynamics,pars.startPos,pars.eta,pars.nStimuli,pars.beta,pars.N,pars.W,pars.Theta,pars.B,pars.profile,pars.sigma,pars.h,pars.boxShape,pars.spontaneousTime,pars.Ter,pars.deltat,settings.nBins,settings.binWidth,settings.seed,settings.nWalkers,settings.nGPUs,settings.gpuIds,settings.loadPerGPU);
end

%walkersAccountedFor check
ok=true;
for i =1:numel(extra)
    ok=ok&&(extra(i).walkersAccountedFor==settings.nWalkers);
end
if ~ok
    if verbose>=1
        display('WARNING: not all walkers accounted for');
    end
end

%add used
used.IDMpars=IDMPars;
used.pars=pars;
used.settings=settings;

end
