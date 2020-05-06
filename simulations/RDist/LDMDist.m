function [distributions,ok,extra,used] = LDMDist(varargin)
%LDMDist Returns simulated RT distributions for LDM.
%   [distributions,ok,extra,used] = LDMDist(varargin) returns simulated
%   RT distributions for LDM. The function transforms input LDM parameters 
%   to more general parameters and forwards them to the RTDist plugin.
%
%   LDMPars (in): struct, containing LDM parameters (see manual)
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
%   $Id: LDMDist.m 89 2015-06-14 09:40:30Z u0066818@kuleuven.be $

%default output
distributions=[];
ok=false;
extra=struct;
used=struct;

LDMPars=varargin{1};
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

%LDMPars
[LDMPars,ok]=checkLDMPars(LDMPars,verbose);
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
pars.boxShape=int32(3);
pars.nDim=int32(1);
pars.dynamics=int32(0);
pars.N=single(0);
pars.beta=single(1);
pars.Theta=single(0);
pars.spontaneousTime=single(0);

pars.D=single(0.5*LDMPars.c^2);
startPos=0.5+(LDMPars.zr-0.5)*LDMPars.a;
if LDMPars.sz==0
    pars.startPos=single(startPos*ones(1,settings.nWalkers));
else
    pars.startPos=single(startPos+LDMPars.sz*(rand(1,settings.nWalkers)-0.5));    
end
pars.W=single(0.5*LDMPars.gamma)/pars.D;%OU
pars.nStimuli=int32(LDMPars.nStimuli);
pars.eta=single(LDMPars.eta)/pars.D;
pars.h=single([0.5*(1-LDMPars.a),0.5*(1-LDMPars.a)]);
pars.B=single(LDMPars.v-LDMPars.gamma*pars.h(1))/pars.D;
if LDMPars.sTer==0
    pars.Ter=single(LDMPars.Ter*ones(1,settings.nWalkers));
else
    pars.Ter=single(LDMPars.Ter+LDMPars.sTer*(rand(1,settings.nWalkers)-0.5));
end
%extend profile if a constant
if numel(LDMPars.profile)==1
    pars.profile=single(LDMPars.profile*ones(settings.nBins,1));
else
    pars.profile=single(LDMPars.profile);
end
%check if profile is compatible with settings
[pars,ok]=checkParField(pars,'profile','single',single(1),[settings.nBins,1],[],[],verbose);
if ~ok
	return    
end

[distributions,extra]=RTDist(pars.nDim,pars.dynamics,pars.startPos,pars.eta,pars.nStimuli,pars.beta,pars.N,pars.W,pars.Theta,pars.B,pars.profile,pars.D,pars.h,pars.boxShape,pars.spontaneousTime,pars.Ter,settings.dt,settings.nBins,settings.binWidth,settings.seed,settings.nWalkers,settings.nGPUs,settings.gpuIds,settings.loadPerGPU);

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
used.LDMPars=LDMPars;
used.pars=pars;
used.settings=settings;

end
