function allOk = testWrappers(varargin)
%testWrappers Tests the different components of the RTDist project.
%   allOk = testWrappers(varargin) simulates specific test distributions 
%   with subsequently LDMDist, LCADist and IDMDist, and compares these to
%   the contents of testPdfs.mat. The user can specify the calculation
%   device and for each test, a benchmark time is displayed.
%
%   nGPUs (in, optional): integer, specifying the number of GPUs to be 
%       used for the tests (0 means CPU and is the default)
%   gpuIds (in, optional): integer [1xnGPUs], specifying the gpuIds
%       (see manual) if nGPUs>0 (the default value is 1:nGPUs)
%   loadPerGPU (in, optional): float [1xnGPUs], specifying the relative 
%       load per GPU if nGPUs>0 (the default value is ones(1,nGPUs))
%   allOk (out): logical, if true, test was completed successful
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
%   $Id: testWrappers.m 69 2015-03-30 16:35:24Z u0066818@kuleuven.be $

settings=struct();
if size(varargin)==0
    settings.nGPUs=0;
else
    settings.nGPUs=varargin{1};
end
if size(varargin,2)>1
    settings.gpuIds=varargin{2};
end
if size(varargin,2)>2
    settings.loadPerGPU=varargin{3};
end
settings.nWalkers=1000000;
settings.seed=1;
[settings,ok]=checkSettings(settings,1);
if ~ok
    display('ERROR: settings not correct');
    return
end

allOk=true;

%LDMPars
LDMPars=struct();
LDMPars.nStimuli=4;
LDMPars.a=0.08;
LDMPars.v=[0.4,0.25,0.1,0];
LDMPars.Ter=0.3;
tic;
[LDMDistributions0,ok,extra,used]=LDMDist(LDMPars,settings,1);allOk=ok&&allOk;
LDMt=toc;

%LCAPars
LCAPars=struct();
LCAPars.nDim=2;
LCAPars.nStimuli=4;
LCAPars.a=[0.08;0.08];
LCAPars.v=[0,0.05,0.1,0.15;0.3,0.25,0.2,0.15];
LCAPars.Ter=0.3;
LCAPars.Gamma=[0.1,0.1;0.1,0.1];
tic;
[LCADistributions0,ok,extra,used]=LCADist(LCAPars,settings,1);allOk=ok&&allOk;
LCAt=toc;

%IDMPars
IDMPars=struct();
IDMPars.nDim=2;
IDMPars.nStimuli=4;
IDMPars.N=[1000;1000];
IDMPars.Theta=[51450;51450];
IDMPars.W=[52500,8400;8400,52500];
IDMPars.B=[2000,2250,2500,2750;3500,3250,3000,2750];
IDMPars.h=[0.4,0.4;0.4,0.4];
IDMPars.dynamics=0;
IDMPars.D=[0.05;0.05];
IDMPars.spontaneousTime=1;
IDMPars.Ter=0.3;
tic;
[IDMDistributions0,ok,extra,used]=IDMDist(IDMPars,settings,1);allOk=ok&&allOk;
IDMt=toc;

LDMPdfs0=double(LDMDistributions0)/double(settings.nWalkers);
LCAPdfs0=double(LCADistributions0)/double(settings.nWalkers);
IDMPdfs0=double(IDMDistributions0)/double(settings.nWalkers);
%LDMPdfs=LDMPdfs0;LCAPdfs=LCAPdfs0;IDMPdfs=IDMPdfs0;
%save('testPdfs.mat','LDMPdfs','LCAPdfs','IDMPdfs');
load('testPdfs.mat');

LDMLlh=sum(sum(LDMPdfs0.*log(max(LDMPdfs,0.000001))));
if (-24.90<LDMLlh)&&(LDMLlh<-24.88)
    display(['LDMDist passed test in ',num2str(LDMt),' sec']);
    figure;
    hold on;
    title('LDMDist');
    plot(LDMPdfs); 
    selection=1:10:3000;
    plot(repmat(selection',1,size(LDMPdfs,2)),LDMPdfs0(selection,1:size(LDMPdfs,2)),'.');    
else
    display('LDMDist did not pass test');
    allOk=false;
end

LCALlh=sum(sum(LCAPdfs0.*log(max(LCAPdfs,0.000001))));
if (-26.25<LCALlh)&&(LCALlh<-26.23)
    display(['LCADist passed test in ',num2str(LCAt),' sec']);    
    figure;
    hold on;
	title('LCADist');
    plot(LCAPdfs);
    selection=1:10:3000;
    plot(repmat(selection',1,size(LCAPdfs,2)),LCAPdfs0(selection,1:size(LCAPdfs,2)),'.');    
else
    display('LCADist did not pass test');
    allOk=false;
end    

IDMLlh=sum(sum(IDMPdfs0.*log(max(IDMPdfs,0.000001))));
if (-22.05<IDMLlh)&&(IDMLlh<-22.03)
    display(['IDMDist passed test in ',num2str(IDMt),' sec']);        
    figure;
    hold on;
	title('IDMDist');
    plot(IDMPdfs);    
    selection=1:10:3000;
    plot(repmat(selection',1,size(IDMPdfs,2)),IDMPdfs0(selection,1:size(IDMPdfs,2)),'.');    
else
    display('IDMDist did not pass test');
    allOk=false;
end    

end