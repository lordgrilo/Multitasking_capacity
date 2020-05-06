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
%   $Id: RTDistTestIndices.m 69 2015-03-30 16:35:24Z u0066818@kuleuven.be $

%test indices
IDMPars.nDim=2;
IDMPars.N=[1000;1000];
IDMPars.W=[52500,8400;8400,52500];
IDMPars.Theta=[51450;51450];
IDMPars.nStimuli=1;
IDMPars.B=[2500;2500];
IDMPars.h=[0.4,0.4;0.4,0.4];
IDMPars.Ter=0.3;
IDMPars.spontaneousTime=1;
IDMPars.dynamics=0;
IDMPars.D=[0.05;0.05];
IDMPars=checkIDMPars(IDMPars);
settings=struct();
settings.seed=0;
settings=checkSettings(settings);

settings.nWalkers=100000;
settings.nGPUs=1;

%check indices B
IDMParsTest=IDMPars;
IDMParsTest.nStimuli=2;
%each column is a stimulus
IDMParsTest.B=[3000,3500;2000,1500];

[distributions,extra,used]=IDMDist(IDMParsTest,settings);
counts=sum(distributions,1);
(counts(1)-counts(2))/(counts(1)+counts(2))
(counts(3)-counts(4))/(counts(3)+counts(4))

%check indices h
IDMParsTest=IDMPars;
%each column is a box
IDMParsTest.h=[0.4,0.4;1,0.4];
IDMParsTest.B=[5000;5000];

[distributions,extra,used]=IDMDist(IDMParsTest,settings);
counts=sum(distributions,1);
(counts(1)-counts(2))/(counts(1)+counts(2))
