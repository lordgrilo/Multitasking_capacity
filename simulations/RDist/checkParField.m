function [pars,ok] = checkParField(pars,field,className,default,dimensions,minValue,maxValue,verbose)
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
%   $Id: checkParField.m 69 2015-03-30 16:35:24Z u0066818@kuleuven.be $

ok=true;
if isfield(pars,field)
    %check class
    if ~isa(pars.(field),className)
        pars.(field)=cast(pars.(field),className);
        if verbose>=2
            display(['INFO: ',field, ' converted to ', className]);
        end
    end
    %check dimensions
    if ~isempty(dimensions)
        if max(size(pars.(field))~=dimensions)
            if verbose>=1
                display(['ERROR: ',field, ' not correctly dimensioned: [',num2str(size(pars.(field),1)),',',num2str(size(pars.(field),2)),'] should be [',num2str(dimensions(1)),',',num2str(dimensions(2)),']']);
            end
            ok=false;
        end
    end
else
    if isempty(default)
        if verbose>=1
            display(['ERROR: ',field, ' required']);
        end
        ok=false;
    else
        if verbose>=2
            display(['INFO: ','using default ',field]);
        end
        pars.(field)=default;
    end
end
%check min/max
if ok
    if numel(minValue)>0
        if max(pars.(field)<minValue)
            if verbose>=1
                display(['ERROR: ',field, ' too low (maybe in combination with other parameter values)']);
            end
            ok=false;
        end
    end
    if numel(maxValue)>0
        if max(pars.(field)>maxValue)
            if verbose>=1
                display(['ERROR: ',field, ' too high (maybe in combination with other parameter values)']);
            end
            ok=false;
        end
    end
end

end
