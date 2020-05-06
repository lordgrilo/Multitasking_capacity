% This script runs a performance test on the current code base. It saves
% the output into performance_logs/
PERF_TEST_DIR = 'performance_logs/';

% Get the current branch we are on, this will be used to save the output
% file
[result, branch_txt] = system('git branch | grep \\*');
branch = strtrim(branch_txt(3:end));

% Range of graph sizes to run the code on
N_range = 3:7;

% Fraction of relevant tasks to set. We will try to construct the worst
% case graph up to N^2 * TASK_FRACTION number of edges. This should be an
% all diagnoal matrix.
TASK_FRACTION = 0.5;

% If we are running the validation code, lets limit the size of our input
% graph to around 8. The code is too slow beyond that.
if(strcmpi(branch, 'validation'))
    N_range = 3:6;
end

fprintf(1, 'Peformance Test: Nrange=(%d,%d)\n', min(N_range), max(N_range));

elapsed = zeros(length(N_range), 1);
for ii=1:length(N_range)
    N = N_range(ii);
    
    % Construct a matrix by filling in the diagnols of the matrix until we
    % get up to TASK_FRACTION*(N^2) number of edges\tasks.
    nT = ceil(TASK_FRACTION*(N^2));
    K = reshape(1:(N^2), [N N]);
    idx = []; dd = 0;
    while(length(idx) < nT)
        idx = [idx; diag(K,dd)];
        
        if(dd == 0 || dd < 0)
            dd = abs(dd) + 1;
        else
            dd = -dd;
        end
    end
    idx = idx(1:nT);
    A = zeros(N); A(idx) = 1;
    
    A
    
    prev_time = tic;
    Giovanni_Sim1_V8_fun(A);
    elapsed(ii) = toc(prev_time);
    fprintf(1, 'Elapsed Time=%f secs\n', elapsed(ii));
end

save(sprintf('%s/%s.mat', PERF_TEST_DIR, branch), 'elapsed', 'N_range');
