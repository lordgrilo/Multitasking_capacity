% This script runs a validation test. It looks in validation_logs/ for mat
% files that contain runs with the unmodified original code base. It then
% compares the output to ensure that the outputs match.
VALIDATION_DATA_DIR = 'validation_logs/';

% A list of variables to check for exact equality. These are the important
% output variables of the code. We need to specify these because certain
% optimizations of the code have added new variables that will make
% structures not equal anymore. Also, certain variables like timing
% data will be different between runs.
CHECK_VARS = {'optimalControlPolicy_meanAcc', 'optimalControlPolicy_respProb', 'A_MIS', 'extractedMIS', 'NFeatures'};

validation_files = dir(sprintf('%s/*.mat', VALIDATION_DATA_DIR));

fprintf(1, 'Validating Current Code Base:\n');
for ii=1:length(validation_files)
    
    full_name = sprintf('%s/%s', VALIDATION_DATA_DIR, validation_files(ii).name);
    
    % Get the inputArg used for the current result data file.
    Svalid = load(full_name);
    fprintf(1, '\tValidating inputArg=%s: ', Svalid.inputArg);
    
    % Run the code with this input argument, also silence output
    outfile = Giovanni_Sim1_V8_fun(Svalid.inputArg, true);
    
    Stest = load(outfile);
    
    for kk=1:length(CHECK_VARS)
        if(~isequal(getfield(Svalid, CHECK_VARS{kk}), getfield(Stest, CHECK_VARS{kk})))
            fprintf(1, '%s field not equal!\n', CHECK_VARS{kk});
        end
    end

    fprintf(1, 'Valid!\n');
end
fprintf(1, 'Done\n');