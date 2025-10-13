csv_file_path = '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/multiff_code/methods/eye_position_analysis/neural_data_analysis/get_neural_data/bruno_mapping_table.csv'

% Read the CSV file into a table
opts = detectImportOptions(csv_file_path, 'Delimiter', ',');

% Set the first column as row names
opts.RowNamesColumn = 0;

% Read the table with the specified options
data_table = readtable(csv_file_path, opts);

% Add a new index column
data_table.Index = (1:height(data_table))';

% Move the new index column to the first position
data_table = movevars(data_table, 'Index', 'Before', data_table.Properties.VariableNames{1});

% Display the first few rows of the modified table
disp(head(data_table));

% iterate through each row of data_table
for i = 1:height(data_table)
    % get the local_file_path and neural_data_path
    neural_data_path = data_table{i, 'hdrive_path'}{1}
    neural_event_time_path = data_table{i, 'neural_event_time_path'}{1};
    % get the parent folder of neural_event_time_path
    local_path = fileparts(neural_event_time_path);
    if ~isfolder(local_path)
        mkdir(local_path);
    end
    disp(neural_data_path)
    AlignNeuralData(neural_data_path, neural_event_time_path)

end
