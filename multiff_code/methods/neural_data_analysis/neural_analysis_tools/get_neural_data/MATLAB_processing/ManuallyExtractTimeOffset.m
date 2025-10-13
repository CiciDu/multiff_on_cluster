
% For Schro data_0328
% outputFileName = '/Users/dusiyi/Documents/Multifirefly-Project/all_monkey_data/time_calibration/monkey_Schro/data_0328/neural_event_time.txt'
% %openNEV('report', 'read', 'm53s393.nev')

% For Schro data_0404
outputFileName = '/Users/dusiyi/Documents/Multifirefly-Project/all_monkey_data/time_calibration/monkey_Schro/data_0404/neural_event_time.txt'
openNEV('report', 'read', 'm53s420.nev')


label = NEV.Data.SerialDigitalIO.UnparsedData
time = NEV.Data.SerialDigitalIO.TimeStampSec

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Need to manually extract and copy data for label and time

% Data (copied from NEV.DATA.SerialDigitalIO.UnparsedData and NEV.DATA.SerialDigitalIO.TimeStampsSec (forgot the precise name for this one)
% label = [16;64;1;64;4;64;4;64;4;64]
% time = [5.09583333333333,5.09593333333333,5.34480000000000,5.34490000000000,20.2745000000000,20.2746000000000,21.4244666666667,21.4245666666667,25.0624000000000,25.0625000000000];

% Create a table
dataTable = table(time', label, 'VariableNames', {'time', 'label'});

% Display the table
disp(dataTable);

% Save the table to a text file
writetable(dataTable, outputFileName);

disp(['Data saved to ', outputFileName]);


