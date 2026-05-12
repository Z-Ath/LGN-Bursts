%% Get the max sink and mean sink per session in layer 4 - can be adapted for other layers (lines 54 + 55) 
% get the max bursting per session to correlate with - can be adapted to
% get tonic (line 92) - currently calculating with max but can change to mean

cd('C:\Users\za23e\OneDrive - Florida State University\AllenNP\Analysis') 

load('session_data.mat'); 
load('cells_LGd_ms.mat')

cd('Z:\Zoe\Allen_Institute_Project\Extracted_Data\lfp_band\layer_info')
layer_table = readtable('layer_channel_summary2');
layer_info = readmatrix('layer_channel_summary2');

% sessions to test, i.e. with the lfp avialable - 17 / 20 
s_test = [ 1 2 3 4 5 6 7 9 10 11 12 13 14 15 16 18 20] ; 
% session_Data s = 8 excluded because session 1086410738 is noted to not have LFP in Allen SDK 
% no LFP uploaded for session_data s = 17 and  1120251466,  s= 19 1122903357,

x_cells = cells(1).fr_x; 
x_csd = -0.1 : (1/2500) : 0.75 ;    %time resolution of LFP 


%% Extract max CSD and max BF rate
close all 

trial_type_strings = {'hit_go', 'missed_go','correct_reject','passive'}; 


colours = {'b','m','r','k'};

cd ( 'Z:\Zoe\Allen_Institute_Project\Extracted_Data\lfp_band\CSD_analysis1')

for t = 1 : 4 

    cd(trial_type_strings{t})
    
    sink_info(t).trial_type = trial_type_strings{t}; 
    
    max_sink_ch_time=[];
    
    for s = s_test
        csd=[]; 
        s_id = num2str ( session_data(s).session_id) ; 
        filename = strcat('session_',s_id,'_csd.npy'); 
    
        if FileExists(filename)
            
            csd = readNPY(filename);
            csd = imgaussfilt(csd, [3, 1]);         % same as in CSD plots. 
    
    
            indx = find ( layer_info(:,1) == session_data(s).session_id ) ; 
            l4_ventral = layer_info(indx,9) ;
            l4_dorsal =  layer_info(indx,10) ;
            csd_l4  = csd(l4_ventral:l4_dorsal ,:) ;   % going towards surface, channel 384 outside of brain . 
    
            % time of the post-stimulus interval ( trimmed to: 0.35 - 0.7 ) -
            % other sinks will be found during the stimulus but those are not
            % of interest. 
            csd_l4 = csd_l4 (:, 1140 : 2001) ; 
    
            max_sink_amp(s,1) = min(csd_l4,[],"all") ;                     % maximum sink corresponds to the lowest value in the matrix. 
            [row, col] = find(csd_l4 == max_sink_amp(s,1));
            max_sink_time(s,1)= x_csd(col + 1140) ;                % find the time of the maximum sink 


            % can also check the average sink , i.e. average sink amplitude
            % for upper quartile of amplitudes
            sink_threshold = ( max_sink_amp(s,1)/ 4 ) * 3 ; 
            mean_sink_amp(s,1) = mean (csd_l4(csd_l4 < sink_threshold)) ; 

        end
    
    end
    
    sink_info(t).max_sink_amps = remove0rows (max_sink_amp); 
    sink_info(t).max_sink_times = remove0rows ( max_sink_time ) ; 
    sink_info(t).mean_sink_amps = remove0rows ( mean_sink_amp ) ; 
    
    
    % get the burst firing rates around that time 
    
    session_ids = unique([cells.session_id]);
    th_sessions = cell2mat({cells.session_id}); % Extract the session numbers into a cell array
    
    for s = s_test
    
        indx_all_th = find(th_sessions == session_data(s).session_id) ; 
       
        % find the maxmimum burst in the post-stimulus interval
        burst_rate_session = mean(vertcat ( cells(indx_all_th).(strcat('fr_', trial_type_strings{t},'_burst')) ) ) ; 
        peak_BF(s,1)  = max(burst_rate_session (671:741)) ; 
        indx_of_max_bursts = find (burst_rate_session(671:741) == max(burst_rate_session (671:741)) )  ; 
        time_of_max_bursts = x_cells(indx_of_max_bursts+671); 
        time_of_peak_BF(s,1)= time_of_max_bursts(1); 
    
    
    end
    
    sink_info(t).peak_BF= remove0rows(peak_BF) ; 
    sink_info(t).ts_peak_BF = remove0rows(time_of_peak_BF);  
   
end

%%
% % Fit linear model to each and all 

figure(11)

for t = 1 :  4

    subplot (3,2,t)

    z =  sink_info(t).max_sink_amps ; 
    zz = sink_info(t).peak_BF ; 

    % Fit linear model
    mdl = fitlm(z, zz);
    
    R2 = mdl.Rsquared.Ordinary;
    figure(11)
    hold on 
    scatter(z, zz, 40, 'filled', colours{t})

    h = plot(mdl);
    h(1).Visible = 'off';      
    legend('off');             
    mdl.Coefficients.pValue(2)
    hold off;
    
    title(sprintf('CSD sink vs Burst strength (R^2 = %.3f)', R2));
    xlabel('Sink amplitude ');
    ylabel('Burst strength ');

    subplot(3,2,5)
    hold on
    scatter(z, zz, 40, 'filled', colours{t})

end 

z = vertcat(sink_info.max_sink_amps) ;
zz = vertcat(sink_info.peak_BF) ;

mdl = fitlm(z, zz);

R2 = mdl.Rsquared.Ordinary;
figure(11)

hold on;
h = plot(mdl);
h(1).Visible = 'off';      % hide data scatter/points
legend('off');             % remove legend
hold off;
mdl.Coefficients.pValue(2)
title(sprintf('CSD sink vs Burst strength (R^2 = %.3f)', R2));
xlabel('Sink amplitude ');
ylabel('Burst strength ');

% cd ('C:\Users\za23e\OneDrive - Florida State University\AllenNP\Analysis\ms_fig_panels')
% exportgraphics(gcf, 'max_burst_to_max_sink_correlations.pdf', 'ContentType', 'vector');
