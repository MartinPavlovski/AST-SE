function [v_mean, v_int_width] = calc_conf_interval( v )
% CALC_CONF_INTERVAL calculates a 90% confidence interval for a population
% mean, based on a sample.
% 
% Requires:
%   v           - observed sample data
% 
% Returns:
%   v_mean      - sample mean
%   v_int_width - the width of the confidence interval

sample_size = max(size(v));
conf_level = 0.1;

v_mean = mean(v);
v_int_width = tinv(1 - conf_level/2,sample_size-1) * (std(v)/sqrt(sample_size));

end