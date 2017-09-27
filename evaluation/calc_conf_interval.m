function [v_mean, v_int_width] = calc_conf_interval( v )

sample_size = max(size(v));
conf_level = 0.1;

v_mean = mean(v);
v_int_width = tinv(1 - conf_level/2,sample_size-1) * (std(v)/sqrt(sample_size));

end

