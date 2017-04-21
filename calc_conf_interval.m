function v_mean_diff_pair = calc_conf_interval( v )

sample_size = max(size(v));
conf_level = 0.1;

v_mean = mean(v);
v_diff = tinv(1 - conf_level/2,sample_size-1) * (std(v)/sqrt(sample_size));
v_mean_diff_pair = [v_mean v_diff];

end

