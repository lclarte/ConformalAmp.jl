using ProgressBars
using ConformalAmp

function run_experiment(x, y, method, fcp, seeds, n_train)
    coverage_list    = []
    mean_length_list = []
    time_list        = []

    n_test = size(x, 1) - n_train
        
    for seed in ProgressBar(1:seeds)
        ci_gamp_list = zeros((n_test, 2))
        time_list = []

        x_train, x_test, y_train, y_test = train_test_split(x, y, n_train, seed)
    
        if method isa ConformalAmp.ExactHomotopy
            # debut = time()
            # ci_gamp_list = ConformalAmp.get_confidence_interval(problem, x_train, y_train, x_test, fcp, method)
            # fin = time()
            # push!(time_list, fin - debut)
            debut = time()
            # build a (1, 13) matrix containing x_test[i, :]
            ci_gamp_list = ConformalAmp.get_confidence_interval(problem, x_train, y_train, x_test, fcp, method)
            fin = time()
                
            push!(time_list, (fin - debut) / n_test)
        else
            for i in 1:n_test
                # compute the confidence ntervals 
                debut = time()
                ci_gamp = ConformalAmp.get_confidence_interval(problem, x_train, y_train, x_test[i, :], fcp, method)
                fin = time()
                
                ci_gamp_list[i, :] = [minimum(ci_gamp), maximum(ci_gamp)]
                push!(time_list, fin - debut)
            end
        end
    
        # compute the coverage of the GAMP
        coverage = mean([ci_gamp_list[i, 1] <= y_test[i] <= ci_gamp_list[i, 2] for i in 1:n_test])
    
        # compute the mean length of the confidence intervals
        mean_length = mean([ci_gamp_list[i, 2] - ci_gamp_list[i, 1] for i in 1:n_test])
    
        push!(coverage_list, coverage)
        push!(mean_length_list, mean_length)
    end

    return coverage_list, mean_length_list, time_list
end