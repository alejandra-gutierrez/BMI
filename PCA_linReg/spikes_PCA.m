function [principal_spikes, Vs, Ds, M] = spikes_PCA(spike_rate, cutoff)
    % function:
    % input:
    %    spike rates
    %       format: [N_neurons x t]
    %   cutoff - 0 < proportion < 1
    %       cutoff to decide how many significant eigenvalues
    % output:
    %   principal_spikes = spikes along principal component coordinates (reduced size)
    %   Vs = eigenvectors (full)
    %   Ds = eigenvalues  (full)
    %   M = number of principal elements conserved in spikes
    
    
    % check if cutoff on eigenvalues within correct range
    if cutoff<0 || cutoff >1
        cutoff = 0.2;
    end
    
    spikes_mean = mean(spike_rate, 2);

    A = cov(spike_rate'); % 98x98 covariance matrix
    [V,D] = eig(A);
    [d,ind] = sort(diag(D), 'descend');
    Ds = D(ind, ind);
    Vs = V(:, ind);
    
    
    Is = find(diag(Ds)<max(Ds,[],'all')*cutoff);
    M = Is(1);
    V_red = Vs(:,1:M); % principal component vectors

    principal_spikes = V_red'*(spike_rate - spikes_mean);
end