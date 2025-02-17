%-------------------------------------------------------------------------%
% A Multiple Imputation Approach for Flexible Modelling of Interval-censored 
% Data with Missing and Censored Covariates
% - Lou Y.C. 
%-------------------------------------------------------------------------%
clear; close all; clc; rng(666)
n = 400; nrep = 100; m = 4; Nimp = 10; maxstvtimes = 100; rho0 = 1;
bet0 = [-0.5; -0.8; 0.6; 1]; pdim = size(bet0, 1); px = pdim-1; alph = [0.0; 0.50];
dtlim = 0.85; %0.50; 
l1 = 0.02; l2 = l1; u1 = 4.98; u2 = u1; nt = 8; td1 = (u1-l1)/(nt+1); vt1 = (l1+td1):td1:(u1-td1);
summary = zeros(nrep, pdim); summarymicc = summary; summaryhoc = summary;
summarycc = summary; summaryful = summary;
sumvar = zeros(nrep, pdim); sumvarmicc = sumvar; sumvartypeB = sumvar;
sumvarhoc = sumvar; sumvarcc = sumvar; sumvarful = sumvar;
opt = optimset('Display','off');
optfmin = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton'); 
parfor rep = 1:nrep
warning('off')
X1 = random('Normal',0,1,n,1); X2 = random('Normal',X1/2,1);
X3 = random('Binomial', 1, exp(X1)./(1+exp(X1))); X = [X1,X2, X3]; X2obs = X2; X3obs = X3;
xi0 = random('Normal',0,0.25,n,1);
% xi01 = random('Normal',-0.6,0.15,n,1);
% xi02 = random('Normal',0.4,0.05,n,1);
% xip = random('Binomial', 1, 0.4, n, 1);
% xi0 = xip .* xi01 + (1 - xip) .* xi02;
ZT = ([ones(n,1), X1] * alph + xi0); Z = ZT; Zobs = Z; W = [X, Z];
u_t = random('Uniform',0,1,n,1);
T = 3*((-rho0.*log(1-u_t) + 1).^(1/rho0) - 1)./exp(W*bet0);  % rho0=1 : Cox
% T = 3*((1-u_t).^(-rho0)-1)./(rho0 * exp(W*bet0)); % rho0=1 : Log-odds
L = zeros(n, 1); R = L; indL = L; indR = L; mi1 = zeros(1,nt);
for iid = 1:n
    while sum(mi1) == 0
        mi1 = random('Binomial',1,0.7,1,nt);
    end
    obst1 = nonzeros(vt1.*mi1); ant1 = length(obst1);
    if T(iid) <= obst1(1)
        L(iid) = l1; R(iid) = obst1(1); indL(iid) = 0; indR(iid) = 1;
    elseif ant1 > 1
        for j = 2:ant1
            if (T(iid) > obst1(j-1)) && (T(iid) <= obst1(j))
                L(iid) = obst1(j-1); R(iid) = obst1(j); indL(iid) = 1; indR(iid) = 1;
            end
        end
    end
    if T(iid) > obst1(ant1)
        L(iid) = obst1(ant1); R(iid) = u1; indL(iid) = 1; indR(iid) = 0;
    end
end
Z_DL_valid = (Z <= dtlim); Zobs(not(Z_DL_valid)) = dtlim; DL_valid = Z_DL_valid;
Z_MS_valid = (random('Binomial', 1, 1./(1+exp(1.5+0.5*X1))) == 0); % MAR
Zobs(not(Z_MS_valid),:) = NaN; MSZ_valid = Z_MS_valid; 
DL_valid(not(MSZ_valid)) = 1; Z_ALL_valid = and(DL_valid, MSZ_valid);
X2obs_id_valid = (random('Binomial', 1, 1./(1+exp(0.6+1.0*X1+(L+R)/3))) == 0); % MAR
X2obs(not(X2obs_id_valid),:) = NaN; MSX2_valid = X2obs_id_valid;
X3obs_id_valid = 1+0*(random('Binomial', 1, 1./(1+exp(0.4+0.8*X1+(L+R)/2))) == 0); % MAR
X3obs(not(X3obs_id_valid),:) = NaN; MSX3_valid = X3obs_id_valid; 
MS_valid = and(MSX2_valid, MSX3_valid); ID_valid = and(DL_valid, MS_valid);
Xobs = [X1, X2obs, X3obs]; Wobs = [X1, X2obs, X3obs, Zobs];
bernl = 0.02; bernr = max(R)+0.1; Laml_true = L/3; Lamr_true = R/3;
bl1 = zeros(n,(m+1)); br1 = bl1;
for iid = 0:m
   bl1(:,(iid+1)) = bern(iid,m,bernl,bernr,L);
   br1(:,(iid+1)) = bern(iid,m,bernl,bernr,R);
end
phl01 = fminunc(@(x)sum((Laml_true-bl1*cumsum(exp(x))).^2),zeros((m+1),1), opt);
phr01 = fminunc(@(x)sum((Lamr_true-br1*cumsum(exp(x))).^2),zeros((m+1),1), opt);
ph01 = (phl01+phr01)/2;
[mlful,~,~,~,~,heslful] = fminunc(@(para) loglik_ic(para, pdim, m, W, bl1, br1, indL, indR, rho0, ones(n,1)), [bet0; ph01], optfmin);
summaryful(rep, :) = mlful(1:pdim);
diaggful = diag(pinv(heslful)); dihful = sqrt(abs(diaggful(1:pdim)));
sumvarful(rep, :) = dihful(1:pdim);
[mlhoc,~,exitflaghoc,~,~,heslhoc] = fminunc(@(para) loglik_ic(para, pdim, m, Wobs, bl1, br1, indL, indR, rho0, MS_valid), [bet0; ph01], optfmin);
summaryhoc(rep, :) = mlhoc(1:pdim);
diagghoc = diag(pinv(heslhoc)); dihhoc = sqrt(abs(diagghoc(1:pdim)));
sumvarhoc(rep, :) = dihhoc(1:pdim);
[mlcc,~,exitflagcc,~,~,heslcc] = fminunc(@(para) loglik_ic(para, pdim, m, Wobs, bl1, br1, indL, indR, rho0, ID_valid), [bet0; ph01], optfmin);
summarycc(rep, :) = mlcc(1:pdim);
diaggcc = diag(pinv(heslcc)); dihcc = sqrt(abs(diaggcc(1:pdim)));
sumvarcc(rep, :) = dihcc(1:pdim);
wrk_cov = [X1, L, R]; summary_impcc = zeros(Nimp, pdim); sumvar_impcc = summary_impcc;
for nofimp = 1:Nimp
reDL_valid = 0*DL_valid; reMSX2_valid = 0*MSX2_valid; reMSX3_valid = 0*MSX3_valid;
reMS_valid = 0*MS_valid; reID_valid = 0*ID_valid;
while sum(reID_valid) <= 5 
    relabel = fix(rand(n,1) * n + 1); reDL_valid = DL_valid(relabel);
    reMSX2_valid = MSX2_valid(relabel); reMSX3_valid = MSX3_valid(relabel);
    reID_valid = ID_valid(relabel); reMS_valid = MS_valid(relabel);
end
reL = L(relabel); reR = R(relabel); reindL = indL(relabel); reindR = indR(relabel); 
rebl1 = bl1(relabel,:); rebr1 = br1(relabel,:); reXobs = Xobs(relabel,:); rewrk_cov = wrk_cov(relabel,:);
reX1 = reXobs(:,1); reX2obs = reXobs(:,2); reX3obs = reXobs(:,3);
reZobs = Zobs(relabel,:);  reZobstemp = reZobs; reZobstemp(not(reDL_valid)) = nan; reWobs = Wobs(relabel,:);
remlcc = fminunc(@(para) loglik_ic(para, pdim, m, reWobs, rebl1, rebr1, reindL, reindR, rho0, reID_valid), [bet0; ph01], optfmin);
betimp = remlcc(1:pdim); epimp = cumsum(exp(remlcc((pdim+1):(pdim+m+1))));
exitaft = 0;
while exitaft <= 1
lyc = random('Uniform',0,1.5);
[gamaft,~,exitaft] = fminunc(@(para) loglik_aft(para, px, m, reXobs, exp(reZobs), reDL_valid, exp(dtlim)), [zeros(px,1); -lyc*ones(m+1,1)], optfmin);
end
regam = gamaft(1:px); resxi = reZobs - reXobs * regam;
[kmf,kmx] = ecdf(resxi,'censoring',reDL_valid==0,'function','survivor');
kmx = kmx(2:end); kmf = kmf(2:end);
linMod_X2 = fitlm([reX1,reX3obs,reZobstemp], reX2obs); linMod_Z = fitlm(reXobs, reZobstemp);
logisMod_X3 = mnrfit([reX1,reX2obs,reZobstemp], reX3obs+1);
X2obs_imp = X2obs; X2obs_imp(isnan(X2obs_imp)) = mean(X2obs,'omitnan'); MSX2_validimp = MSX2_valid;
X3obs_imp = X3obs; X3obs_imp(isnan(X3obs_imp)) = mode(X3obs); MSX3_validimp = MSX3_valid;
Xobs_imp = [X1, X2obs_imp, X3obs_imp]; Zobs_imp = Zobs; DL_validimp = DL_valid; MSZ_validimp = MSZ_valid;
for ii = 1:n
    stvtimes = 0;
    while ~DL_validimp(ii)
        stvtimes = stvtimes + 1;
        res_org_b = Zobs_imp(ii) - Xobs_imp(ii,:) * regam;
        omega_probrow = (res_org_b <= kmx') .* kmf'; omega_probb = omega_probrow./sum(omega_probrow,2);
        imput_xi_b = mnrnd(1,omega_probb);
        if any(imput_xi_b)
            imput_xi_b_index = find(imput_xi_b);
            if imput_xi_b_index == size(kmx,1)
                imput_xi_b_real = kmx(imput_xi_b_index);
            else
                imput_xi_b_real = random('Uniform', kmx(imput_xi_b_index), kmx(imput_xi_b_index+1));
            end
            ztem = Xobs_imp(ii,:) * regam + imput_xi_b_real;  ux = random('Uniform',0,1,1,1);
            ss = exp(- indL(ii) .* Gx((bl1(ii,:)*epimp) .* exp([Xobs_imp(ii,:), ztem] * betimp), rho0)) - ...
                 indR(ii) .* exp(- Gx((br1(ii,:)*epimp) .* exp([Xobs_imp(ii,:), ztem] * betimp), rho0));
            if ux <= ss
                Zobs_imp(ii,:) = ztem; DL_validimp(ii) = true;
            end
            if stvtimes > maxstvtimes
                Zobs_imp(ii,:) = NaN; DL_validimp(ii) = true;
            end
        else
            Zobs_imp(ii,:) = NaN; DL_validimp(ii) = true;
        end
    end
    stvtimes = 0;
    while ~MSZ_validimp(ii)
        stvtimes = stvtimes + 1; ztem2 = random(linMod_Z, Xobs_imp(ii,:));
        ux = random('Uniform',0,1,1,1);
        ss = exp(- indL(ii) .* Gx((bl1(ii,:)*epimp) .* exp([X1(ii,:), X2obs_imp(ii,:), X3obs_imp(ii,:), ztem2] * betimp), rho0)) - ...
            indR(ii) .* exp(- Gx((br1(ii,:)*epimp) .* exp([X1(ii,:), X2obs_imp(ii,:), X3obs_imp(ii,:), ztem2] * betimp), rho0));
        if ux <= ss
            Zobs_imp(ii,:) = ztem2; MSZ_validimp(ii) = true;
        end
        if stvtimes > maxstvtimes
            Zobs_imp(ii,:) = NaN; MSZ_validimp(ii) = true;
        end
    end
    stvtimes = 0;
    while ~MSX2_validimp(ii)
        if ~isnan(Zobs_imp(ii,:))
            stvtimes = stvtimes + 1; x2tem = random(linMod_X2, [X1(ii,:),X3obs_imp(ii,:),Zobs_imp(ii,:)]);
            ux = random('Uniform',0,1,1,1);
            ss = exp(- indL(ii) .* Gx((bl1(ii,:)*epimp) .* exp([X1(ii,:), x2tem, X3obs_imp(ii,:), Zobs_imp(ii,:)] * betimp), rho0)) - ...
                 indR(ii) .* exp(- Gx((br1(ii,:)*epimp) .* exp([X1(ii,:), x2tem, X3obs_imp(ii,:), Zobs_imp(ii,:)] * betimp), rho0));
            if ux <= ss
                X2obs_imp(ii,:) = x2tem; MSX2_validimp(ii) = true;
            end
            if stvtimes > maxstvtimes
                X2obs_imp(ii,:) = NaN; MSX2_validimp(ii) = true;
            end
        else
            X2obs_imp(ii,:) = NaN; MSX2_validimp(ii) = true;
        end
    end
    stvtimes = 0;
    while ~MSX3_validimp(ii)
        x3tem = [];
        if ~any([isnan(Zobs_imp(ii,:)), isnan(X2obs_imp(ii,:))])
            stvtimes = stvtimes + 1; x3tem = find(mnrnd(1, mnrval(logisMod_X3, [X1(ii,:),X2obs_imp(ii,:),Zobs_imp(ii,:)]))==1)-1; 
            if isempty(x3tem)
                 X3obs_imp(ii,:) = NaN; MSX3_validimp(ii) = true;
            else
            ux = random('Uniform',0,1,1,1);
            ss = exp(- indL(ii) .* Gx((bl1(ii,:)*epimp) .* exp([X1(ii,:), X2obs_imp(ii,:), x3tem, Zobs_imp(ii,:)] * betimp), rho0)) - ...
                 indR(ii) .* exp(- Gx((br1(ii,:)*epimp) .* exp([X1(ii,:), X2obs_imp(ii,:), x3tem, Zobs_imp(ii,:)] * betimp), rho0));
            if ux <= ss
                X3obs_imp(ii,:) = x3tem; MSX3_validimp(ii) = true;
            end
            if stvtimes > maxstvtimes
                X3obs_imp(ii,:) = NaN; MSX3_validimp(ii) = true;
            end
            end
        else
            X3obs_imp(ii,:) = NaN; MSX3_validimp(ii) = true;
        end
    end
end
[mlimpcc,~,exitflagimp,~,~,heslimpcc] = fminunc(@(para) loglik_ic(para, pdim, m, [X1, X2obs_imp, X3obs_imp, Zobs_imp], bl1, br1, indL, indR, rho0, ones(n,1)), [0*bet0; ph01], optfmin);
summary_impcc(nofimp, :) = mlimpcc(1:pdim);
diaggimpcc = diag(pinv(heslimpcc)); dihimpcc = sqrt(abs(diaggimpcc(1:pdim)));
sumvar_impcc(nofimp, :) = dihimpcc(1:pdim);
end
summarymicc(rep, :) = mean(summary_impcc, 1);
Bvarcc = std(summary_impcc(:,1:pdim)); Wvarcc = mean(sumvar_impcc.^2, 1);
sumvarmicc(rep, :) = sqrt(Wvarcc + (1 + 1/Nimp) * (Bvarcc.^2));                 
rep
end

%%
rs = nrep;
biasful = mean(summaryful, 1)' - bet0; sseful = std(summaryful)'; seeful = mean(sumvarful, 1)';
cpful = mean( ( repmat(bet0', rs, 1) >= summaryful - 1.96*sumvarful ) & ...
    ( repmat(bet0', rs, 1) <= summaryful + 1.96*sumvarful ) )';
Full = [bet0, biasful, sseful, seeful, cpful]

biashoc = mean(summaryhoc, 1)' - bet0; ssehoc = std(summaryhoc)'; seehoc = mean(sumvarhoc, 1)';
cphoc = mean( ( repmat(bet0', rs, 1) >= summaryhoc - 1.96*sumvarhoc ) & ...
    ( repmat(bet0', rs, 1) <= summaryhoc + 1.96*sumvarhoc ) )';
Hoc = [bet0, biashoc, ssehoc, seehoc, cphoc]
biascc = mean(summarycc, 1)' - bet0; ssecc = std(summarycc)'; seecc = mean(sumvarcc, 1)';
cpcc = mean( ( repmat(bet0', rs, 1) >= summarycc - 1.96*sumvarcc ) & ...
    ( repmat(bet0', rs, 1) <= summarycc + 1.96*sumvarcc ) )';
CC = [bet0, biascc, ssecc, seecc, cpcc]
biasmicc = mean(summarymicc, 1)' - bet0; ssemicc = std(summarymicc)'; seemicc = mean(sumvarmicc, 1)';
cpmicc = mean( ( repmat(bet0', rs, 1) >= summarymicc - 1.96*sumvarmicc ) & ...
    ( repmat(bet0', rs, 1) <= summarymicc + 1.96*sumvarmicc ) )';
MI = [bet0, biasmicc, ssemicc, seemicc, cpmicc]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function b = bern(j,p,l,u,t)
    b = mycombnk(p,j)*(((t-l)/(u-l)).^j).*((1-(t-l)/(u-l)).^(p-j));
end

function b = bern_partial(j,p,l,u,t)
    if j == 0
        b_part = p * ((1-(t-l)/(u-l)).^(p-1)) .* (-1/(u-l));
    elseif j == p
        b_part = p * (((t-l)/(u-l)).^(p-1)) .* (1/(u-l));
    else
        b_part = j * (1/(u-l)) * (((t-l)/(u-l)).^(j-1)).*((1-(t-l)/(u-l)).^(p-j)) + ...
              (p-j) * (-1/(u-l)) * (((t-l)/(u-l)).^j).*((1-(t-l)/(u-l)).^(p-j-1));
    end
    b = mycombnk(p,j) * b_part;
end

function output = Gx(x, rho)
    output = ((1 + x).^rho - 1)/rho;
%     output = log(1 + rho * x) ./ rho;
end

function output = loglik_aft(para, pz, m, X, T, delC, dllim)
gam1 = para(1:pz); phi1 = para((pz+1):(pz+m+1)); ep1 = cumsum(exp(phi1));
bernl = max(min(T.*exp(X * -gam1))-0.02, 0); bernr = max(dllim.*exp(X * -gam1)) + 0.02;
bt1 = zeros(size(T,1),(m+1)); pbt1 = bt1;
for i = 0:m
   bt1(:,(i+1)) = bern(i,m,bernl,bernr,T.*exp(X * -gam1));
   pbt1(:,(i+1)) = bern_partial(i,m,bernl,bernr,T.*exp(X * -gam1));
end
lam1 = pbt1 * ep1; Lam1 = bt1 * ep1;
llik = (lam1 .* exp(X * -gam1)).^delC .* exp(-Lam1);
output = - sum(log(llik), 'omitnan');
end

function output = loglik_ic(para, pz, m, Z, bl, br, indL, indR, r, valid)
bet1 = para(1:pz); phi1 = para((pz+1):(pz+m+1)); ep1 = cumsum(exp(phi1));
sl1 = exp(- indL .* Gx(exp(Z * bet1) .* (bl * ep1), r));
sr1 = exp(- Gx(exp(Z * bet1) .* (br * ep1), r));
llik = sl1 - indR .* sr1;
output = - sum(valid.*log(llik), 'omitnan');
end

function m=mycombnk(n,k)
if nargin < 2, error('Too few input parameters'); end
s = isscalar(k) & isscalar(n);
if (~s), error('Non-scalar input'); end
ck = k > n;
if (ck), error('Invalid input'); end
z = k >= 0 & n > 0;
if (~z), error('Negative or zero input'); end
m = factorial(n)/(factorial(k)*factorial(n-k));
end 
 




