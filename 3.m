% TensorVAR: INM(old)
clear all

% compile();
H = load('../Artemasov64ant3kmhlong.mat');
% H = load('Channels/H_uma32.mat');
H = H.H_BB_DL;
% H_uma32 = cat(5, H_uma32{:});
%%
H_BB_DL2 = permute(H,[5, 2, 4, 3, 1]);
% H_uma32 = permute(H_uma32,[5, 2, 4, 3, 1]);
num_threads             = 48;

ue_idx                  = 1;

% Enable or disable quantization by factors tx, sc and tti
quantize_rx             = 0;
quantize_tx             = 0;
quantize_sc             = 0;
quantize_tti            = 1;

tti_order               = 0;    % 2 order of extrapolation, order>=0
sc_format               = 0;    % 1 sc representation format, 0 - ALS-based, 1 - find phase
tx_format               = 0;    % 1 tx representation format, 0 - ALS-based, 1 - find phase
rx_format               = 0;    % rx representation format, 0 - ALS-based, 1 - find phase

rank                    = 8;    % 6 originally

tti_cut                 = 1;
n_tx_onedim_1           = 8;    % 8 originally for 32
n_tx_onedim_2           = 4;    % 2 originally for 32

firstALS_iterations     = 50;   % 50 originally
tailALS_iterations      = 10;

extfft_multiplier       = 30;
extfft_resolution       = 1.0;

SNR                     = 30.0;


% n_ue = 40;
n_rx=4;
n_tti=2;
n_sc=50;
n_tx=32;
% f1_freq = 8;
% total_tti = 4;

%num_factors = floor(total_tti/f1_freq);

% Uest1{1} = zeros(4,rank,num_factors,n_ue);
% Uest1{2} = zeros(8,rank,num_factors,n_ue);
% Uest1{3} = zeros(4,rank,num_factors,n_ue);
% Uest1{4} = zeros(48,rank,num_factors,n_ue);
% Uest1{5} = zeros(2,rank,num_factors,n_ue);
% 
% Uest2{1} = zeros(4,rank,num_factors,n_ue);
% Uest2{2} = zeros(8,rank,num_factors,n_ue);
% Uest2{3} = zeros(4,rank,num_factors,n_ue);
% Uest2{4} = zeros(48,rank,num_factors,n_ue);
% Uest2{5} = zeros(2,rank,num_factors,n_ue);

% for jj=1:10
jj = 1;
channel_index = jj;

% for int_tti=1:num_factors
int_tti = 1;
ue_cnt=jj;
ue_idx = ue_cnt;

%% to prediction
H_tmp(:,1:4,:,:)=H_BB_DL2(jj, :, 1+(int_tti-1)*f1_freq:4+(int_tti-1)*f1_freq, :, :);
H_tmp2 = H_tmp;clc

%%

real_part = randn(n_rx, n_tti, n_sc, n_tx);
imag_part = randn(n_rx, n_tti, n_sc, n_tx);

test = complex(real_part, imag_part);
H_uma32_ = permute(H_uma32, [2, 4, 3, 1, 5]);

% set a user
user_num = 1;
H_uma32_tcb = H_uma32_(:, :, :, :, user_num);
%%
res_ten = mex_fun(H_uma32_tcb, ue_idx-1,num_threads,n_rx,n_tti,n_sc,n_tx,quantize_rx,...
    quantize_tx,quantize_sc,quantize_tti,tti_order,sc_format,tx_format,...
    rx_format,rank,tti_cut,n_tx_onedim_1,n_tx_onedim_2,firstALS_iterations,...
    tailALS_iterations,extfft_multiplier,extfft_resolution,SNR);

% end
% end
%% Channel recovery

H_out(:) = res_ten(:);

dimentionality = 5;

% ******************************************************** First polarization
total=zeros(4*2*50*32,1);

shift = 0;
ii=0;

for r=0:(rank-1)
    Dvoiki(ii+1,r+1,1:4) = H_out(r*4+1:r*4+4);

    Uest1{1}(:,r+1,int_tti,ue_cnt) = squeeze( Dvoiki(ii+1,r+1,1:4) );
end
shift = shift + rank*4;

ii=1;
for r=0:(rank-1)
    Dvoiki(ii+1,r+1,1:8) = H_out(shift + r*8+1:shift + r*8+8);

    Uest1{2}(:,r+1,int_tti,ue_cnt) = squeeze( Dvoiki(ii+1,r+1,1:8) );
end
shift = shift + rank*8;

ii=2;
for r=0:(rank-1)
    Dvoiki(ii+1,r+1,1:4) = H_out(shift + r*4+1:shift + r*4+4);

    Uest1{3}(:,r+1,int_tti,ue_cnt) = squeeze( Dvoiki(ii+1,r+1,1:4) );
end
shift = shift + rank*4;

ii=3;
for r=0:(rank-1)
    Dvoiki(ii+1,r+1,1:48) = H_out(shift + r*48+1:shift + r*48+48);

    Uest1{4}(:,r+1,int_tti,ue_cnt) = squeeze( Dvoiki(ii+1,r+1,1:48) );
end
shift = shift + rank*48;

ii=4;
for r=0:(rank-1)
    Dvoiki(ii+1,r+1,1:2) = H_out(shift + r*2+1:shift + r*2+2);

    Uest1{5}(:,r+1,int_tti,ue_cnt) = squeeze( Dvoiki(ii+1,r+1,1:2) );
end

%recovery of original channel
for r=1:rank
tmp2=kron( squeeze(Dvoiki(2,r,1:8)), squeeze(Dvoiki(1,r,1:4)) );
kr_rank = zeros(size(tmp2));
kr_rank(1:size(tmp2)) = tmp2;
for ii=3:dimentionality
    if ii==3
    tmp3=squeeze(Dvoiki(ii,r,1:4));
    end
    if ii==4
    tmp3=squeeze(Dvoiki(ii,r,1:48));
    end
    if ii==5
    tmp3=squeeze(Dvoiki(ii,r,1:2));
    end

    tmp = kron(tmp3,kr_rank(:));
    kr_rank(1:size(tmp)) = tmp;
end
total(:) = total(:)+squeeze(kr_rank(:));

end
% 
%     total2(:,:,:,:) = reshape(total(:),[4,32,48,2]);
%     
% %    total2(:,:,:,:,ue_cnt) = sqrt(10.) * 1.41421 * total2(:,:,:,:,ue_cnt);
%     total2(:,:,:,:) = 1.41421 * total2(:,:,:,:);
% 
% 
%     % ********************************  Second polarozation
%     total=zeros(4*48*32*2,1);
%     polar_shift = 528;
% 
%     shift = 0;
%     ii=0;
%     for r=0:(rank-1)
%         Dvoiki(ii+1,r+1,1:4) = H_out(polar_shift+r*4+1:polar_shift+r*4+4);
%         
%         Uest2{1}(:,r+1,int_tti,ue_cnt) = squeeze( Dvoiki(ii+1,r+1,1:4) );
%     end
%     shift = shift + rank*4;
% 
%     ii=1;
%     for r=0:(rank-1)
%         Dvoiki(ii+1,r+1,1:8) = H_out(polar_shift+shift + r*8+1:polar_shift+shift + r*8+8);
%         
%         Uest2{2}(:,r+1,int_tti,ue_cnt) = squeeze( Dvoiki(ii+1,r+1,1:8) );
%     end
%     shift = shift + rank*8;
% 
%     ii=2;
%     for r=0:(rank-1)
%         Dvoiki(ii+1,r+1,1:4) = H_out(polar_shift+shift + r*4+1:polar_shift+shift + r*4+4);
% 
%         Uest2{3}(:,r+1,int_tti,ue_cnt) = squeeze( Dvoiki(ii+1,r+1,1:4) );
%     end
%     shift = shift + rank*4;
% 
%     ii=3;
%     for r=0:(rank-1)
%         Dvoiki(ii+1,r+1,1:48) = H_out(polar_shift+shift + r*48+1:polar_shift+shift + r*48+48);
%         
%         Uest2{4}(:,r+1,int_tti,ue_cnt) = squeeze( Dvoiki(ii+1,r+1,1:48) );
%     end
%     shift = shift + rank*48;
% 
%     ii=4;
%     for r=0:(rank-1)
%         Dvoiki(ii+1,r+1,1:2) = H_out(polar_shift+shift + r*2+1:polar_shift+shift + r*2+2);
%         
%         Uest2{5}(:,r+1,int_tti,ue_cnt) = squeeze( Dvoiki(ii+1,r+1,1:2) );
%     end
% 
%     % recovery of original channel
%     for r=1:rank
%     tmp2=kron( squeeze(Dvoiki(2,r,1:8)), squeeze(Dvoiki(1,r,1:4)) );
%     kr_rank = zeros(size(tmp2));
%     kr_rank(1:size(tmp2)) = tmp2;
%     for ii=3:dimentionality
%         if ii==3
%         tmp3=squeeze(Dvoiki(ii,r,1:4));
%         end
%         if ii==4
%         tmp3=squeeze(Dvoiki(ii,r,1:48));
%         end
%         if ii==5
%         tmp3=squeeze(Dvoiki(ii,r,1:2));
%         end
% 
%         tmp = kron(tmp3,kr_rank(:));
%         kr_rank(1:size(tmp)) = tmp;
%     end
%     total(:) = total(:)+squeeze(kr_rank(:));
%     
%     end
% 
%     total2(:,:,:,:) = reshape(total(:),[4,32,48,2]);
%     
% %    total2(:,:,:,:,ue_cnt) = sqrt(10.) * 1.41421 * total2(:,:,:,:,ue_cnt);
%     total2(:,:,:,:) = 1.41421 * total2(:,:,:,:);
% 
% 
% 
%     
% end
% %end of recovery of original channel
% 
% 
% end
%% Visualization
figure()

plot(1:48,squeeze(H_tmp(1,1,1:48,33)),1:48,squeeze(total2(1,1,1:48,1)*res_ten(1057) ));

grid
zoom
