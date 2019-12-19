function [R] = generateY()

%---------
% READ INPUT FILES
%---------
fileID = fopen('input_param.txt','r');
formatSpec = '%f';
A = fscanf(fileID, formatSpec);

fileID2 = fopen('curr_angle.txt','r');
timestep1 = fscanf(fileID2, formatSpec);
fclose(fileID2);

%---------
% STORE VALUES FROM INPUT PARAMETERS
%---------
Pt = A(1);
angle_j = A(2);
L = A(2);
Nt = A(3);
start_a = A(4);
end_a = A(5);
G = A(6);
Tmax = A(7);
r = A(8);
version = A(9);
fclose(fileID);
timestep2 = timestep1(1);

%---------
% INITIAL CALCULATIONS
%---------
inc = (end_a-start_a)/angle_j;
vAngle = start_a:inc:(end_a+inc);
deg2rad = pi/180;
vTheta_rad = vAngle*deg2rad;
Nr = Nt;

%---------
% INPUT NEW LOCATIONS IF V2 OR V3
%---------
locations = [];
if (version == 3) || (version == 2)
    for c = 1:r
        c_2 = [A(9+c)];
        locations = [locations, c_2];
    end
end

%---------
% C MATRIX GENERATION
%---------
C = zeros(Nt,Nt);
if timestep2 ~= 0
    fileID3 = fopen('q_vals.txt','r');
    ii  = 1;
    while ~feof(fileID3)
        c_mat(ii, :) = str2num(fgets(fileID3));
        ii = ii + 1;
    end
    fclose(fileID3);
    for row=1:Nt
        for col=1:Nt
            C(row, col) = c_mat((row-1) * Nt + col);
        end
    end
    C
else
    temp_in = (randn(Nt,Nt)+j*randn(Nt,Nt))/sqrt(2);
    for idx=1:Nt
        C(idx,:) = sqrt(Pt/Nt)*(temp_in(idx,:)/norm(temp_in(idx,:)));
    end
end

%---------
% CALCULATIONS
%---------
for idx_theta=1:L
    phi = sin(vTheta_rad(idx_theta));
    TxArray(idx_theta,:) = exp(j*pi*phi*[0:1:(Nt-1)]);
    RxArray(idx_theta,:) = exp(j*pi*phi*[0:1:(Nr-1)]);
    h(idx_theta,:) = kron((C'*TxArray(idx_theta,:)'),RxArray(idx_theta,:)');
end

% MAKE INITIAL ENVIRONMENT
%---------
target = sqrt(10/2) .* (randn(r, 1) + j * randn(r, 1));
alpha = zeros(L, G);

%---------
% CHOOSE TARGET LOCATIONS BASED ON MODE
%---------
if (version == 3) || (version == 2)
    target_temp = locations;
else
    target_temp = [1 425 665 800];
    % (4, 42) (7, 2) (10, 77) (14, 6)
    alpha(5, 43) = 10;
    alpha(8, 3) = 10;
    alpha(11, 78) = 10;
    alpha(15, 7) = 10;
end

%---------
% MAKE NOISE AND COMPLETE ENVIRONMENT
%---------
noise = sqrt(0.01/2)*(randn(Nt*Nr,L,G)+j*randn(Nt*Nr,L,G));
y_rec = zeros(Nt*Nr,L,G);
for l=1:L
    for g=1:G
        y_rec(:,l,g)=(alpha(l,g).*h(l,:))'+noise(:,l,g);
    end
end

%---------
% SAVE OUTPUT FILES
%---------
save('target_temp');
FileDataTT = load('target_temp.mat');
csvwrite('target_temp.csv', FileDataTT.target_temp);

save('y_rec');
FileData = load('y_rec.mat');
csvwrite('y_rec.csv', FileData.y_rec);