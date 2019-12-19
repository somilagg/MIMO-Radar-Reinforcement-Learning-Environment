function [R] = weightDesign()

%---------
% READ INPUT FILES
%---------
fileID = fopen('input_param.txt','r');
formatSpec = '%f';
A = fscanf(fileID, formatSpec);
fclose(fileID);

fileID2 = fopen('curr_angle.txt','r');
B = fscanf(fileID2, formatSpec);
fclose(fileID2);

%---------
% STORE VALUES FROM INPUT PARAMETERS
%---------
Pt = A(1);
angle_j = B;
Nt = A(3);
start_a = A(4);
end_a = A(5);

%---------
% INITIAL CALCULATIONS
%---------
deg2rad = pi/180;
vTheta_rad = angle_j*deg2rad;
vTheta_rad;
phi = sin(vTheta_rad);
for idx=1:length(B)
    TxArray(:,idx) = exp(1i*pi*phi(idx)*[0:1:(Nt-1)]);
end

%---------
% CVX CALCULATIONS
%---------

cvx_begin sdp quiet
    variable R(Nt,Nt) hermitian toeplitz
    variable eta
    maximize eta
    R >= 0;
    eta >= 0;
    trace(R)<=Pt;
    for idx=1:length(B)
        real(TxArray(:,idx)'*R*TxArray(:,idx)) >= eta;
    end
cvx_end

%---------
% SAVE MATLAB STATUS
%---------
mstatus = fopen('matlab_status.txt', 'w');
fprintf(mstatus, cvx_status);
fclose(mstatus);

%---------
% FINAL CALCULATIONS
%---------
[U,S,V] = svd(R);
C = U*S^(1/2)*V';

%---------
% SAVE OUTPUT FILES
%---------
save('C');
FileData = load('C.mat');
csvwrite('C.csv', FileData.C);