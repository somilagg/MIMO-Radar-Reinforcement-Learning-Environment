%--------------------------
% Synthesis of C from R
%--------------------------
[U_init, verify] = Unitary(Nt);
C_past = zeros(Nt,Nt);
C = [];
j = 1;
U_new = U_init;
error = Inf;
for idx =1:10;
    C = [];
    % Step-1
    Z = R^(1/2)*U_new;
    for i=1:Nt
        C(:,i) = sqrt(Pt/(Nt*norm(Z(:,i))^2))*Z(:,i);
    end
    % Step-2
    [U,S,V] = svd(C*R^(1/2));
    U_new = V*U';
end

function [U,verify]= Unitary(n)
% generate a random complex matrix
X = complex(rand(n),rand(n))/sqrt(2);
% factorize the matrix
[Q,R] = qr(X);
R = diag(diag(R)./abs(diag(R)));
% unitary matrix
U = Q*R;
% verification
verify = U*U';
end