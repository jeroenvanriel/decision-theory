P0 = [
0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ;
0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ;
0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ;
0. , 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ;
0. , 0. , 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ;
0. , 0. , 0. , 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. , 0. ;
0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0.5, 0. , 0. , 0. , 0. ;
0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0.5, 0. , 0. , 0. ;
0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0.5, 0. , 0. ;
0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0.5, 0. ;
0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0. ;
0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1.
]

P1 = [
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. ;
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.
]

# choose to stay if X_t >= 0
Pf0 = zeros (12,12)
for i = 1:5
  Pf0(i,:) = P1(i,:)
endfor
for i = 6:11
  Pf0(i,:) = P0(i,:)
 endfor
 # always stay in the terminal state
 Pf0(12,:) = P1(12,:)

r = [0 0 0 0 0 0 1 2 3 4 5 0]
alpha = 0.95

x = (eye(12) - alpha * Pf0) \ r'


v_prev = zeros(12,1)
v = zeros(12,1)

M = 1000
m = 0
eps = 0.1

while M - m > eps
  for i = 1:12
    v(i) = max(0, (i-6) + alpha .* (P0(i,:) * v_prev))
  end

  M = max(v - v_prev)
  m = min(v - v_prev)
  
  v_prev = v
end