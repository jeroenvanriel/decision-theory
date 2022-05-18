% Transition probabilities after taking action 0: p^0(i,j)
global rho = 0.5;
global P0 = [
1-rho, rho  , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0. , 0. ;
1-rho, 0.   , rho  , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0. , 0. ;
0.   , 1-rho, 0.   , rho  , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0. , 0. ;
0.   , 0.   , 1-rho, 0.   , rho  , 0.   , 0.   , 0.   , 0.   , 0.   , 0. , 0. ;
0.   , 0.   , 0.   , 1-rho, 0.   , rho  , 0.   , 0.   , 0.   , 0.   , 0. , 0. ;
0.   , 0.   , 0.   , 0.   , 1-rho, 0.   , rho  , 0.   , 0.   , 0.   , 0. , 0. ;
0.   , 0.   , 0.   , 0.   , 0.   , 1-rho, 0.   , rho  , 0.   , 0.   , 0. , 0. ;
0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1-rho, 0.   , rho  , 0.   , 0. , 0. ;
0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1-rho, 0.   , rho  , 0. , 0. ;
0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1-rho, 0.   , rho, 0. ;
0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1-rho, rho, 0. ;
0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0. , 1.
];

% Transition probabilities after taking action 1: p^1(i,j)
global P1 = [
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
];

% Direct rewards
global r0 = [-5 -4 -3 -2 -1 0 1 2 3 4 5 0];
global r1 = [ 0  0  0  0  0 0 0 0 0 0 0 0];
% Discount factor
global alpha = 0.95;

function rewards = policyToRewards (policy)
  global r0, global r1
  for i = 1:12
    if (policy(i) == 0)
      rewards(i) = r0(i);
    elseif (policy(i) == 1)
      rewards(i) = r1(i);
    end
  end
end

function transitions = policyToTransitions (policy)
  global P0, global P1
  transitions = zeros (12,12);
  for i = 1:11
    if policy(i) == 0
      transitions(i,:) = P0(i,:);
    elseif policy(i) == 1
      transitions(i,:) = P1(i,:);
    end
  end
  % Always stay in the final state
  transitions(12,:) = P1(12,:);
end

function policy = valuesToPolicy (values)
  global alpha, global P0
  policy = zeros(1,12);
  for i = 1:11
    % Take the argmax to determine the optimal action.
    % The direct reward and weighted value for choosing action 1 is always 0.
    [_ ix] = max([(i-6) + alpha .* (P0(i,:) * values), 0]);
    policy(i) = ix - 1;
  end
  policy(12) = 1; % keep taking action 1 once out (does not really matter)
end


%%% Policy iteration %%%
% The initial policy is to stay invested if X_t >= 0.
f = [1 1 1 1 1 0 0 0 0 0 0 1];
f_prev = NaN(1,12);

while any(f != f_prev)
  % Derive the transitions and rewards given the policy
  Pf = policyToTransitions(f);
  r = policyToRewards(f);
  
  % Solve the functional equation.
  v = (eye(12) - alpha * Pf) \ r';
  
  % Perform a step of policy improvement
  f_prev = f; % store to check for optimality
  f = valuesToPolicy(v);
end

v
f

%%% Value iteration (successive approximation) %%%
v_prev = zeros(12,1);
v = zeros(12,1);

M = 1000;
m = 0;
eps = 0.00001;

while M - m > eps
  for i = 1:11
    v(i) = max(0, (i-6) + alpha .* (P0(i,:) * v_prev));
  end

  M = max(v - v_prev);
  m = min(v - v_prev);
  
  v_prev = v;
end

v
valuesToPolicy(v)
