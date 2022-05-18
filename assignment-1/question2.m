% Direct rewards
global r0 = [-5 -4 -3 -2 -1 0 1 2 3 4 5 0];
global r1 = [ 0  0  0  0  0 0 0 0 0 0 0 0];
% Discount factor
global alpha = 0.95;

% Transition probabilities after taking action 0: p^0(i,j)
function prob = get_p0 (rho)
  prob = zeros(12);
  prob(1,1) = 1-rho;
  prob(1,2) = rho;
  prob(11,10) = 1-rho;
  prob(11,11) = rho;
  for i = 2:10
    prob(i,i-1) = 1-rho;
    prob(i,i+1) = rho;
  endfor
  prob(12,12) = 1;
end

% Transition probabilities after taking action 1: p^1(i,j)
global p1 = zeros(12);
p1(:,12) = 1;

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

function transitions = policyToTransitions (policy, p0)
  global p1
  transitions = zeros (12,12);
  for i = 1:11
    if policy(i) == 0
      transitions(i,:) = p0(i,:);
    elseif policy(i) == 1
      transitions(i,:) = p1(i,:);
    end
  end
  % Always stay in the final state
  transitions(12,:) = p1(12,:);
end

function policy = valuesToPolicy (values, p0)
  global alpha
  policy = zeros(1,12);
  for i = 1:11
    % Take the argmax to determine the optimal action.
    % The direct reward and weighted value for choosing action 1 is always 0.
    [_ ix] = max([(i-6) + alpha .* (p0(i,:) * values), 0]);
    policy(i) = ix - 1;
  end
  policy(12) = 1; % keep taking action 1 once out (does not really matter)
end

%%% Policy Evaluation %%%
function v = policyEvaluation (f, p0)
  global alpha

  % Derive the transitions and rewards given the policy
  Pf = policyToTransitions(f, p0);
  r = policyToRewards(f);
  
  % Solve the functional equation.
  v = (eye(12) - alpha * Pf) \ r';
end

%%% Policy iteration %%%
function [v, f] = policyIteration (rho)
  p0 = get_p0(rho);
  
  % The initial policy is to stay invested if X_t >= 0.
  f = [1 1 1 1 1 0 0 0 0 0 0 1];
  f_prev = NaN(1,12);

  while any(f != f_prev)
    % Evaluate the policy
    v = policyEvaluation(f, p0);
    
    % Perform a step of policy improvement
    f_prev = f; % store to check for optimality
    f = valuesToPolicy(v, p0);
  end
end

%%% Value iteration (successive approximation) %%%
function [v, f, steps] = valueIteration (rho)
  global alpha
  p0 = get_p0(rho);

  v_prev = zeros(12,1);
  v = zeros(12,1);

  M = 1000;
  m = 0;
  eps = 0.00001;
  steps = 0; % count number of steps till convergence

  while M - m > eps
    steps = steps + 1;

    for i = 1:11
      v(i) = max(0, (i-6) + alpha .* (p0(i,:) * v_prev));
    end

    M = max(v - v_prev);
    m = min(v - v_prev);
    
    v_prev = v;
  end

  f = valuesToPolicy(v, p0);
 end

 
%%% Comparison of different values of rho %%%

N = 20;
policies1 = zeros(N+1, 12);
policies2 = zeros(N+1, 12);
values1 = zeros(N+1, 12);
values2 = zeros(N+1, 12);
valueIterationSteps = zeros(N+1,1);
valuesf0 = zeros(N+1, 12);
rhos = zeros(N+1);

for n = 1:N+2
  rho = (1/N) * (n-1);
  rhos(n) = rho;
  
  % Optimal rewards, policies
  [v1, f1] = policyIteration(rho);
  [v2, f2, steps] = valueIteration(rho);
  valueIterationSteps(n) = steps;
  policies1(n,:) = f1;
  policies2(n,:) = f2;
  values1(n,:) = v1;
  values2(n,:) = v2;
  
  % Rewards for f0
  valuesf0(n,:) = policyEvaluation(f0, get_p0(rho));
endfor

% print column with rhos, then column with policies, then column with values
function printPolicies (rhos, policies, values)
  for i = 1 : length(policies) - 1
    valuestr = "";
    for j = 1:12
      str = num2str(values(i,j), 3);
      valuestr = [valuestr sprintf("%6s ", str)];
    end
    fprintf( '%4d %s  %s \n', rhos(i), mat2str(policies(i,:)), valuestr );
  end
end

fprintf("\nPolicy Iteration:\n")
printPolicies(rhos, policies1, values1)
fprintf("\nValue Iteration:\n")
printPolicies(rhos, policies2, values2)
fprintf("\nRequired number of iteration:\n %s \n", mat2str(valueIterationSteps))


f0 = [1 1 1 1 1 0 0 0 0 0 0 1];
fprintf("\nPolicy Evaluation for f0:\n")
for i = 1 : N + 1
    valuestr = "";
    for j = 1:12
      str = num2str(valuesf0(i,j), 3);
      valuestr = [valuestr sprintf("%6s ", str)];
    end
    fprintf( '%4d %s \n', rhos(i), valuestr );
  end
