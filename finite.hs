import Data.Ord (comparing)
import Data.List (maximumBy, minimumBy, unfoldr)
import System.Random (RandomGen, mkStdGen, uniformR)

-- general types

type Prob  = Float

type State = Int
type Action = Int
type Value = Float

data Step = Step { ac :: Action, rw :: Value, st :: State } deriving (Show)

states :: [State]
actions :: [Action]
reward :: Action -> State -> Value
p :: Action -> State -> State -> Prob

-- problem instance definition

states = [0,1]
actions = [1,2]

reward 1 0 = 1
reward 2 0 = 0
reward 1 1 = 2
reward 2 1 = 2

p a i j = case (a,i,j) of 
  (1,0,0) -> 1/2; (1,0,1) -> 1/2;
  (1,1,0) -> 2/3; (1,1,1) -> 1/3;
  (2,0,0) -> 1/4; (2,0,1) -> 3/4;
  (2,1,0) -> 1/3; (2,1,1) -> 2/3;

q 0 = 2
q 1 = 1

-- general formulas

g :: (State -> Value) -> State -> Action -> Value
g v i a = reward a i + (sum $ [(p a i j) * (v j) | j <- states])

v :: Int -> State -> Value
v n i
  | n == 0 = q i
  | otherwise = minimum $ map (g v' i) actions
  where v' = v (n-1)

argmin f xs = minimumBy (comparing f) xs
argmax f xs = maximumBy (comparing f) xs

type Iteration = Int
type Rule = Iteration -> State -> Action

f :: Iteration -> State -> Action
f n i = argmin (g v' i) actions
  where v' = v (n-1)

-- simulation

pureGen = mkStdGen 42
roll :: RandomGen g => g -> (Prob, g)
roll = uniformR (0,1)         
rolls :: RandomGen g => g -> [Prob]
rolls = unfoldr (Just . roll)
randomSequence :: [Prob]
randomSequence = rolls pureGen

-- sample from a probability distribution
-- effectively returns the index of the chosen probability element
sample :: [Prob] -> Prob -> Int
sample (p:ps) u
  | u < p = 0
  | otherwise = 1 + sample ps (u-p)

-- TODO: support strategies
nextAction :: Rule -> Iteration -> State -> Action
nextAction f t i = f t i
nextValue a i u = reward a i
nextStateD a i = map (\j -> (p a i) j) states
nextState a i u = sample (nextStateD a i) u

nextStep :: Rule -> Iteration -> State -> Float -> Step
nextStep f t i u = Step a v j
  where a = nextAction f t i
        v = nextValue a i u
        j = nextState a i u

-- simulate n steps starting in i and iteration t
-- n encodes "steps left"
-- t encodes "current step" to chose the correct function
steps :: Rule -> Iteration -> State -> Int -> [Prob] -> [Step]
steps f t i 0 us = []
steps f t i n (u:us) = s : (steps f (t+1) (st s) (n-1) us)
  where s = (nextStep f t i u)

stepsSample :: Rule -> State -> Int -> [Step]
stepsSample f i n = steps f 1 i n randomSequence

stepsSamples 0 us = []
stepsSamples n (u1:u2:us) = (steps f 1 0 2 [u1,u2]) : (stepsSamples (n-1) us)

totalReward :: [Step] -> Value
totalReward xs = foldl folder 0 xs
  where folder b s = b + rw s

averageReward f i n = (/ fromIntegral n) $ totalReward $ stepsSample f i n

-- assume for now that we can use a stationary rule ???
-- start in state 0
i0 = 0 :: State
fStar = f n
  where n=1

