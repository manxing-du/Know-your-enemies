# train ddpg with a random market model
random_om = False
# 1: click, 2: pctr 3: win_click=1, win_lose=pctr
reward = 2

# camps = ['1458', '2259', '2261', '2821', '2997', '3358', '3386', '3427', '3476']
# linear agent bidding ratio
lin_ratio = 50.
# q_function = indi, concat, integ
q_func = 'indi'

# Agent number: N = DDPG + (N-1) * Lin
agent_num = 3
# noise to add to pctr to ensure many linear agents are not just duplicates.
if agent_num > 2:
    noise = 0.001
else:
    noise = 0
# seed (use at least 3 different seeds, up to 10)
seed = 0

# compete mode: 1: with ipinyou market price  2: without ipinyou market price
compete_mode = 2

