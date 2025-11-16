import torch

"""
Distances between multiple agents
"""

# for i, agent in enumerate(agents):
#     other_agents = [a for j, a in enumerate(agents) if j != i]
#     # other_agents = next(aid for aid in agents if aid != agent)
#     print(f"Ego Agent: {agent}, Other Agents: {other_agents}")

# """
# tensor repeat
# """
# a = torch.ones(2,3)
# b = a.repeat(4, 1)
# c = a.repeat(4, 2)
# print(f"Original tensor: {a}")
# print(f"- shape: {a.shape}")
# print("Repeated tensor:", b)
# print(f"- shape: {b.shape}")
# print("Repeated tensor:", c)
# print(f"- shape: {c.shape}")

# """
# Indexing with list of indexes
# """
# robots = {
#     "drone_1": torch.ones(2, 3),
#     "drone_2": torch.ones(2, 3) * 2,
#     "drone_3": torch.ones(2, 3) * 3,
# }
# other_agents = ["drone_2", "drone_3"]
# robots[other_agents[0]]

"""Tensor Prod"""
a1 = torch.tensor([
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 1],
])
b1 = torch.prod(a1, dim=1)
a2 = torch.ones(2,3,1,1)
# b2 = torch.prod(a2[:,0,:,0], dim=2)
b3 = torch.prod(a2[:,0,:,0], dim=1)
b4 = torch.prod(a2[:,0,:,0], dim=0)
# print(f"Input tensor:\n{a1}")
# print(f"Prod over dim 1:\n{b1}")
# print(f"- shape: {b1.shape}")
print(f"Input tensor:\n{a2}")
print(f"- shape: {a2.shape}")
# print(f"Prod over dim 2:\n{b2}")
# print(f"- shape: {b2.shape}")
print(f"Prod over dim 1:\n{b3}")
print(f"- shape: {b3.shape}")
print(f"Prod over dim 0:\n{b4}")
print(f"- shape: {b4.shape}")