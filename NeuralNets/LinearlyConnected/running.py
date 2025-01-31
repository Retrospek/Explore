from mock_data import *

data = getIRIS()
dataset = Dataset(data, shuffle=True, batch_size=32)

dataloader = DataLoader(dataset)
batched_data = dataloader.data

print(f"First DataLoader Dimension: {len(batched_data)}")
print(f"Second DataLoader Dimension: {len(batched_data[0])}")
print(f"Third DataLoader Dimension: {len(batched_data[0][0])}")
print(f"X DataLoader Dimension: {len(batched_data[0][0][0])}")
print(f"Y DataLoader Dimension: {batched_data[0][1][0]}")