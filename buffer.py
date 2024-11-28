import torch

# Load the checkpoint
state_dict = torch.load("minifasnet_model.pth", map_location=torch.device("cpu"))

# Adjust the weights of the Linear layer
state_dict["linear.weight"] = torch.nn.init.xavier_uniform_(torch.empty(128, 4608))
state_dict["linear.bias"] = torch.zeros(128)

# Save the adjusted checkpoint
torch.save(state_dict, "adjusted_minifasnet_model.pth")
