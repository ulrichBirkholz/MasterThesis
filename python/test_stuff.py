import torch

# Two arrays of tensors
array1 = [torch.tensor([1, 2]), torch.tensor([3, 4])]
array2 = []

# Concatenate arrays of tensors along the first dimension
combined = torch.cat(array1 + array2, dim=0)

#print(combined)

# Suppose you have the following tensor of labels
labels = torch.tensor([1, 2, 2, 3, 1, 4, 5, 4, 3, 2, 1])

# To find out how many unique labels there are, you can use:
unique_labels = torch.unique(labels)

print(unique_labels)  # prints: tensor([1, 2, 3, 4, 5])

# To find out the count of unique labels, use:
num_unique_labels = unique_labels.size(0)

print(num_unique_labels)  # prints: 5
