# import pandas as pd
#
# df = pd.read_csv('demo_app/dataset/small_test_dataset/test_50_unlabeled.csv')
# print(df)


# import torch
#
# model_data = torch.load("demo_app/model/global_model.pt", map_location="cpu")
#
# print(len(model_data))  # if list
# print(model_data[0].shape)  # first layer weight
# print(len(model_data))
#
# import torch
# from model.gat_residual_bn import StrongResidualGAT
#
# checkpoint = torch.load("demo_app/model/global_model.pt", map_location="cpu")
#
# # if isinstance(checkpoint, list):
# #     print("FL ndarray format")
# # else:
# print(checkpoint["gat.lin.weight"].shape)
