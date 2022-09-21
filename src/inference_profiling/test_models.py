# Measured the performance of pytorch models on a local environment for various batch sizes.
# Varied the batch sizes in factors of 2 from 2 to 1024
# for models: "Namibia_North_2020" and "Malawi_2020_September".The execution time
# on the average was fund to be ~1s. The peak process memory was negligible and
# peak process cpu ranged between 8-70%.
# Additional details can be found in the "local_model_tests.csv".

# Importing all the packages
import torch
import sys

# batch=int(input("Enter the batch size"))
timesteps = 3
bands = 18

batch_size = int(sys.argv[1])
model_name = sys.argv[2]

try:
    model = torch.jit.load(model_name + ".pt").eval()

    dummy_data = torch.randn((batch_size, timesteps, bands))
    # print(dummy_data)
    # print(dummy_data.shape)
    print("\n For batch size of " + str(batch_size))

    # Making predictions on the data
    with torch.no_grad():
        print("The predictions on the model are ", model(dummy_data))


except ValueError:
    print("The given model name could not be found in the model repository.")
