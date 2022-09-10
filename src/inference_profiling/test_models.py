#Importing all the packages
import logging
import torch
import sys

#batch=int(input("Enter the batch size"))
timesteps=3
bands=18

#logging.basicConfig(filename=args.model_name +'.log', filemode='a',level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger()
logger.info("Starting the logger.")

batch_size=int(sys.argv[1])
model_name=sys.argv[2]
try: 
    model=torch.jit.load(model_name+".pt").eval()

    dummy_data=torch.randn((batch_size,timesteps,bands))
    #print(dummy_data)
    #print(dummy_data.shape)
    logger.info("\n For batch size of "+str(batch_size))

    with torch.no_grad():
        print("The predictions on the model are ",model(dummy_data))


except ValueError:
    print("The given model name could not be found in the model repository.")
    logging.error("The given model is not present in the model repository.")