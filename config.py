# Dictionary storing network parameters.
params = {
    'batch_size': 128,# Batch size.
    'num_epochs': 30,# Number of epochs to train for.
    'learning_rate_G': 1e-3, # Learning rate for Generator.
    'learning_rate_D': 2e-4,
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 3,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'QuickDraw',
    #'classes' : ["mushroom", "door", "hand", "key", "t-shirt", "smiley face", "candle", "eye", "star", "pants"],
    'classes' : ['light bulb', 'leaf', 'penguin ', 'sailboat', 'car', 'donut', 'star', 'hourglass', 'washing machine', 'mountain'],
    'num_img' : 12000 # Set to -1 for all
}# Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!

# Set appropriate hyperparameters depending on the dataset used.
# The values given in the InfoGAN paper are used.
# num_z : dimension of incompressible noise.
# num_dis_c : number of discrete latent code used.
# dis_c_dim : dimension of discrete latent code.
# num_con_c : number of continuous latent code used.
if(params['dataset'] == 'MNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2
elif(params['dataset'] == 'SVHN'):
    params['num_z'] = 124
    params['num_dis_c'] = 4
    params['dis_c_dim'] = 10
    params['num_con_c'] = 4
elif(params['dataset'] == 'CelebA'):
    params['num_z'] = 128
    params['num_dis_c'] = 10
    params['dis_c_dim'] = 10
    params['num_con_c'] = 0
elif(params['dataset'] == 'FashionMNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2
# Input parametre til vores. Vi skal lege med disse på et tidspunkt. skal vist summere op til 74?
elif(params['dataset'] == 'QuickDraw'):
    params['num_dis_c'] = 1
    params['dis_c_dim'] = len(params['classes']) 
    params['num_con_c'] = 2 #Hyperparameter. Beskriver antallet af kontiuerte værdier vi kan lege med.
    params['num_z'] = 74-params['dis_c_dim']-params['num_con_c'] #Let the len of the noise sum up to 74 depending on other params.