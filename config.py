# Dictionary storing network parameters.
params = {
    'batch_size': 128,# Batch size.
    'num_epochs': 100,# Number of epochs to train for.
    'learning_rate_G': 1e-3, # Learning rate for Generator.
    'learning_rate_D': 2e-4,
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 25,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'QuickDraw',
    'classes' : ["cup", "door", "circle", "hexagon", "t-shirt", "smiley face", "candle", "eye", "star", "pants"],
    'num_img' : 6000 # Set to -1 for all
}# Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!