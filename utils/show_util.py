from configs import config

def showToolName():
    describe_image = '\n' + \
    '  _   _                      _ _______ ______ \n' + \
    ' | \ | |                    | |__   __|  ____| \n' + \
    ' |  \| | ___ _   _ _ __ __ _| |  | |  | |__ \n' + \
    ' | . ` |/ _ \ | | | \'__/ _` | |  | |  |  __| \n' + \
    ' | |\  |  __/ |_| | | | (_| | |  | |  | |____ \n' + \
    ' |_| \_|\___|\__,_|_|  \__,_|_|  |_|  |______| \n' + \
    'version ' + str(config.version_num) + '\n\n'
    print(describe_image)

def showTrainParams(data_path):
    print('\nParameters configuration\n'
        '====================================System settings========================================\n'
        '  [Setting] Input data used to train model = [ ' + str(data_path) + ' ]\n'
        '  [Setting] Whether to use LTR, TIR terminal features = [ ' + str(config.use_terminal) + ' ]\n'
        '  [Setting] Whether to use TSD features = [ ' + str(config.use_TSD) + ' ]\n'
        '  [Setting] Whether to use domain features = [ ' + str(config.use_domain) + ' ]\n'
        '  [Setting] Whether to use 5-bp terminal ends features = [ ' + str(config.use_ends) + ' ]\n'
        '  [Setting] Input thread num = [ ' + str(config.threads) + ' ]\n'
        '  [Setting] The k-mer size used to convert internal sequences to k-mer frequency features = [ ' + str(config.internal_kmer_sizes) + ' ]\n'
        '  [Setting] The k-mer size used to convert terminal sequences to k-mer frequency features = [ ' + str(config.terminal_kmer_sizes) + ' ]\n'
        '  [Setting] The number of CNN convolutional layers = [ ' + str(config.cnn_num_convs) + ' ]\n'
        '  [Setting] The number of filters in each CNN convolutional layer = [ ' + str(config.cnn_filters_array) + ' ]\n'
        '  [Setting] The kernel size in each of CNN convolutional layer = [ ' + str(config.cnn_kernel_sizes_array) + ' ]\n'
        '  [Setting] The threshold of CNN Dropout = [ ' + str(config.cnn_dropout) + ' ]\n'
        '  [Setting] The batch size in training model = [ ' + str(config.batch_size) + ' ]\n'
        '  [Setting] The number of epochs in training model = [ ' + str(config.epochs) + ' ]\n'
        '  [Setting] Whether to use breakpoint training = [ ' + str(config.use_checkpoint) + ' ]'
    )

def showTestParams(data_path, model_path):
    print('\nParameters configuration\n'
          '====================================System settings========================================\n'
          '  [Setting] Input data to be classified = [ ' + str(data_path) + ' ]\n'
          '  [Setting] Input the path of trained model, absolute path = [ ' + str(model_path) + ' ]\n'
          '  [Setting] Input thread num = [ ' + str(config.threads) + ' ]'
    )