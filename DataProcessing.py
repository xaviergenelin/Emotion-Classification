import pandas  as pd
import tarfile as tf
import os

# Variables
# ----------------------------------------------------------
path      = 'data/empatheticdialogues.tar.gz' # Data path
metadata  = False                                           # Print metadata


# Fixed
# ----------------------------------------------------------
directory = path.replace('.tar.gz', '/')


# Checks
# ----------------------------------------------------------
if metadata:
    print(os.path.exists(path))
    print(os.path.dirname(path))
    print(path.replace('.tar.gz', '/'))
    print()

pd.set_option('display.max_columns', None)


# Extract All Data
# ----------------------------------------------------------
print('Extracting Data...')

# Open File
file = tf.open(path)
#
## Extract File
file.extractall(os.path.dirname(path))
#
## Close File
file.close()


# Process Data
# ----------------------------------------------------------
print('Processing Data...\n')

for file in sorted(os.listdir(directory)):
    if file.endswith('.csv'):
        # Determine file name        
        print('------------------')
        print(f'File: {file}')
        
        name = os.path.splitext(file)[0]

        
        # Process data and clean unnecessary whitespace and _comma_ instances
        data        = pd.read_csv(directory + '/' + file, usecols=['prompt','context']).drop_duplicates().reset_index(drop=True)
        data.prompt = data.prompt.str.replace('_comma_', ', ').str.replace('  ', ' ')
        data.rename(columns={'context':'label'}, inplace=True)
        
        
        # Data information
        print(f'Rows: {len(data)}\n')
        
        if metadata:
            print(data.head())
            print(data.tail())
        
        
        # Process train and valid separately from test        
        if   name == 'train': train = data
        elif name == 'valid': valid = data
        elif name == 'test':  test  = data
        else:
            print('Error')
            

# Training Data
# ----------------------------------------------------------
xtrain = pd.concat([train['prompt'], valid['prompt']], axis=0).reset_index(drop=True).to_frame()
ytrain = pd.concat([train['label'],  valid['label']],  axis=0).reset_index(drop=True).to_frame()

print('------------------')
print('Data: xtrain (train + valid)')
print(f'Rows: {len(xtrain)}\n')
print(xtrain.head())
print(xtrain.tail())

print('\n------------------')
print('Data: ytrain (train + valid)')
print(f'Rows: {len(ytrain)}\n')
print(ytrain.head())
print(ytrain.tail())

print('\n------------------')
print(f'Statistics: ytrain')
labels = ytrain.groupby('label').size().to_frame(name='count')
labels['percentage'] = (labels['count'] / labels['count'].sum()) * 100

print(labels)
print(f'\nPercentage total: {labels["percentage"].sum()}')


# Test Data
# ----------------------------------------------------------
xtest  = test['prompt'].to_frame(name='prompt')
ytest  = test['label'].to_frame(name='label')

print('\n------------------')
print('Data: xtest')
print(f'Rows:  {len(xtest)}\n')
print(xtest.head())
print(xtest.tail())

print('\n------------------')
print('Data: ytest')
print(f'Rows:  {len(ytest)}\n')
print(ytest.head())
print(ytest.tail())

print('\n------------------')
print(f'Statistics: ytest')
labels = ytest.groupby('label').size().to_frame(name='count')
labels['percentage'] = (labels['count'] / labels['count'].sum()) * 100

print(labels)
print(f'\nPercentage total: {labels["percentage"].sum()}')

train = xtrain.join(ytrain)
test = xtest.join(ytest)
xtrain.to_csv('data/train.csv')
xtest.to_csv('data/test.csv')






































# import pandas  as pd
# import tarfile as tf
# import os

# # Variables
# # ----------------------------------------------------------
# path      = 'data/empatheticdialogues.tar.gz' # Data path
# metadata  = False                                           # Print metadata


# # Fixed
# # ----------------------------------------------------------
# directory = path.replace('.tar.gz', '/')


# # Checks
# # ----------------------------------------------------------
# if metadata:
#     print(os.path.exists(path))
#     print(os.path.dirname(path))
#     print(path.replace('.tar.gz', '/'))
#     print()

# pd.set_option('display.max_columns', None)


# # Extract All Data
# # ----------------------------------------------------------
# print('Extracting Data...')

# # Open File
# file = tf.open(path)
# #
# ## Extract File
# file.extractall(os.path.dirname(path))
# #
# ## Close File
# file.close()


# # Process Data
# # ----------------------------------------------------------
# print('Processing Data...\n')

# for file in sorted(os.listdir(directory)):
#     if file.endswith('.csv'):
#         # Determine file name        
#         print('------------------')
#         print(f'File: {file}')
        
#         name = os.path.splitext(file)[0]

        
#         # Process data and clean unnecessary whitespace and _comma_ instances
#         data        = pd.read_csv(directory + '/' + file, usecols=['prompt', 'context']).drop_duplicates().reset_index(drop=True)
#         data.prompt = data.prompt.str.replace('_comma_', ', ').str.replace('  ', ' ')
#         data.rename(columns={'context':'label'}, inplace=True)
        
        
#         # Data information
#         print(f'Rows: {len(data)}\n')
        
#         if metadata:
#             print(data.head())
#             print(data.tail())
        
        
#         # Process train and valid separately from test        
#         if   name == 'train': train = data
#         elif name == 'valid': valid = data
#         elif name == 'test':  test  = data
#         else:
#             print('Error')
            

# # Training Data
# # ----------------------------------------------------------
# # train = pd.concat([train['prompt'], valid['prompt']], axis=0).reset_index(drop=True).to_frame()
# # ytrain = pd.concat([train['label'],  valid['label']],  axis=0).reset_index(drop=True).to_frame()

# train = pd.concat([train, valid, test], axis=0).reset_index(drop=True)
# # ytrain = pd.concat([train['label'],  valid['label'], test['label']],  axis=0).reset_index(drop=True).to_frame()


# print('------------------')
# print('Data: train (train + valid)')
# print(f'Rows: {len(train)}\n')
# print(train.head())
# print(train.tail())

# # print('\n------------------')
# # print('Data: ytrain (train + valid)')
# # print(f'Rows: {len(ytrain)}\n')
# # print(ytrain.head())
# # print(ytrain.tail())

# print('\n------------------')
# print(f'Statistics: ytrain')
# labels = train.groupby('label').size().to_frame(name='count')
# labels['percentage'] = (labels['count'] / labels['count'].sum()) * 100

# print(labels)
# print(f'\nPercentage total: {labels["percentage"].sum()}')


# # # Test Data
# # # ----------------------------------------------------------
# # xtest  = test['prompt'].to_frame(name='prompt')
# # ytest  = test['label'].to_frame(name='label')

# # print('\n------------------')
# # print('Data: xtest')
# # print(f'Rows:  {len(xtest)}\n')
# # print(xtest.head())
# # print(xtest.tail())

# # print('\n------------------')
# # print('Data: ytest')
# # print(f'Rows:  {len(ytest)}\n')
# # print(ytest.head())
# # print(ytest.tail())

# # print('\n------------------')
# # print(f'Statistics: ytest')
# # labels = ytest.groupby('label').size().to_frame(name='count')
# # labels['percentage'] = (labels['count'] / labels['count'].sum()) * 100

# # print(labels)
# # print(f'\nPercentage total: {labels["percentage"].sum()}')

# # train = train.join(ytrain)
# # test = xtest.join(ytest)
# train.to_csv('data/train.csv')
# # xtest.to_csv('data/test.csv')





# # AARON'S JUNK
# # import pandas  as pd
# # import tarfile as tf
# # import os

# # # Variables
# # # ----------------------------------------------------------
# # path      = 'data/empatheticdialogues.tar.gz' # Data path
# # metadata  = False                                           # Print metadata


# # # Fixed
# # # ----------------------------------------------------------
# # directory = path.replace('.tar.gz', '/')


# # # Checks
# # # ----------------------------------------------------------
# # if metadata:
# #     print(os.path.exists(path))
# #     print(os.path.dirname(path))
# #     print(path.replace('.tar.gz', '/'))
# #     print()

# # pd.set_option('display.max_columns', None)


# # # Extract All Data
# # # ----------------------------------------------------------
# # print('Extracting Data...')

# # # Open File
# # file = tf.open(path)
# # #
# # ## Extract File
# # file.extractall(os.path.dirname(path))
# # #
# # ## Close File
# # file.close()


# # # Process Data
# # # ----------------------------------------------------------
# # print('Processing Data...\n')

# # for file in sorted(os.listdir(directory)):
# #     if file.endswith('.csv'):
# #         # Determine file name        
# #         print('------------------')
# #         print(f'File: {file}')
        
# #         name = os.path.splitext(file)[0]

        
# #         # Process data and clean unnecessary whitespace and _comma_ instances
# #         data        = pd.read_csv(directory + '/' + file, usecols=['prompt']).drop_duplicates().reset_index(drop=True)
# #         data.prompt = data.prompt.str.replace('_comma_', ', ').str.replace('  ', ' ')
# #         data.rename(columns={'context':'label'}, inplace=True)
        
        
# #         # Data information
# #         print(f'Rows: {len(data)}\n')
        
# #         if metadata:
# #             print(data.head())
# #             print(data.tail())
        
        
# #         # Process train and valid separately from test        
# #         if   name == 'train': train = data
# #         elif name == 'valid': valid = data
# #         elif name == 'test':  test  = data
# #         else:
# #             print('Error')
            

# # # Training Data
# # # ----------------------------------------------------------
# # # train = pd.concat([train['prompt'], valid['prompt']], axis=0).reset_index(drop=True).to_frame()
# # # ytrain = pd.concat([train['label'],  valid['label']],  axis=0).reset_index(drop=True).to_frame()

# # train = pd.concat([train[['prompt', 'utterance_idx', 'conv_id', 'speaker_idx', 'label']], valid[['prompt', 'utterance_idx', 'conv_id', 'speaker_idx', 'label']], test[['prompt', 'utterance_idx', 'conv_id', 'speaker_idx', 'label']]], axis=0).reset_index(drop=True)
# # # ytrain = pd.concat([train['label'],  valid['label'], test['label']],  axis=0).reset_index(drop=True).to_frame()


# # print('------------------')
# # print('Data: train (train + valid)')
# # print(f'Rows: {len(train)}\n')
# # print(train.head())
# # print(train.tail())

# # # print('\n------------------')
# # # print('Data: ytrain (train + valid)')
# # # print(f'Rows: {len(ytrain)}\n')
# # # print(ytrain.head())
# # # print(ytrain.tail())

# # print('\n------------------')
# # print(f'Statistics: ytrain')
# # labels = train.groupby('label').size().to_frame(name='count')
# # labels['percentage'] = (labels['count'] / labels['count'].sum()) * 100

# # print(labels)
# # print(f'\nPercentage total: {labels["percentage"].sum()}')


# # # # Test Data
# # # # ----------------------------------------------------------
# # # xtest  = test['prompt'].to_frame(name='prompt')
# # # ytest  = test['label'].to_frame(name='label')

# # # print('\n------------------')
# # # print('Data: xtest')
# # # print(f'Rows:  {len(xtest)}\n')
# # # print(xtest.head())
# # # print(xtest.tail())

# # # print('\n------------------')
# # # print('Data: ytest')
# # # print(f'Rows:  {len(ytest)}\n')
# # # print(ytest.head())
# # # print(ytest.tail())

# # # print('\n------------------')
# # # print(f'Statistics: ytest')
# # # labels = ytest.groupby('label').size().to_frame(name='count')
# # # labels['percentage'] = (labels['count'] / labels['count'].sum()) * 100

# # # print(labels)
# # # print(f'\nPercentage total: {labels["percentage"].sum()}')

# # # train = train.join(ytrain)
# # # test = xtest.join(ytest)
# # train.to_csv('data/train.csv')
# # # xtest.to_csv('data/test.csv')


