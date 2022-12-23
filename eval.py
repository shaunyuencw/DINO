import os

FOLDER_TO_EVAL = 'checkpoints/ship_experiment'
DATA_FOLDER = '../data/ship_200train_500val_200test'
DATA_NAME = '200train'
ARCH_TYPE = 'vitbase'


checkpoints = [f.name for f in os.scandir(FOLDER_TO_EVAL) if f.is_file() and f.name.endswith('.pth')]
checkpoints.sort()

checkpoints.remove('checkpoint0200.pth')

# checkpoints = ['checkpoint0200.pth']
# print(checkpoints)

#checkpoints = ['checkpoint0180.pth', 'checkpoint0200.pth']

for checkpoint in checkpoints:
    epochs = int(checkpoint.replace('.pth', '').replace('checkpoint', ''))
    LOG_FILE_NAME = f'dino_{ARCH_TYPE}_logs_{epochs}.txt'
    print(LOG_FILE_NAME)

    if os.path.exists('checkpoint.pth.tar'):
        os.remove('checkpoint.pth.tar')

    command = f"python eval_linear.py --arch vit_base  --pretrained_weights {FOLDER_TO_EVAL}/{checkpoint} --num_workers 16 \
--val_freq 5 --batch_size_per_gpu 256 --patch_size 16 --epochs 100 --data_path {DATA_FOLDER} --num_labels 4"

    print(f"Running '{command}'")
    os.system(command)
    
    if not os.path.exists(f"{FOLDER_TO_EVAL}/{DATA_NAME}"):
        os.mkdir(f"{FOLDER_TO_EVAL}/{DATA_NAME}")
    
    if os.path.exists('log.txt'):
        os.system(f"cp log.txt {FOLDER_TO_EVAL}/{DATA_NAME}/{LOG_FILE_NAME}")
        os.remove("log.txt")
    else:
        print("Something went wrong, there is no log...")